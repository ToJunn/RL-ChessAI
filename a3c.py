import os, time, math, random, multiprocessing as mp
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from env import ChessEnv
from typing import Tuple

# 1. IMPORT ActorCriticNet TỪ FILE MODEL.PY
# Giả sử bạn đã đặt file ActorCriticNet trong model.py
from model import ActorCriticNet 

# Hàm to_tensor phải được cập nhật để chấp nhận device
def to_tensor(obs_np, device='cpu'):
    # (8,8,12) -> (1,12,8,8). Chuyển sang device
    # ChessEnv trả về (8,8,12) (H,W,C), cần permute thành (C,H,W)
    return th.from_numpy(obs_np).float().permute(2,0,1).unsqueeze(0).to(device)

class SharedAdam(th.optim.Adam):
    """
    Adapter cho Adam để chia sẻ trạng thái optimizer giữa các worker A3C.
    """
    def __init__(self, params, lr=1e-4, **kwargs):
        super().__init__(params, lr=lr, **kwargs)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = th.zeros(1, dtype=th.long)
                state['exp_avg'] = th.zeros_like(p.data)
                state['exp_avg_sq'] = th.zeros_like(p.data)
                # Dữ liệu trạng thái optimizer phải được chia sẻ bộ nhớ cho A3C
                state['exp_avg'].share_memory_() 
                state['exp_avg_sq'].share_memory_()
                state['step'].share_memory_()

def worker(rank, global_net, optimizer, cfg, writer_dir, device):
    """
    Hàm worker cho A3C. Mỗi worker chạy một môi trường riêng biệt.
    """
    # Đảm bảo mỗi worker chỉ dùng 1 luồng CPU cho các phép tính numpy/python
    th.set_num_threads(1) 
    env = ChessEnv()
    
    # Tạo local_net và chuyển sang device (GPU nếu được chỉ định)
    local_net = ActorCriticNet()
    local_net.load_state_dict(global_net.state_dict())
    local_net.to(device) 
    
    writer = SummaryWriter(os.path.join(writer_dir, f"worker_{rank}"))
    total_step = 0
    
    while True:
        obs, info = env.reset()
        buffer_obs, buffer_actions, buffer_rewards, buffer_values, buffer_logps = [], [], [], [], []
        
        # 1. THU THẬP KINH NGHIỆM (Rollout)
        for t in range(cfg['t_max']):
            # Chuyển input về device (GPU/CPU)
            x = to_tensor(obs, device=device) 
            
            # Tính toán trên local_net (trên device)
            logits, value = local_net(x) 
            
            # Xử lý mask và sampling trên CPU/Device
            mask = info.get("action_mask")
            # Đưa logits về CPU nếu cần xử lý mask phức tạp (thao tác numpy/list)
            logits_on_cpu = logits.squeeze(0).cpu() 
            
            # Áp dụng Action Masking
            inf = -1e9
            mask_t = th.from_numpy(mask.astype(np.float32))
            logits_masked = logits_on_cpu + th.where(mask_t>0, th.zeros_like(logits_on_cpu), th.full_like(logits_on_cpu, inf))
            
            dist = th.distributions.Categorical(logits=logits_masked)
            action = int(dist.sample().item())
            logp = dist.log_prob(th.tensor(action))
            
            obs2, r, term, trunc, info = env.step(action)
            
            # Lưu trữ: values phải được đưa về CPU để tránh lỗi chia sẻ bộ nhớ
            buffer_obs.append(obs); 
            buffer_actions.append(action); 
            buffer_rewards.append(r); 
            buffer_values.append(value.cpu()); # Chuyển Value về CPU
            buffer_logps.append(logp.cpu()) # Chuyển LogP về CPU

            obs = obs2
            total_step += 1
            if term or trunc:
                break
                
        # 2. TÍNH TOÁN RETURNS (Monte Carlo/TD)
        R = 0.0
        if not (term or trunc):
            # Lấy giá trị Bootstrap V(s')
            x = to_tensor(obs, device=device)
            # Tính toán trên device
            with th.no_grad():
                _, R_t = local_net(x)
            R = R_t.item() # Lấy giá trị scalar về CPU
        
        returns = []
        R_val = R 
        for r in reversed(buffer_rewards):
            R_val = r + cfg['gamma'] * R_val
            returns.append(R_val)
        returns.reverse()
        
        # 3. TÍNH TOÁN LOSS VÀ BACKPROPAGATION
        
        # Chuyển dữ liệu đã thu thập về lại device để tính toán loss
        returns_t = th.tensor(returns, dtype=th.float32).to(device)
        values_t = th.stack([v.squeeze(0) for v in buffer_values]).detach().to(device)
        logps_t = th.stack(buffer_logps).to(device)
        
        adv = returns_t - values_t
        
        # Policy Loss: -(log_prob * Advantage)
        policy_loss = -(logps_t * adv.detach()).mean()
        # Value Loss: 0.5 * MSE
        value_loss = 0.5 * adv.pow(2).mean()
        
        # Tổng Loss (Entropy bị tắt)
        loss = policy_loss + cfg['vf_coef'] * value_loss - cfg['ent_coef'] * 0.0 
        
        optimizer.zero_grad()
        loss.backward()
        
        # 4. TỔNG HỢP GRADIENT VỀ GLOBAL NET (trên CPU Shared Memory)
        for global_p, local_p in zip(global_net.parameters(), local_net.parameters()):
            if local_p.grad is not None:  # <--- KIỂM TRA QUAN TRỌNG ĐÃ THÊM
                # Sao chép gradient từ local_net (device) về global_net (CPU)
                grad = local_p.grad.cpu()
                if global_p.grad is None:
                    global_p._grad = grad
                else:
                    global_p.grad = grad
        
        # Thực hiện cập nhật trọng số trên Global Net
        optimizer.step()
        
        # Đồng bộ hóa: local_net <- global_net
        local_net.load_state_dict(global_net.state_dict())
        
        # Ghi log
        writer.add_scalar("loss/total", loss.item(), total_step)
        
        if cfg.get('max_steps') and total_step >= cfg['max_steps']:
            break
            
    writer.close()

def train_a3c(num_workers=4, total_steps=100_000, log_dir="./logs/a3c", device="cpu"):
    """
    Hàm khởi tạo quá trình huấn luyện A3C.
    device: 'cpu' hoặc 'cuda' (để sử dụng GPU cho tính toán mạng nơ-ron)
    """
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"Bắt đầu huấn luyện A3C với {num_workers} workers. Device: {device}")

    # Global Net LUÔN ĐƯỢC ĐẶT TRÊN CPU để sử dụng Shared Memory
    global_net = ActorCriticNet()
    global_net.share_memory() 
    
    optimizer = SharedAdam(global_net.parameters(), lr=1e-4)
    
    cfg = {'t_max': 20, 'gamma': 0.99, 'vf_coef': 0.5, 'ent_coef': 0.01, 'max_steps': total_steps}
    
    procs = []
    for rank in range(num_workers):
        # Truyền device (cuda/cpu) cho worker
        p = mp.Process(target=worker, args=(rank, global_net, optimizer, cfg, log_dir, device))
        p.start()
        procs.append(p)
        
    for p in procs:
        p.join()
        
    # Lưu mô hình cuối cùng (được lưu từ CPU)
    th.save(global_net.state_dict(), os.path.join(log_dir, "a3c_final.pt"))
    print("Hoàn thành huấn luyện A3C.")
    return global_net

if __name__ == '__main__':
    # Ví dụ chạy: 
    # Nếu muốn dùng GPU: train_a3c(num_workers=8, device='cuda')
    # Nếu muốn dùng CPU: train_a3c(num_workers=4, device='cpu')
    pass