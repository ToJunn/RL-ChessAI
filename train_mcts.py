from __future__ import annotations

import os, math, time, random, argparse, collections
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

import numpy as np
import torch as th
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

# --- Local imports (v3) ---
# Đảm bảo bạn import từ file env đã được tối ưu

from envv4 import ChessEnv, move_index, PROMO_KINDS, BASE_ACTION_DIM

from modelv2 import ActorCriticV2, ExtractorConfig

# ===================== MCTS =====================
@dataclass
class MCTSConfig:
    n_sim: int = 200
    c_puct: float = 1.4
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    temperature: float = 1.0
    temperature_cutoff: int = 20   # after this ply, use argmax

class Node:
    __slots__ = ("P","N","W","Q","children","terminal","result")
    def __init__(self, prior: float = 0.0):
        self.P = float(prior)
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.children: Dict[int, Node] = {}
        self.terminal = False
        self.result: Optional[float] = None

class MCTS:
    def __init__(self, net: ActorCriticV2, device: str, cfg: MCTSConfig, action_dim: int):
        self.net = net
        self.device = device
        self.cfg = cfg
        self.action_dim = action_dim # NEW: Cần biết action space đầy đủ

    @th.no_grad()
    def _policy_value(self, env: ChessEnv) -> Tuple[np.ndarray, float]:
        obs = th.as_tensor(env._obs(), dtype=th.float32, device=self.device).unsqueeze(0)
        logits, value = self.net(obs)
        logits = logits.squeeze(0)
        # Lấy action mask (sẽ là 20480 nếu env config đúng)
        mask = th.from_numpy(env.get_action_mask()).to(self.device)
        logits = logits.masked_fill(mask <= 0, float("-inf"))
        policy = th.softmax(logits, dim=-1).detach().cpu().numpy()
        return policy, float(value.item())

    def run(self, env: ChessEnv) -> Tuple[np.ndarray, Dict[int, float]]:
        """Return visit distribution pi over action space and child Q-values."""
        root = Node()
        P, v = self._policy_value(env)
        
        # get_action_mask() giờ trả về 20480
        legal_idx = np.where(env.get_action_mask() > 0)[0]
        if len(legal_idx) == 0:
            # NEW: pi phải có kích thước action_dim
            pi = np.zeros(self.action_dim, dtype=np.float32)
            return pi, {}
        p_legal = P[legal_idx]
        p_legal = p_legal / (p_legal.sum() + 1e-8)

        # Dirichlet noise (giữ nguyên)
        if self.cfg.dirichlet_alpha > 0 and self.cfg.dirichlet_eps > 0:
            noise = np.random.dirichlet([self.cfg.dirichlet_alpha] * len(legal_idx))
            p_legal = (1 - self.cfg.dirichlet_eps) * p_legal + self.cfg.dirichlet_eps * noise

        for a, p in zip(legal_idx, p_legal):
            root.children[a] = Node(float(p))

        # Simulations
        for _ in range(self.cfg.n_sim):
            # _simulate không thay đổi, nó dùng env đã được copy
            scratch = self._simulate(env, root)

        # Visit distribution
        # NEW: visits phải có kích thước action_dim
        visits = np.zeros(self.action_dim, dtype=np.float32)
        for a, ch in root.children.items():
            visits[a] = ch.N
        if visits.sum() > 0:
            visits = visits / visits.sum()
        return visits, {a: ch.Q for a, ch in root.children.items()}

    def _simulate(self, env: ChessEnv, root: Node) -> float:
        # (Logic _simulate không thay đổi. Nó độc lập với action_dim,
        # chỉ cần env.step(action_int) hoạt động)
        
        # lightweight copy
        board_backup = env.board.copy()
        last_move_bak = env.last_move
        ply_bak = env._ply
        # Lấy obs_history stack (quan trọng!)
        obs_history_bak = collections.deque(env._obs_history, maxlen=env.ocfg.obs_stack_size)

        node = root
        path: List[Tuple[Node, int]] = []

        # Selection
        while True:
            if len(node.children) == 0:
                break
            # pick argmax over UCT
            total = sum(c.N for c in node.children.values()) + 1e-8
            best_a, best_score = None, -1e9
            for a, c in node.children.items():
                u = self.cfg.c_puct * c.P * math.sqrt(total) / (1 + c.N)
                score = c.Q + u
                if score > best_score:
                    best_score, best_a = score, a
            a = best_a
            # step action (env.step sẽ tự động cập nhật obs_history nội bộ)
            obs2, reward, term, trunc, info = env.step(int(a))
            path.append((node, a))
            if term or trunc:
                node = node.children.get(a, None)
                if node is None:
                    node = Node(0.0)
                node.terminal = True
                res = info.get("result")
                if res == "1-0":
                    node.result = +1.0 if (not env.board.turn) else -1.0
                elif res == "0-1":
                    node.result = -1.0 if (not env.board.turn) else +1.0
                else:
                    node.result = 0.0
                break
            node = node.children.get(a)
            if node is None:
                node = Node(0.0)
                break

        # Expansion (if not terminal)
        if not node.terminal:
            P, v = self._policy_value(env)
            legal_idx = np.where(env.get_action_mask() > 0)[0]
            if len(legal_idx) == 0:
                node.terminal = True
                node.result = 0.0
            else:
                p_legal = P[legal_idx]
                p_legal = p_legal / (p_legal.sum() + 1e-8)
                for a in legal_idx:
                    # Chú ý: P[a] đã là P từ 20480-dim policy
                    node.children[a] = Node(float(P[a]))
        else:
            v = node.result if node.result is not None else 0.0

        # Backup (giữ nguyên)
        value = float(v)
        flip = 1.0
        for (parent, action) in reversed(path):
            child = parent.children[action]
            child.N += 1
            child.W += flip * value
            child.Q = child.W / child.N
            flip = -flip

        # Restore env (RẤT QUAN TRỌNG: phải khôi phục cả obs_history)
        env.board = board_backup
        env.last_move = last_move_bak
        env._ply = ply_bak
        env._obs_history = obs_history_bak # Khôi phục obs stack
        env._attack_cache.clear(); env._attack_cache_ply_id = -1
        return value


# ===================== Replay Buffer =====================
class ReplayBuffer:
    # (Không thay đổi)
    def __init__(self, capacity: int = 200000):
        self.capacity = int(capacity)
        self.buf = collections.deque(maxlen=self.capacity)
    def push(self, obs: np.ndarray, pi: np.ndarray, z: float):
        # obs (N*C,H,W) và pi (20480,) đều là float32
        self.buf.append((obs.astype(np.float32), pi.astype(np.float32), float(z)))
    def sample(self, batch_size: int):
        idx = np.random.choice(len(self.buf), size=batch_size, replace=False)
        obs, pi, z = zip(*[self.buf[i] for i in idx])
        return np.stack(obs), np.stack(pi), np.asarray(z, dtype=np.float32)
    def __len__(self):
        return len(self.buf)
    def save_npz(self, path: str):
        try: np.savez_compressed(path, data=list(self.buf))
        except Exception: pass
    def load_npz(self, path: str):
        if not os.path.isfile(path): return
        try:
            data = np.load(path, allow_pickle=True)["data"].tolist()
            self.buf = collections.deque(data, maxlen=self.capacity)
        except Exception: pass


# ===================== Self-Play Worker =====================
def play_game_worker(net_state_dict, device: str, args, out_queue: mp.Queue, seed: int, worker_id: int):
    th.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    # NEW: Khởi tạo env với config mới
    obs_config = {
        "flip_perspective": args.flip_perspective,
        "obs_stack_size": args.obs_stack_size
    }
    dummy_env = ChessEnv(
        illegal_action_mode="reject",
        enable_promotion_actions=True,  # BẮT BUỘC
        enable_auto_queen=False,        # BẮT BUỘC
        observation_config=obs_config
    )
    
    ACTION_DIM = dummy_env.action_space.n # Sẽ là 20480

    net = ActorCriticV2(dummy_env.observation_space,
                        action_dim=ACTION_DIM, # NEW
                        extractor_cfg=ExtractorConfig(base_channels=args.base_channels,
                                                      n_res_blocks=args.n_res_blocks,
                                                      use_se=args.se,
                                                      dropout=args.dropout,
                                                      proj_dim=args.proj_dim)).to(device)
    net.load_state_dict(net_state_dict)
    net.eval()
    
    mcts = MCTS(net, device, MCTSConfig(n_sim=args.n_sim, c_puct=args.c_puct,
                                        dirichlet_alpha=args.dirichlet_alpha,
                                        dirichlet_eps=args.dirichlet_eps,
                                        temperature=args.temperature,
                                        temperature_cutoff=args.temperature_cutoff),
                action_dim=ACTION_DIM) # NEW: Pass action_dim

    # Worker sẽ chạy N game và TỰ THOÁT
    # Vòng lặp main() sẽ khởi động lại nó với model mới
    for g in range(args.games_per_worker):
        env = ChessEnv(
            illegal_action_mode="reject",
            enable_promotion_actions=True, # BẮT BUỘC
            enable_auto_queen=False,       # BẮT BUỘC
            observation_config=obs_config
        )
        
        states: List[np.ndarray] = []
        dists: List[np.ndarray] = []
        players: List[bool] = []
        ply = 0
        terminated = False
        while not terminated:
            pi, _ = mcts.run(env) # pi giờ là (20480,)
            
            # Temperature schedule
            if ply >= args.temperature_cutoff:
                a = int(np.argmax(pi))
                # NEW: Dùng ACTION_DIM
                pi = np.eye(ACTION_DIM, dtype=np.float32)[a]
            else:
                # NEW: Dùng ACTION_DIM
                a = int(np.random.choice(np.arange(ACTION_DIM), p=pi))

            states.append(env._obs()) # obs là (N*C, H, W)
            dists.append(pi)          # pi là (20480,)
            players.append(env.board.turn)

            _, _, term, trunc, info = env.step(a)
            terminated = term or trunc
            ply += 1
            if ply > args.max_game_plies:
                break

        # Terminal value (giữ nguyên)
        res = info.get("result", "1/2-1/2") if isinstance(info, dict) else "1/2-1/2"
        if res == "1-0": z_white = 1.0
        elif res == "0-1": z_white = -1.0
        else: z_white = 0.0
        
        for s, pi_val, stm_is_white in zip(states, dists, players):
            z = z_white if stm_is_white else -z_white
            out_queue.put((s, pi_val, float(z)))
    
    # Worker tự thoát sau khi hoàn thành 'games_per_worker'


# ===================== Train Loop =====================

def loss_fn(policy_logits: th.Tensor, target_pi: th.Tensor, value: th.Tensor, target_z: th.Tensor,
            policy_coef=1.0, value_coef=1.0, reg_coef=1e-4) -> th.Tensor:
    # (Không thay đổi)
    logp = th.log_softmax(policy_logits, dim=-1)
    policy_loss = -(target_pi * logp).sum(dim=-1).mean()
    value_loss = th.mean((value - target_z) ** 2)
    reg = 0.0
    for p in policy_logits.parameters() if hasattr(policy_logits, 'parameters') else []:
        reg = reg + (p**2).sum()
    return policy_coef * policy_loss + value_coef * value_loss + reg_coef * 0.0


def evaluate(net: ActorCriticV2, device: str, n_episodes: int = 16, opponent: str = "random", obs_stack_size: int = 4) -> Tuple[float, float, float, float]:
    # (Class RandomOpp và HeurOpp giữ nguyên)
    class RandomOpp:
        def choose(self, board):
            legal = list(board.legal_moves)
            if not legal: return None
            mv = random.choice(legal)
            # Phải trả về action_idx đầy đủ
            promo_id = PROMO_KINDS.index(mv.promotion) if mv.promotion in PROMO_KINDS else 0
            return (move_index(mv.from_square, mv.to_square) * 5) + promo_id
            
    class HeurOpp:
        PIECE = {1:1,2:3,3:3,4:5,5:9,6:0}
        def _m(self, board, mv):
            if board.is_capture(mv):
                if board.is_en_passant(mv): return 1
                t = board.piece_at(mv.to_square)
                return self.PIECE.get(t.piece_type,0) if t else 0
            return 0
        def choose(self, board):
            best, sc = None, -1e9
            legal_moves = list(board.legal_moves)
            if not legal_moves: return None
            for mv in legal_moves:
                s = self._m(board, mv)
                if s > sc: sc, best = s, mv
            if best is None: # Nếu không có nước đi nào tốt (ví dụ: toàn bộ là -1e9)
                best = random.choice(legal_moves)
            
            promo_id = PROMO_KINDS.index(best.promotion) if best.promotion in PROMO_KINDS else 0
            return (move_index(best.from_square, best.to_square) * 5) + promo_id
            
    opp = RandomOpp() if opponent=="random" else HeurOpp()
    
    # NEW: Khởi tạo env eval
    env = ChessEnv(
        illegal_action_mode="reject",
        enable_promotion_actions=True,
        enable_auto_queen=False,
        observation_config={"obs_stack_size": obs_stack_size} # Dùng stack_size
    )
    
    wins=draws=losses=0
    for _ in range(n_episodes):
        obs, info = env.reset(); done=False
        while not done:
            if env.board.turn:
                mask = env.get_action_mask() # (20480,)
                obs_t = th.as_tensor(obs, dtype=th.float32, device=device).unsqueeze(0)
                logits, _ = net(obs_t) # (1, 20480)
                logits = logits.squeeze(0)
                logits = logits.masked_fill(th.from_numpy(mask).to(device)<=0, float("-inf"))
                a = int(th.argmax(logits).item())
                obs, _, term, trunc, info = env.step(a)
            else:
                a = opp.choose(env.board) # Đã trả về 20480 idx
                if a is None:
                    legal = np.where(env.get_action_mask()>0)[0]
                    a = int(np.random.choice(legal)) if len(legal)>0 else 0
                obs, _, term, trunc, info = env.step(a)
            done = term or trunc
        r = info.get("result","1/2-1/2")
        if r=="1-0": wins+=1
        elif r=="0-1": losses+=1
        else: draws+=1
    S = (wins + 0.5*draws)/max(1,n_episodes)
    elo = None
    s = max(1e-6, min(1-1e-6, S))
    try: elo = -400.0*math.log10(1.0/s-1.0)
    except Exception: elo = None
    return S, wins/n_episodes, draws/n_episodes, elo if elo is not None else 0.0


def parse_args():
    ap = argparse.ArgumentParser()
    # Self-play / MCTS
    ap.add_argument("--n_workers", type=int, default=8)
    ap.add_argument("--games_per_worker", type=int, default=16) # Số game mỗi worker chơi TRƯỚC KHI khởi động lại
    ap.add_argument("--n_sim", type=int, default=200)
    ap.add_argument("--c_puct", type=float, default=1.4)
    ap.add_argument("--dirichlet_alpha", type=float, default=0.3)
    ap.add_argument("--dirichlet_eps", type=float, default=0.25)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--temperature_cutoff", type=int, default=20)
    ap.add_argument("--max_game_plies", type=int, default=200)

    # Replay/Train
    ap.add_argument("--buffer_size", type=int, default=400000)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--train_steps", type=int, default=100000)
    ap.add_argument("--lr", type=float, default=1e-3)

    # Model
    ap.add_argument("--proj_dim", type=int, default=512)
    ap.add_argument("--base_channels", type=int, default=64)
    ap.add_argument("--n_res_blocks", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--se", action="store_true")

    # Infra
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--logdir", type=str, default="./logs/mcts_run")
    ap.add_argument("--save", type=str, default="./checkpoints/az_run.pt")
    ap.add_argument("--best_path", type=str, default="./checkpoints/az_best.pt")
    ap.add_argument("--resume_from", type=str, default="")
    ap.add_argument("--buffer_npz", type=str, default="./checkpoints/az_buffer.npz")
    ap.add_argument("--eval_freq", type=int, default=5000)

    # Obs options
    ap.add_argument("--flip_perspective", action="store_true")
    ap.add_argument("--obs_stack_size", type=int, default=4, help="Number of obs frames to stack") # NEW

    return ap.parse_args()


def main():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    args = parse_args()
    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    os.makedirs(os.path.dirname(args.best_path), exist_ok=True)

    device = "cuda" if (args.device == "auto" and th.cuda.is_available()) else (args.device if args.device != "auto" else "cpu")
    print("Using device:", device)

    # Net
    # NEW: Khởi tạo dummy env với config mới
    dummy_env = ChessEnv(
        illegal_action_mode="reject",
        enable_promotion_actions=True,
        enable_auto_queen=False,
        observation_config={
            "flip_perspective": args.flip_perspective,
            "obs_stack_size": args.obs_stack_size
        }
    )
    net = ActorCriticV2(dummy_env.observation_space,
                        action_dim=dummy_env.action_space.n, # NEW
                        extractor_cfg=ExtractorConfig(base_channels=args.base_channels,
                                                      n_res_blocks=args.n_res_blocks,
                                                      use_se=args.se,
                                                      dropout=args.dropout,
                                                      proj_dim=args.proj_dim)).to(device)
    opt = th.optim.Adam(net.parameters(), lr=args.lr)

    # Resume
    global_step = 0
    if args.resume_from and os.path.isfile(args.resume_from):
        try:
            state = th.load(args.resume_from, map_location=device)
            net.load_state_dict(state.get("state_dict", state))
            print("Loaded:", args.resume_from)
        except Exception as e:
            print("[WARN] failed to load checkpoint:", e)

    rb = ReplayBuffer(capacity=args.buffer_size)
    rb.load_npz(args.buffer_npz)
    writer = SummaryWriter(log_dir=args.logdir)

    out_queue: mp.Queue = mp.Queue(maxsize=10000)
    workers: List[mp.Process] = [] # NEW: Danh sách worker đang chạy
    
    best_elo = -1e9
    last_eval = 0
    last_save = 0

    try:
        # NEW: Vòng lặp quản lý worker
        while global_step < args.train_steps:
            
            # 1. Quản lý/Khởi động lại worker (Stale Data Fix)
            workers = [p for p in workers if p.is_alive()] # Xóa worker đã join
            if len(workers) < args.n_workers:
                print(f"Topping up workers: {len(workers)}/{args.n_workers} running.")
                w_state = net.state_dict() # Lấy model MỚI NHẤT
                for i in range(args.n_workers - len(workers)):
                    w_seed = args.seed + global_step + i # Seed khác nhau mỗi lần
                    p = mp.Process(target=play_game_worker, args=(w_state, device, args, out_queue, w_seed, i))
                    p.daemon = True
                    p.start()
                    workers.append(p)

            # 2. Lấy dữ liệu từ queue (giữ nguyên)
            drained = 0
            while not out_queue.empty() and drained < 10000:
                s, pi, z = out_queue.get()
                rb.push(s, pi, z)
                drained += 1
            if drained > 0 and global_step % 100 == 0: # Giảm log
                writer.add_scalar("buffer/size", len(rb), global_step)

            if len(rb) < args.batch_size:
                time.sleep(0.1) # Chờ buffer đầy
                continue

            # 3. Sample và train (giữ nguyên)
            obs, pi, z = rb.sample(args.batch_size)
            obs_t = th.as_tensor(obs, dtype=th.float32, device=device)
            pi_t = th.as_tensor(pi, dtype=th.float32, device=device)
            z_t = th.as_tensor(z, dtype=th.float32, device=device)

            logits, value = net(obs_t)
            logp = th.log_softmax(logits, dim=-1)
            policy_loss = -(pi_t * logp).sum(dim=-1).mean()
            value_loss = ((value.squeeze() - z_t) ** 2).mean() # Squeeze value
            loss = policy_loss + value_loss

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            global_step += 1

            if global_step % 50 == 0:
                writer.add_scalar("train/policy_loss", float(policy_loss.item()), global_step)
                writer.add_scalar("train/value_loss", float(value_loss.item()), global_step)
                writer.add_scalar("train/loss", float(loss.item()), global_step)

            # 4. Periodic eval & save (giữ nguyên)
            if global_step - last_eval >= args.eval_freq:
                last_eval = global_step
                S_r, wr_r, dr_r, elo_r = evaluate(net, device, 16, "random", args.obs_stack_size)
                writer.add_scalar("eval_random/score", S_r, global_step)
                writer.add_scalar("eval_random/elo", elo_r, global_step)

                S_h, wr_h, dr_h, elo_h = evaluate(net, device, 16, "heur", args.obs_stack_size)
                writer.add_scalar("eval_heur/score", S_h, global_step)
                writer.add_scalar("eval_heur/elo", elo_h, global_step)

                cur_metric = elo_h
                if cur_metric > best_elo:
                    best_elo = cur_metric
                    th.save({"state_dict": net.state_dict()}, args.best_path)

            if global_step - last_save >= 1000:
                last_save = global_step
                
                # 1. Tạo tên file checkpoint duy nhất (ví dụ: mcts_run_step_2000.pt)
                #    Nó sẽ dùng tên file trong --save làm cơ sở.
                chk_path = os.path.splitext(args.save)[0] + f"_step{global_step}.pt"
                
                # 2. Lưu model checkpoint
                th.save({"state_dict": net.state_dict()}, chk_path)
                print(f"--- Saved Model Checkpoint (Step {global_step}) ---")

                # 3. Chỉ lưu buffer (chậm) mỗi 5000 steps (ví dụ)
                #    Việc này rất chậm, không nên làm thường xuyên
                if global_step % 5000 == 0:
                    rb.save_npz(args.buffer_npz)
                    print(f"--- Saved Replay Buffer (Step {global_step}) ---")

    finally:
        for p in workers:
            p.join(timeout=1.0)
        th.save({"state_dict": net.state_dict()}, args.save)
        rb.save_npz(args.buffer_npz)
        writer.close()
        print("Saved:", args.save)


if __name__ == "__main__":
    main()