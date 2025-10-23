import math, random, time
import chess
import numpy as np
import torch as th
import torch.nn.functional as F
from env import board_to_planes # Cần hàm này để tạo quan sát cho mạng
from model import ActorCriticNet # Import kiến trúc mạng chung

# --- Biến Toàn Cục để Quản lý Model và Device ---
_model: ActorCriticNet = None
_device: th.device = th.device("cpu")
_action_dim = 64 * 64

class MCTSNode:
    def __init__(self, board: chess.Board, parent=None, move=None, prior_p=1.0):
        self.board = board
        self.parent = parent
        self.move = move  # move that led to this node
        self.children = {}
        self.N = 0  # Visit count
        self.W = 0.0 # Total value
        self.Q = 0.0 # Mean value (W/N)
        self.P = prior_p # Prior probability from the Policy Network

    def uct_score(self, c_puct=1.0):
        """Công thức UCT kiểu AlphaZero/Policy-Value."""
        if self.N == 0:
            # Nếu chưa được thăm, UCT score là vô cùng lớn (hoặc P*sqrt(N_parent))
            if self.parent:
                # Tránh log(0)
                return float('inf') if self.parent.N == 0 else self.Q + c_puct * self.P * (math.sqrt(self.parent.N) / 1.0)
            return float('inf') # Node gốc
            
        # UCT = Q + c_puct * P * sqrt(N_parent) / (1 + N_child)
        return self.Q + c_puct * self.P * math.sqrt(self.parent.N) / (1 + self.N)

# --- Chức năng Tải Model và Đánh giá Board ---

def load_model_for_mcts(model_path: str, device_str="cpu"):
    """Tải mạng nơ-ron lên thiết bị và đặt vào biến toàn cục."""
    global _model, _device
    _device = th.device(device_str)
    
    # Tạo mô hình và tải trọng số
    model = ActorCriticNet() 
    
    # Nếu tải file .zip (từ PPO/SB3), cần xử lý khác
    if model_path.endswith(".pt"):
         model.load_state_dict(th.load(model_path, map_location=_device))
    elif model_path.endswith(".zip"):
        # Xử lý tải mô hình PPO/SB3 và lấy trọng số
        from stable_baselines3.common.save_util import load_parameters
        params = load_parameters(model_path, _device)
        model.load_state_dict(params['policy']) # Giả định 'policy' chứa ActorCriticNet weights
    else:
        raise ValueError("Unsupported model file type. Use .pt or .zip.")
    
    _model = model.to(_device)
    _model.eval()
    print(f"MCTS Model loaded successfully onto {_device}.")

def to_tensor(obs_np):
    # (8,8,12) -> (1,12,8,8)
    return th.from_numpy(obs_np).float().permute(2,0,1).unsqueeze(0).to(_device)

def evaluate_board(board: chess.Board) -> tuple[np.ndarray, float]:
    """
    Sử dụng mạng nơ-ron (trên GPU) để đánh giá trạng thái.
    Trả về Policy Logits và Value.
    """
    if _model is None:
        # Fallback (chỉ nên xảy ra khi không tải model, tức là MCTS cơ bản)
        return None, 0.0

    obs = board_to_planes(board)
    x = to_tensor(obs)
    
    with th.no_grad():
        logits, value_t = _model(x)
        # Logits: (1, 4096) -> Policy distribution P
        # Value: (1,) -> Estimated win probability V
        
        # Áp dụng mặt nạ hành động HỢP LỆ (Quan trọng cho MCTS)
        mask = np.zeros(_action_dim, dtype=np.float32)
        legal_moves = list(board.legal_moves)
        for mv in legal_moves:
            idx = 64 * mv.from_square + mv.to_square
            mask[idx] = 1.0
            
        logits_np = logits.squeeze(0).cpu().numpy()
        mask_t = th.from_numpy(mask).to(_device)
        
        # Áp dụng mask: cộng -inf vào các logits bất hợp lệ
        inf = 1e9
        logits_masked = logits + th.where(mask_t>0, th.zeros_like(logits), th.full_like(logits, -inf))
        
        # Tính Policy P (phân phối xác suất)
        policy_p = F.softmax(logits_masked, dim=-1).squeeze(0).cpu().numpy()
        
        value = value_t.item()
        
    return policy_p, value # Policy (4096-dim array), Value (scalar)

def choose_action(board: chess.Board, n_simulations=400, time_limit=None, c_puct=1.0):
    """
    MCTS kiểu AlphaZero: Selection -> Evaluation (thay Rollout) -> Backpropagation.
    """
    root = MCTSNode(board.copy())
    start = time.time()
    sims = 0
    
    # 1. EVALUATION NODE GỐC (Lần đầu)
    policy_p, value = evaluate_board(root.board)
    if policy_p is not None:
        # Expansion ban đầu
        for mv in root.board.legal_moves:
            idx = 64 * mv.from_square + mv.to_square
            new_board = root.board.copy()
            new_board.push(mv)
            root.children[mv] = MCTSNode(new_board, parent=root, move=mv, prior_p=policy_p[idx])
        # Backup (chỉ value)
        root.N = 1 # Đã thăm 1 lần (Evaluation)
        root.W = value
        root.Q = value

    while sims < n_simulations:
        node = root
        
        # 2. SELECTION (Dựa trên UCT của AZ)
        while node.children:
            best_move, best_child = max(node.children.items(), key=lambda kv: kv[1].uct_score(c_puct))
            node = best_child
        
        # Nước đi dẫn đến node hiện tại đã hết legal moves hoặc là node lá (chưa mở rộng)
        if node.board.is_game_over():
             # 3. BACKUP (Game Over)
             reward = 0.0
             if node.board.is_checkmate():
                 # Win/Loss: +1.0 cho người vừa di chuyển, -1.0 cho người sắp di chuyển
                 # 'value' là từ góc nhìn của side-to-move TẠI NƯỚC ĐÓ.
                 # Nếu người vừa di chuyển thắng, value = -1.0 (cho side-to-move của node hiện tại)
                 reward = -1.0 
             elif node.board.is_stalemate() or node.board.is_insufficient_material() or node.board.is_seventyfive_moves() or node.board.is_fivefold_repetition():
                 reward = 0.0
             # Backup kết quả terminal
             cur = node
             while cur is not None:
                cur.N += 1
                cur.W += reward
                cur.Q = cur.W / cur.N
                reward *= -1.0 # Đảo dấu cho mỗi cấp độ lên
                cur = cur.parent
             sims += 1
             if time_limit and (time.time() - start) > time_limit: break
             continue

        # 3. EXPANSION (Nếu node chưa được mở rộng) và EVALUATION
        
        # Policy P và Value V từ mạng nơ-ron
        policy_p, value = evaluate_board(node.board)
        
        # Expansion: Mở rộng node
        for mv in node.board.legal_moves:
            idx = 64 * mv.from_square + mv.to_square
            new_board = node.board.copy()
            new_board.push(mv)
            # Khởi tạo node con với Prior P từ mạng
            node.children[mv] = MCTSNode(new_board, parent=node, move=mv, prior_p=policy_p[idx])
            
        # 4. BACKUP (Chỉ Value V)
        cur = node
        reward = value # V là phần thưởng (estimate)
        while cur is not None:
            cur.N += 1
            cur.W += reward
            cur.Q = cur.W / cur.N
            reward *= -1.0 # Đảo dấu vì Value là từ góc nhìn của side-to-move
            cur = cur.parent

        sims += 1
        if time_limit and (time.time() - start) > time_limit:
            break
            
    # CHỌN NƯỚC ĐI (Dựa trên số lần thăm N cao nhất - robust nhất)
    if not root.children:
        return None
        
    # Chọn nước đi với N cao nhất (hoặc Q cao nhất nếu N bằng nhau)
    best_move = max(root.children.items(), key=lambda kv: kv[1].N)[0]
    action_idx = 64 * best_move.from_square + best_move.to_square
    return action_idx