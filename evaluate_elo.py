from __future__ import annotations

import argparse
import math
import os
import time
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch as th
import chess
import chess.engine # <-- THAY ĐỔI: Import engine thay vì uci

# Import từ các file của bạn
from envv4 import ChessEnv, move_index, PROMO_KINDS
from modelv2 import ActorCriticV2, ExtractorConfig
from sb3_contrib import MaskablePPO

# -------------------- Hàm tính ELO (Giữ nguyên) --------------------
def elo_from_score(s: float) -> Optional[float]:
    s = float(np.clip(s, 1e-6, 1 - 1e-6))
    try:
        return -400.0 * math.log10(1.0 / s - 1.0)
    except Exception:
        return None

# -------------------- Agent Wrappers (Giữ nguyên) --------------------
class PPOAgent:
    def __init__(self, path: str, device: str = 'cpu'):
        self.model = MaskablePPO.load(path, device=device)
        self.device = device
    
    @th.no_grad()
    def choose(self, obs: np.ndarray, mask: np.ndarray) -> int:
        a, _ = self.model.predict(obs, deterministic=True, action_masks=mask)
        return int(a)

class A3CAgent:
    def __init__(self, path: str, obs_space, action_dim: int, device: str = 'cpu', extractor_cfg: Optional[ExtractorConfig] = None, strict: bool = False):
        self.device = device
        self.net = ActorCriticV2(obs_space, action_dim=action_dim, extractor_cfg=extractor_cfg).to(device)
        try:
            state = th.load(path, map_location=device)
            sd = state.get('state_dict', state)
            missing, unexpected = self.net.load_state_dict(sd, strict=strict)
            if not strict:
                if unexpected: print('[WARN] Unexpected keys:', unexpected)
                if missing: print('[WARN] Missing keys:', missing)
        except Exception as e:
            print(f"Lỗi khi tải A3C model: {e}")
            print("Đảm bảo các tham số kiến trúc (base_channels, proj_dim...) khớp với checkpoint!")
            raise e
        self.net.eval()
    
    @th.no_grad()
    def choose(self, obs: np.ndarray, mask: np.ndarray) -> int:
        obs_t = th.as_tensor(obs, dtype=th.float32).unsqueeze(0).to(self.device)
        logits, value = self.net(obs_t)
        logits = logits.squeeze(0)
        mask_t = th.from_numpy(mask).to(self.device)
        logits = logits.masked_fill(mask_t <= 0, float('-inf'))
        a = int(th.argmax(logits).item())
        return a

# -------------------- Hàm chơi 1 ván (CẬP NHẬT) --------------------

def play_game(agent, engine: chess.engine.SimpleEngine, env: ChessEnv, agent_plays_white: bool, movetime: float) -> Tuple[str, str]:
    """
    Chơi 1 ván cờ.
    Trả về: (Kết quả "1-0", "0-1", "1/2-1/2", Lý do)
    """
    obs, info = env.reset()
    board = chess.Board(env.board.fen()) # Khởi tạo board từ FEN của env
    
    try:
        while not board.is_game_over(claim_draw=True):
            is_agent_turn = (board.turn == chess.WHITE and agent_plays_white) or \
                            (board.turn == chess.BLACK and not agent_plays_white)
            
            if is_agent_turn:
                # ----------------- Lượt đi của AGENT -----------------
                mask = env.get_action_mask()
                if not np.any(mask):
                    break 
                
                action_idx = agent.choose(obs, mask)
                move = env._decode_action(action_idx)
                
                if move is None or move not in board.legal_moves:
                    # Rất hiếm khi xảy ra nếu mask đúng, nhưng vẫn check
                    print(f"Agent tried illegal move: {move}")
                    return ("0-1" if agent_plays_white else "1-0", "Illegal_Move_By_Agent")
                
            else:
                # ----------------- Lượt đi của STOCKFISH -----------------
                # THAY ĐỔI: Dùng chess.engine API
                limit = chess.engine.Limit(time=movetime)
                result = engine.play(board, limit)
                if result.move is None:
                    break
                move = result.move

            # Áp dụng nước đi cho cả 2 môi trường
            board.push(move)
            # Dùng hàm helper _decode_action_index (đã thêm ở bản trước) để step env
            obs, _, term, trunc, info = env.step(env._decode_action_index(move))
            
    except Exception as e:
        print(f"Lỗi trong ván cờ: {e}")
        return ("1/2-1/2", f"Error: {e}")

    result = board.result(claim_draw=True)
    return (result, "Game_Over")

# -------------------- Main (CẬP NHẬT) --------------------

def main():
    ap = argparse.ArgumentParser()
    # (Các tham số giữ nguyên)
    ap.add_argument('--model_path', type=str, required=True, help="Đường dẫn tới PPO .zip hoặc A3C .pt")
    ap.add_argument('--agent_type', type=str, choices=['ppo', 'a3c'], required=True)
    ap.add_argument('--stockfish_path', type=str, required=True, help="Đường dẫn tới stockfish.exe")
    ap.add_argument('--stockfish_elo', type=int, default=1400, help="Đặt Elo cố định cho Stockfish (ví dụ 1400)")
    ap.add_argument('--games', type=int, default=50, help="Tổng số ván chơi (1/2 Trắng, 1/2 Đen)")
    ap.add_argument('--movetime', type=int, default=500, help="Stockfish suy nghĩ (ms) mỗi nước")
    ap.add_argument('--device', type=str, default='auto')
    ap.add_argument('--proj_dim', type=int, default=512)
    ap.add_argument('--base_channels', type=int, default=64)
    ap.add_argument('--n_res_blocks', type=int, default=3)
    ap.add_argument('--se', action='store_true')
    ap.add_argument('--dropout', type=float, default=0.0)
    ap.add_argument('--obs_stack_size', type=int, default=4, help="Bắt buộc")
    ap.add_argument('--flip_perspective', action='store_true')
    args = ap.parse_args()

    device = 'cuda' if (args.device=='auto' and th.cuda.is_available()) else (args.device if args.device!='auto' else 'cpu')
    print(f"Using device: {device}")

    # 1. Khởi tạo Môi trường (Giữ nguyên)
    obs_config = {
        "flip_perspective": args.flip_perspective,
        "obs_stack_size": args.obs_stack_size
    }
    dummy_env = ChessEnv(
        illegal_action_mode="reject",
        enable_promotion_actions=True,
        enable_auto_queen=False,
        observation_config=obs_config
    )
    # Thêm hàm helper (Giữ nguyên)
    def _decode_action_index(self, move: chess.Move) -> int:
        promo_id = PROMO_KINDS.index(move.promotion) if move.promotion in PROMO_KINDS else 0
        return (move_index(move.from_square, move.to_square) * 5) + promo_id
    ChessEnv._decode_action_index = _decode_action_index
    
    print(f"Observation space: {dummy_env.observation_space.shape}")
    print(f"Action space: {dummy_env.action_space.n}")

    # 2. Tải Agent (Giữ nguyên)
    agent = None
    extractor_cfg = ExtractorConfig(base_channels=args.base_channels, n_res_blocks=args.n_res_blocks,
                                    use_se=args.se, dropout=args.dropout, proj_dim=args.proj_dim)
    
    if args.agent_type == 'ppo':
        print(f"Loading PPO model from: {args.model_path}")
        agent = PPOAgent(args.model_path, device=device)
    elif args.agent_type == 'a3c':
        print(f"Loading A3C model from: {args.model_path}")
        agent = A3CAgent(args.model_path, 
                         obs_space=dummy_env.observation_space, 
                         action_dim=dummy_env.action_space.n,
                         device=device, 
                         extractor_cfg=extractor_cfg, 
                         strict=True)
    
    # 3. Tải Stockfish (CẬP NHẬT)
    print(f"Loading Stockfish from: {args.stockfish_path}")
    engine = None
    try:
        # THAY ĐỔI: Dùng SimpleEngine.popen_uci
        engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file '{args.stockfish_path}'.")
        print("Hãy tải Stockfish và đặt đường dẫn chính xác.")
        return
        
    # THAY ĐỔI: Dùng engine.configure
    try:
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": args.stockfish_elo})
        print(f"Stockfish Elo set to: {args.stockfish_elo}")
    except Exception:
        print(f"Không thể đặt UCI_Elo. Dùng 'Skill Level' (có thể không chính xác).")
        skill = max(0, min(20, int((args.stockfish_elo - 1000) / 100)))
        engine.configure({"Skill Level": skill})
        print(f"Stockfish Skill Level set to: {skill} (ước tính ~{args.stockfish_elo} Elo)")

    # 4. Chạy giải đấu (CẬP NHẬT)
    n_games = args.games
    wins = 0
    draws = 0
    losses = 0
    movetime_sec = args.movetime / 1000.0 # Chuyển ms sang giây
    
    print(f"Starting gauntlet: {args.agent_type.upper()} vs Stockfish (Elo {args.stockfish_elo}) for {n_games} games...")
    
    for i in range(n_games):
        agent_plays_white = (i % 2 == 0)
        start_time = time.time()
        
        # CẬP NHẬT: Truyền movetime_sec
        result, reason = play_game(agent, engine, dummy_env, agent_plays_white, movetime_sec)
        
        duration = time.time() - start_time
        
        if (result == "1-0" and agent_plays_white) or (result == "0-1" and not agent_plays_white):
            wins += 1
            status = "WIN"
        elif (result == "0-1" and agent_plays_white) or (result == "1-0" and not agent_plays_white):
            losses += 1
            status = "LOSS"
        else:
            draws += 1
            status = "DRAW"
            
        print(f"Game {i+1}/{n_games} ({'W' if agent_plays_white else 'B'}): {status} ({result}) | Reason: {reason} | Time: {duration:.1f}s")
        
    engine.quit()

    # 5. Báo cáo kết quả (Giữ nguyên)
    score = (wins + 0.5 * draws) / n_games
    elo_diff = elo_from_score(score)
    agent_elo = args.stockfish_elo + elo_diff
    
    print("\n--- GAUNTLET RESULTS ---")
    print(f"Agent: {args.model_path} ({args.agent_type.upper()})")
    print(f"Opponent: Stockfish (Elo {args.stockfish_elo})")
    print(f"Games played: {n_games}")
    print(f"Score: {wins} W / {draws} D / {losses} L  (Score: {score*100:.1f}%)")
    print("------------------------")
    print(f"ELO Difference: {elo_diff:+.0f} Elo")
    print(f"Estimated Agent ELO: {agent_elo:.0f}")

if __name__ == "__main__":
    main()