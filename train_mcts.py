import os
import time
import argparse
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import chess
import torch as th
import chess.pgn
from env import ChessEnv
from mcts import choose_action, load_model_for_mcts 

def play_episode(env: ChessEnv, agent_side: str, n_sims: int, time_limit=None):
    """
    agent_side: "white" (MCTS plays White), "black" (MCTS plays Black), "both" (MCTS vs MCTS)
    MCTS (AZ) luôn được sử dụng nếu model được tải. n_sims chỉ điều chỉnh sức mạnh tìm kiếm.
    returns (result_int, total_reward, ply, moves_list)
      result_int: 1 = MCTS win, 0 = draw, -1 = MCTS loss (from perspective of MCTS agent)
    """
    obs, info = env.reset()
    moves = []
    total_reward = 0.0
    
    n_sims_white = n_sims if agent_side in ("white", "both") else 1 
    n_sims_black = n_sims if agent_side in ("black", "both") else 1 
    
    while True:
        is_white_to_move = env.board.turn == chess.WHITE
        current_n_sims = n_sims_white if is_white_to_move else n_sims_black
        
        if current_n_sims > 1:
            idx = choose_action(env.board, n_simulations=current_n_sims, time_limit=time_limit)
        else:
            idx = None

        if idx is None:
            legal = list(env.board.legal_moves)
            if not legal:
                break
            mv = legal[np.random.randint(len(legal))]
            idx = 64*mv.from_square + mv.to_square

        from_sq = idx // 64
        to_sq = idx % 64
        mv = chess.Move(from_sq, to_sq)
        
        if mv not in env.board.legal_moves:
            found = None
            for legal_mv in env.board.legal_moves:
                if legal_mv.from_square == from_sq and legal_mv.to_square == to_sq:
                    found = legal_mv
                    break
            if found is not None:
                mv = found
                idx = 64*mv.from_square + mv.to_square
            else:
                legal = list(env.board.legal_moves)
                if not legal: break
                mv = legal[np.random.randint(len(legal))]
                idx = 64*mv.from_square + mv.to_square
                
        moves.append(mv)
        obs, r, done, trunc, info = env.step(int(idx))
        total_reward += float(r)
        if done or trunc:
            break

    result = 0
    if env.board.is_checkmate():
        last_mover = chess.BLACK if env.board.turn == chess.WHITE else chess.WHITE
        if agent_side == "both":
            result = 1
        else:
            mcts_color = chess.WHITE if agent_side == "white" else chess.BLACK
            result = 1 if last_mover == mcts_color else -1
    else:
        result = 0

    return result, total_reward, len(moves), moves

def save_pgn(moves, out_path):
    game = chess.pgn.Game()
    node = game
    board = chess.Board()
    for mv in moves:
        node = node.add_variation(mv)
        board.push(mv)
    with open(out_path, "w", encoding="utf-8") as f:
        print(game, file=f)

def main():
    parser = argparse.ArgumentParser(description="AZ-MCTS Evaluation Script.")
    parser.add_argument("--episodes", type=int, default=50, help="Number of games to play.")
    parser.add_argument("--n_sims", type=int, default=200, help="Number of MCTS simulations per move for the agent side(s).")
    parser.add_argument("--mode", choices=["selfplay","vs_random"], default="selfplay",
                        help="selfplay: MCTS vs MCTS; vs_random: MCTS vs Random (MCTS alternates colors)")
    parser.add_argument("--time_limit", type=float, default=None, help="Per-move time limit (s) for MCTS")
    parser.add_argument("--logdir", type=str, default="./logs/mcts", help="Directory to save logs and PGN files.")
    parser.add_argument("--save_pgns", action="store_true", help="Save game records as PGN files.")
    
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained PyTorch model (.pt or .zip) for MCTS evaluation.")
    parser.add_argument("--device", type=str, default="auto", help="Device to use for NN evaluation (e.g., cuda, cpu).")
    
    args = parser.parse_args()

    device = args.device
    if args.device == "auto":
        device = "cuda" if th.cuda.is_available() else "cpu"
    print(f"Sử dụng thiết bị: {device}")


    os.makedirs(args.logdir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.logdir)
    env = ChessEnv()

    try:
        load_model_for_mcts(args.model_path, device)
    except Exception as e:
        print(f"Lỗi khi tải mô hình MCTS: {e}. Vui lòng kiểm tra đường dẫn và định dạng file.")
        return

    wins = draws = losses = 0
    lengths = []
    durations = []

    for epi in range(args.episodes):
        if args.mode == "selfplay":
            agent_side = "both"
        elif args.mode == "vs_random":
            agent_side = "white" if (epi % 2 == 0) else "black"
        else:
            agent_side = "both"

        t0 = time.time()
        res, total_r, ply, moves = play_episode(env, agent_side, args.n_sims, time_limit=args.time_limit)
        dt = time.time() - t0
        durations.append(dt)
        lengths.append(ply)
        if res == 1:
            wins += 1
        elif res == -1:
            losses += 1
        else:
            draws += 1

        writer.add_scalar("episode/reward", total_r, epi)
        writer.add_scalar("episode/length", ply, epi)
        writer.add_scalar("episode/duration_s", dt, epi)
        if args.save_pgns:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = os.path.join(args.logdir, f"game_{epi}_{ts}.pgn")
            save_pgn(moves, fname)

        print(f"[{epi+1}/{args.episodes}] Mode={agent_side} Result={res} Ply={ply} Reward={total_r:.3f} Time={dt:.2f}s (Sims: {args.n_sims})")

    # summary (vẫn giữ lại trong TensorBoard để phân tích sau)
    writer.add_scalar("summary/wins", wins, 0)
    writer.add_scalar("summary/draws", draws, 0)
    writer.add_scalar("summary/losses", losses, 0)
    writer.add_scalar("summary/mean_length", float(np.mean(lengths)) if lengths else 0.0, 0)
    writer.add_scalar("summary/mean_time_per_game", float(np.mean(durations)) if durations else 0.0, 0)
    writer.close()

    # Dòng in tổng kết W/D/L cuối cùng đã bị loại bỏ theo yêu cầu
    print("Logs saved to", args.logdir)

if __name__ == "__main__":
    main()