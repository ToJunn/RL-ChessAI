from __future__ import annotations

import os
import time
import math
import argparse
import multiprocessing as mp
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
import random # Cần cho fallback

import numpy as np
import torch as th
import torch.nn as nn
import torch.multiprocessing as torch_mp
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.utils import set_random_seed

# --- Local imports (v3) ---
from envv4 import ChessEnv, move_index, PROMO_KINDS, BASE_ACTION_DIM

from modelv2 import ActorCriticV2 as ActorCritic, ExtractorConfig


# ================= Opponents (UPDATED) =================
class RandomOpponent:
    def choose(self, board) -> Optional[int]:
        legal = list(board.legal_moves)
        if not legal:
            return None
        mv = np.random.choice(legal)
        # NEW: Trả về 20480-dim action index
        promo_id = PROMO_KINDS.index(mv.promotion) if mv.promotion in PROMO_KINDS else 0
        return (move_index(mv.from_square, mv.to_square) * 5) + promo_id

class MaterialHeuristicOpponent:
    PIECE = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}
    def _material_delta(self, board, mv) -> float:
        if board.is_capture(mv):
            if board.is_en_passant(mv): return self.PIECE[1]
            tgt = board.piece_at(mv.to_square)
            if tgt is not None: return self.PIECE[tgt.piece_type]
        return 0.0
    
    def choose(self, board) -> Optional[int]:
        best, best_score = None, -1e9
        legal_moves = list(board.legal_moves)
        if not legal_moves: return None
        
        for mv in legal_moves:
            sc = self._material_delta(board, mv)
            if sc > best_score:
                best_score, best = sc, mv
        
        if best is None: # Fallback
            best = random.choice(legal_moves)
            
        # NEW: Trả về 20480-dim action index
        promo_id = PROMO_KINDS.index(best.promotion) if best.promotion in PROMO_KINDS else 0
        return (move_index(best.from_square, best.to_square) * 5) + promo_id


# ================= Evaluation (UPDATED) =================
@dataclass
class EvalResult:
    episodes: int
    mean_len: float
    win_rate: float
    draw_rate: float
    loss_rate: float
    score: float
    elo_diff: Optional[float]

def elo_from_score(s: float) -> Optional[float]:
    s = float(np.clip(s, 1e-6, 1 - 1e-6))
    return -400.0 * math.log10(1.0 / s - 1.0)

@th.no_grad()
def evaluate(model: ActorCritic, device: str, n_episodes: int, opponent_type: str = "random", obs_stack_size: int = 4) -> EvalResult:
    # NEW: Cập nhật env dùng để đánh giá
    env = ChessEnv(
        illegal_action_mode="reject",
        enable_promotion_actions=True, # BẮT BUỘC
        enable_auto_queen=False,       # BẮT BUỘC
        observation_config={"obs_stack_size": obs_stack_size} # NEW
    )
    opp = RandomOpponent() if opponent_type == "random" else MaterialHeuristicOpponent()

    wins = draws = losses = 0
    lengths: List[int] = []
    model.eval()

    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_len = 0
        while not done:
            if env.board.turn:  # agent move
                mask = env.get_action_mask() # (20480,)
                obs_t = th.as_tensor(obs, dtype=th.float32).unsqueeze(0).to(device)
                a = model.act(obs_t, mask, device=device, deterministic=True)
                obs, _, term, trunc, info = env.step(a)
            else:
                a = opp.choose(env.board) # NEW: Đã trả về 20480-dim action
                if a is None:
                    legal_idxs = np.where(env.get_action_mask() > 0)[0]
                    if len(legal_idxs) == 0:
                        break
                    a = int(np.random.choice(legal_idxs))
                obs, _, term, trunc, info = env.step(a)
            done = term or trunc
            ep_len += 1
        
        # (Logic tính ELO giữ nguyên)
        res = info.get("result")
        if res == "1-0": wins += 1
        elif res == "0-1": losses += 1
        else: draws += 1
        lengths.append(ep_len)

    S = (wins + 0.5 * draws) / max(1, n_episodes)
    return EvalResult(
        episodes=n_episodes,
        mean_len=float(np.mean(lengths) if lengths else 0.0),
        win_rate=wins / n_episodes,
        draw_rate=draws / n_episodes,
        loss_rate=losses / n_episodes,
        score=S,
        elo_diff=elo_from_score(S),
    )


# ================= Worker (UPDATED) =================

def worker_proc(global_net: ActorCritic,
                optimizer: th.optim.Optimizer,
                device: str,
                rank: int,
                args,
                global_counter: mp.Value,
                log_queue: mp.Queue,
                stop_flag: mp.Event):

    worker_seed = args.seed + rank
    np.random.seed(worker_seed)
    th.manual_seed(worker_seed)

    # NEW: Cập nhật env của worker
    obs_config = {
        "flip_perspective": args.flip_perspective,
        "obs_stack_size": args.obs_stack_size # NEW
    }
    env = ChessEnv(
        illegal_action_mode="reject",
        enable_promotion_actions=True, # BẮT BUỘC
        enable_auto_queen=False,       # BẮT BUỘC
        observation_config=obs_config,
        termination_config={
            "use_fide_stalemate": args.use_fide_stalemate,
            "use_fide_insufficient": args.use_fide_insufficient,
            "use_fide_50move": args.use_fide_50move,
        },
        start_fen=(args.start_fen or None),
        seed=worker_seed,
    )
    
    ACTION_DIM = env.action_space.n # Sẽ là 20480

    # Local copy (UPDATED)
    local_net = ActorCritic(
        env.observation_space, # (N*C, H, W)
        action_dim=ACTION_DIM, # NEW: Sửa lỗi 4096
        extractor_cfg=ExtractorConfig(
            base_channels=args.base_channels,
            n_res_blocks=args.n_res_blocks,
            use_se=args.se,
            dropout=args.dropout,
            proj_dim=args.proj_dim,
        ),
    ).to(device)
    local_net.load_state_dict(global_net.state_dict())

    # (Logic A3C bên dưới giữ nguyên, nó độc lập với action_dim)
    gamma = args.gamma
    t_max = args.t_max
    ent_coef = args.entropy_coef
    val_coef = args.value_coef
    max_grad_norm = args.max_grad_norm
    opt_epochs = getattr(args, 'opt_epochs', 1)

    while not stop_flag.is_set():
        obs, info = env.reset()
        done = False
        ep_rewards: List[float] = []

        traj_values: List[th.Tensor] = []
        traj_logp: List[th.Tensor] = []
        traj_ent: List[th.Tensor] = []
        traj_rewards: List[float] = []

        for _ in range(1, t_max + 1):
            obs_t = th.as_tensor(obs, dtype=th.float32).unsqueeze(0).to(device)
            logits, value = local_net(obs_t) # logits là (1, 20480)
            mask = env.get_action_mask()     # mask là (20480,)
            mask_t = th.from_numpy(mask).to(device)
            logits = logits.squeeze(0).masked_fill(mask_t <= 0, float("-inf"))
            
            probs = th.softmax(logits, dim=-1)
            dist = th.distributions.Categorical(probs)
            action = int(dist.sample().item())
            logp = dist.log_prob(th.tensor(action, device=device)).reshape(())
            ent = dist.entropy().reshape(())

            obs2, reward, term, trunc, info = env.step(action)
            done = term or trunc

            traj_values.append(value.squeeze())
            traj_logp.append(logp)
            traj_ent.append(ent)
            traj_rewards.append(float(reward))
            ep_rewards.append(float(reward))

            obs = obs2

            with global_counter.get_lock():
                global_counter.value += 1
                total_steps = global_counter.value

            if done:
                break

        if done:
            R = th.tensor(0.0, device=device)
        else:
            obs_t = th.as_tensor(obs, dtype=th.float32).unsqueeze(0).to(device)
            _, v = local_net(obs_t)
            R = v.detach().squeeze()

        returns: List[th.Tensor] = []
        for r in reversed(traj_rewards):
            R = th.tensor(r, device=device).reshape(()) + gamma * R
            returns.insert(0, R)

        vals  = th.stack([v.reshape(()) for v in traj_values])
        rets  = th.stack([R.reshape(()) for R in returns])
        advs  = (rets - vals).detach()
        logps = th.stack(traj_logp)
        ents  = th.stack(traj_ent)

        for i in range(max(1, opt_epochs)):
            policy_loss = -(logps * advs).sum()
            value_loss  = 0.5 * (rets - vals).pow(2).sum()
            entropy_loss= ents.mean()
            loss = policy_loss + val_coef * value_loss - ent_coef * entropy_loss
        
            optimizer.zero_grad()
        
            # SỬA LỖI: Chỉ giữ lại graph nếu đây không phải là epoch cuối
            is_last_iteration = (i == max(1, opt_epochs) - 1)
            loss.backward(retain_graph=(not is_last_iteration))
        
            # (Phần code còn lại của vòng lặp giữ nguyên)
            nn.utils.clip_grad_norm_(local_net.parameters(), max_grad_norm)
            for gp, lp in zip(global_net.parameters(), local_net.parameters()):
                if lp.grad is not None:
                    if gp.grad is None:
                        gp.grad = lp.grad.detach().clone()
                    else:
                        gp.grad.copy_(lp.grad.detach())
            optimizer.step()
            local_net.load_state_dict(global_net.state_dict())

        if rank == 0:
            with th.no_grad():
                policy_loss = float((-(logps * advs).sum()).item())
                value_loss  = float((0.5 * (rets - vals).pow(2).sum()).item())
                entropy_val = float(ents.mean().item())
            log_queue.put({
                "steps": int(total_steps),
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy_val,
                "ep_return": float(np.sum(ep_rewards)) if ep_rewards else 0.0,
            })

        if total_steps >= args.timesteps:
            stop_flag.set()


# ================= CLI & Train (UPDATED) =================

def parse_args():
    ap = argparse.ArgumentParser()
    # (Các tham số A3C, Infra, Model... giữ nguyên)
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--n_envs", type=int, default=8)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--logdir", type=str, default="./logs/a3c_run")
    ap.add_argument("--save", type=str, default="./checkpoints/a3c_model.pt")
    ap.add_argument("--best_path", type=str, default="./checkpoints/a3c_best.pt")
    ap.add_argument("--resume_from", type=str, default="")
    ap.add_argument("--save_every", type=int, default=100_000)
    ap.add_argument("--eval_freq", type=int, default=25_000)
    ap.add_argument("--eval_eps", type=int, default=32)
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--t_max", type=int, default=20)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--entropy_coef", type=float, default=0.01)
    ap.add_argument("--value_coef", type=float, default=0.5)
    ap.add_argument("--max_grad_norm", type=float, default=0.5)
    ap.add_argument("--opt_epochs", type=int, default=1)
    ap.add_argument("--proj_dim", type=int, default=512)
    ap.add_argument("--base_channels", type=int, default=64)
    ap.add_argument("--n_res_blocks", type=int, default=3)
    ap.add_argument("--se", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.1)

    # Env options
    ap.add_argument("--flip_perspective", action="store_true")
    ap.add_argument("--use_fide_stalemate", action="store_true")
    ap.add_argument("--use_fide_insufficient", action="store_true")
    ap.add_argument("--use_fide_50move", action="store_true")
    ap.add_argument("--start_fen", type=str, default="")
    
    # NEW: Thêm tham số bắt buộc
    ap.add_argument("--obs_stack_size", type=int, default=4, help="Number of obs frames to stack")
    
    return ap.parse_args()


def main():
    try:
        torch_mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    args = parse_args()
    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    os.makedirs(os.path.dirname(args.best_path), exist_ok=True)

    device = "cuda" if (args.device == "auto" and th.cuda.is_available()) else (args.device if args.device != "auto" else "cpu")
    print("Using device:", device)
    set_random_seed(args.seed)
    writer = SummaryWriter(log_dir=args.logdir)

    # Dummy env (UPDATED)
    obs_config = {
        "flip_perspective": args.flip_perspective,
        "obs_stack_size": args.obs_stack_size # NEW
    }
    dummy_env = ChessEnv(
        illegal_action_mode="reject",
        enable_promotion_actions=True, # BẮT BUỘC
        enable_auto_queen=False,       # BẮT BUỘC
        observation_config=obs_config,
        termination_config={
            "use_fide_stalemate": args.use_fide_stalemate,
            "use_fide_insufficient": args.use_fide_insufficient,
            "use_fide_50move": args.use_fide_50move,
        },
        start_fen=(args.start_fen or None),
        seed=args.seed
    )
    
    ACTION_DIM = dummy_env.action_space.n # Sẽ là 20480

    global_net = ActorCritic(
        dummy_env.observation_space, # (N*C, H, W)
        action_dim=ACTION_DIM,       # NEW: Sửa lỗi 4096
        extractor_cfg=ExtractorConfig(
            base_channels=args.base_channels,
            n_res_blocks=args.n_res_blocks,
            use_se=args.se,
            dropout=args.dropout,
            proj_dim=args.proj_dim,
        ),
    ).to(device)
    global_net.share_memory()

    optimizer = th.optim.Adam(global_net.parameters(), lr=args.learning_rate)

    if args.resume_from:
        try:
            print("Loading checkpoint:", args.resume_from)
            state = th.load(args.resume_from, map_location=device)
            sd = state.get("state_dict", state)
            global_net.load_state_dict(sd)
        except Exception as e:
            print("[WARN] Failed to load checkpoint:", e)

    global_counter = mp.Value('i', 0)
    log_queue: mp.Queue = mp.Queue()
    stop_flag = mp.Event()

    # Spawn workers (giữ nguyên, args sẽ được truyền vào)
    ctx = torch_mp.get_context("spawn")
    workers = []
    for rank in range(args.n_envs):
        p = ctx.Process(target=worker_proc, args=(global_net, optimizer, device, rank, args, global_counter, log_queue, stop_flag))
        p.daemon = True
        p.start()
        workers.append(p)

    last_eval = 0
    last_save = 0
    best_elo = -1e9
    best_score = -1e9
    last_flush = time.time()

    try:
        while not stop_flag.is_set():
            while not log_queue.empty():
                rec = log_queue.get()
                steps = rec["steps"]
                writer.add_scalar("train/policy_loss", rec["policy_loss"], steps)
                writer.add_scalar("train/value_loss", rec["value_loss"], steps)
                writer.add_scalar("train/entropy", rec["entropy"], steps)
                writer.add_scalar("train/ep_return", rec["ep_return"], steps)

                # Eval (UPDATED)
                if steps - last_eval >= args.eval_freq:
                    last_eval = steps
                    # NEW: Truyền obs_stack_size
                    res_r = evaluate(global_net, device, args.eval_eps, "random", args.obs_stack_size)
                    writer.add_scalar("eval_random/win_rate", res_r.win_rate, steps)
                    if res_r.elo_diff is not None:
                        writer.add_scalar("eval_random/elo_diff", res_r.elo_diff, steps)

                    res_h = evaluate(global_net, device, args.eval_eps, "heur", args.obs_stack_size)
                    writer.add_scalar("eval_heur/win_rate", res_h.win_rate, steps)
                    if res_h.elo_diff is not None:
                        writer.add_scalar("eval_heur/elo_diff", res_h.elo_diff, steps)

                    # (Logic lưu best model giữ nguyên)
                    cur_metric = res_h.elo_diff if res_h.elo_diff is not None else -1e9
                    cur_score = res_h.score
                    improved = False
                    if cur_metric is not None and cur_metric > best_elo:
                        best_elo = cur_metric
                        improved = True
                    elif cur_metric is None and cur_score > best_score:
                        best_score = cur_score
                        improved = True
                    if improved and args.best_path:
                        th.save({"state_dict": global_net.state_dict()}, args.best_path)

                # (Logic save định kỳ giữ nguyên)
                if steps - last_save >= args.save_every:
                    last_save = steps
                    path = os.path.splitext(args.save)[0] + f"_step{steps}.pt"
                    th.save({"state_dict": global_net.state_dict()}, path)

            if time.time() - last_flush > 3.0:
                writer.flush()
                last_flush = time.time()

            if global_counter.value >= args.timesteps:
                stop_flag.set()

            time.sleep(0.05)
    finally:
        for p in workers:
            p.join(timeout=1.0)
        th.save({"state_dict": global_net.state_dict()}, args.save)
        writer.close()
        print("Saved:", args.save)


if __name__ == "__main__":
    main()