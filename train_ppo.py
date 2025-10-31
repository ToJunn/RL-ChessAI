from __future__ import annotations

import os
import json
import math
import time
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Callable

import numpy as np
import torch as th
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed

# --- Local imports (v4) ---
from envv4 import ChessEnv, BASE_ACTION_DIM, move_index, PROMO_KINDS


from modelv2 import make_policy_kwargs_v2, ExtractorConfig


# -------------------- Opponents (UPDATED) --------------------
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
        
        if best is None: # Nếu mọi nước đi đều tệ
            best = np.random.choice(legal_moves)
            
        # NEW: Trả về 20480-dim action index
        promo_id = PROMO_KINDS.index(best.promotion) if best.promotion in PROMO_KINDS else 0
        return (move_index(best.from_square, best.to_square) * 5) + promo_id


# -------------------- Env factory (UPDATED) --------------------

def make_env_fn(
    seed: int,
    illegal_mode: str = "reject",
    flip_perspective: bool = False,
    obs_stack_size: int = 1, # NEW
    use_fide_stalemate: bool = False,
    use_fide_insufficient: bool = False,
    use_fide_50move: bool = False,
    start_fen: Optional[str] = None,
):
    def _thunk():
        env = ChessEnv(
            illegal_action_mode=illegal_mode,
            enable_promotion_actions=True, # NEW: BẮT BUỘC
            enable_auto_queen=False,       # NEW: BẮT BUỘC
            observation_config={
                "flip_perspective": flip_perspective,
                "obs_stack_size": obs_stack_size # NEW
            },
            termination_config={
                "use_fide_stalemate": use_fide_stalemate,
                "use_fide_insufficient": use_fide_insufficient,
                "use_fide_50move": use_fide_50move,
            },
            start_fen=start_fen,
            seed=seed,
        )
        # ActionMasker hoạt động tốt với 20480-dim
        return ActionMasker(env, lambda e: e.get_action_mask())
    return _thunk


# -------------------- Evaluation (UPDATED) --------------------
@dataclass
class EvalResult:
    episodes: int
    mean_len: float
    win_rate: float
    draw_rate: float
    loss_rate: float
    score: float
    elo_vs_random: float | None
    elo_vs_heur: float | None

def estimate_elo_from_score(s: float) -> float | None:
    # (Hàm này giữ nguyên)
    eps = 1e-6
    s = float(np.clip(s, eps, 1 - eps))
    try: return -400.0 * math.log10(1.0 / s - 1.0)
    except Exception: return None

def evaluate(model: MaskablePPO, n_episodes: int = 16, opponent_type: str = "random", obs_stack_size: int = 1) -> EvalResult:
    # NEW: Cập nhật env dùng để đánh giá
    env = ChessEnv(
        illegal_action_mode="reject",
        enable_promotion_actions=True,
        enable_auto_queen=False,
        observation_config={"obs_stack_size": obs_stack_size} # NEW
    )
    opp = RandomOpponent() if opponent_type == "random" else MaterialHeuristicOpponent()

    wins = draws = losses = 0
    lengths = []

    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_len = 0
        while not done:
            if env.board.turn:
                mask = env.get_action_mask() # (20480,)
                action, _ = model.predict(obs, deterministic=True, action_masks=mask)
                obs, _, term, trunc, info = env.step(int(action))
                done = term or trunc
            else:
                a = opp.choose(env.board) # NEW: Đã trả về 20480-dim action
                if a is None:
                    legal_idxs = np.where(env.get_action_mask() > 0)[0]
                    if len(legal_idxs) == 0:
                        done = True
                        break
                    a = int(np.random.choice(legal_idxs))
                obs, _, term, trunc, info = env.step(int(a))
                done = term or trunc
            ep_len += 1
        
        # (Logic tính ELO giữ nguyên)
        res = info.get("result")
        if res == "1-0": wins += 1
        elif res == "0-1": losses += 1
        else: draws += 1
        lengths.append(ep_len)

    S = (wins + 0.5 * draws) / max(1, n_episodes)
    elo = estimate_elo_from_score(S)
    if opponent_type == "random": elo_rnd, elo_heur = elo, None
    else: elo_rnd, elo_heur = None, elo
    return EvalResult(
        episodes=n_episodes,
        mean_len=float(np.mean(lengths) if lengths else 0),
        win_rate=wins / n_episodes,
        draw_rate=draws / n_episodes,
        loss_rate=losses / n_episodes,
        score=S,
        elo_vs_random=elo_rnd,
        elo_vs_heur=elo_heur,
    )


# -------------------- Callbacks (UPDATED) --------------------
class EvalAndCheckpointCallback(BaseCallback):
    def __init__(
        self,
        writer: SummaryWriter,
        logdir: str,
        obs_stack_size: int = 1, # NEW
        eval_freq: int = 25000,
        eval_eps: int = 16,
        save_every: int = 100000,
        save_path: Optional[str] = None,
        best_path: Optional[str] = None,
    ):
        super().__init__(verbose=0)
        self.writer = writer
        self.logdir = logdir
        self.obs_stack_size = obs_stack_size # NEW
        self.eval_freq = eval_freq
        self.eval_eps = eval_eps
        self.save_every = save_every
        self.save_path = save_path
        self.best_path = best_path
        self.last_eval_step = 0
        self.last_save_step = 0
        self.best_elo = -1e9
        self.best_score = -1e9

    def _on_step(self) -> bool:
        step = int(self.num_timesteps)
        if step - self.last_eval_step >= self.eval_freq:
            self.last_eval_step = step
            # NEW: Truyền obs_stack_size vào hàm đánh giá
            res_r = evaluate(self.model, n_episodes=self.eval_eps, opponent_type="random", obs_stack_size=self.obs_stack_size)
            res_h = evaluate(self.model, n_episodes=self.eval_eps, opponent_type="heur", obs_stack_size=self.obs_stack_size)

            # (Logic log TB và lưu best model giữ nguyên)
            self.writer.add_scalar("eval_random/win_rate", res_r.win_rate, step)
            if res_r.elo_vs_random is not None:
                self.writer.add_scalar("eval_random/elo_diff", res_r.elo_vs_random, step)
            self.writer.add_scalar("eval_heur/win_rate", res_h.win_rate, step)
            if res_h.elo_vs_heur is not None:
                self.writer.add_scalar("eval_heur/elo_diff", res_h.elo_vs_heur, step)

            cur_metric = (res_h.elo_vs_heur if res_h.elo_vs_heur is not None else -1e9)
            cur_score = res_h.score
            improved = False
            if cur_metric is not None and cur_metric > self.best_elo:
                self.best_elo = cur_metric
                improved = True
            elif cur_metric is None and cur_score > self.best_score:
                self.best_score = cur_score
                improved = True
            if improved and self.best_path:
                self.model.save(self.best_path)

        if self.save_path and step - self.last_save_step >= self.save_every:
            self.last_save_step = step
            path = os.path.splitext(self.save_path)[0] + f"_step{step}.zip"
            self.model.save(path)
        return True


# -------------------- LR schedules (Giữ nguyên) --------------------

def make_lr_schedule(initial_lr: float, mode: str) -> Callable[[float], float] | float:
    mode = (mode or "constant").lower()
    if mode == "constant": return initial_lr
    if mode == "linear":
        def fn(frac: float) -> float:
            return initial_lr * frac
        return fn
    return initial_lr


# -------------------- CLI & Train (UPDATED) --------------------

def parse_args():
    ap = argparse.ArgumentParser()
    # (Các tham số PPO, Infra, Model... giữ nguyên)
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--n_envs", type=int, default=8)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--logdir", type=str, default="./logs/ppo_run2")
    ap.add_argument("--save", type=str, default="./checkpoints/ppo_maskable_run2.zip")
    ap.add_argument("--save_every", type=int, default=100_000)
    ap.add_argument("--best_path", type=str, default="./checkpoints/ppo_maskable_best.zip")
    ap.add_argument("--resume_from", type=str, default="")
    ap.add_argument("--eval_freq", type=int, default=25_000)
    ap.add_argument("--eval_eps", type=int, default=32)
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--lr_schedule", type=str, choices=["constant", "linear"], default="constant")
    ap.add_argument("--ent_coef", type=float, default=0.005)
    ap.add_argument("--clip_range", type=float, default=0.2)
    ap.add_argument("--n_steps", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--n_epochs", type=int, default=3)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae_lambda", type=float, default=0.95)
    ap.add_argument("--vf_coef", type=float, default=0.5)
    ap.add_argument("--max_grad_norm", type=float, default=0.5)
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


def make_vec_env(n_envs: int, seed: int, env_kwargs: Dict):
    # (Hàm này giữ nguyên, nó chỉ truyền env_kwargs)
    if n_envs > 1:
        return SubprocVecEnv([make_env_fn(seed + i, **env_kwargs) for i in range(n_envs)])
    else:
        return DummyVecEnv([make_env_fn(seed, **env_kwargs)])


def main():
    args = parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    os.makedirs(os.path.dirname(args.best_path), exist_ok=True)
    with open(os.path.join(args.logdir, "hparams.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    device = "cuda" if (args.device == "auto" and th.cuda.is_available()) else (args.device if args.device != "auto" else "cpu")
    print("Using device:", device)
    set_random_seed(args.seed)
    writer = SummaryWriter(log_dir=args.logdir)

    # Env (UPDATED)
    env_kwargs = dict(
        illegal_mode="reject",
        # NEW: Bật 20480 action space
        # enable_promotion_actions=True, (Đã chuyển vào make_env_fn)
        # enable_auto_queen=False,       (Đã chuyển vào make_env_fn)
        obs_stack_size=args.obs_stack_size, # NEW
        flip_perspective=args.flip_perspective,
        use_fide_stalemate=args.use_fide_stalemate,
        use_fide_insufficient=args.use_fide_insufficient,
        use_fide_50move=args.use_fide_50move,
        start_fen=(args.start_fen or None),
    )
    vec_env = make_vec_env(args.n_envs, args.seed, env_kwargs)

    # Policy kwargs (Giữ nguyên)
    # modelv2.py sẽ tự động xử lý obs_stack_size
    extractor_cfg = ExtractorConfig(
        base_channels=args.base_channels,
        n_res_blocks=args.n_res_blocks,
        use_se=args.se,
        dropout=args.dropout,
        proj_dim=args.proj_dim,
    )
    policy_kwargs = make_policy_kwargs_v2(
        features_dim=args.proj_dim,
        net_arch_pi=(256,),
        net_arch_vf=(256,),
        extractor_cfg=extractor_cfg,
    )
    lr = make_lr_schedule(args.learning_rate, args.lr_schedule)

    # Model (Giữ nguyên)
    # MaskablePPO sẽ tự động lấy obs/action space từ vec_env
    model = MaskablePPO(
        policy="CnnPolicy",
        env=vec_env,
        learning_rate=lr,
        ent_coef=args.ent_coef,
        clip_range=args.clip_range,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=args.logdir,
        device=device,
    )

    if args.resume_from:
        try:
            print("Loading checkpoint:", args.resume_from)
            model.load(args.resume_from)
        except Exception as e:
            print("[WARN] Failed to load checkpoint:", e)

    # Callback (UPDATED)
    cb = EvalAndCheckpointCallback(
        writer=writer,
        logdir=args.logdir,
        obs_stack_size=args.obs_stack_size, # NEW
        eval_freq=args.eval_freq,
        eval_eps=args.eval_eps,
        save_every=args.save_every,
        save_path=args.save,
        best_path=args.best_path,
    )

    # Train (Giữ nguyên)
    run_name = "PPO"
    tb_dir = Path(args.logdir) / run_name
    tb_dir.mkdir(parents=True, exist_ok=True)
    model.learn(total_timesteps=args.timesteps, callback=cb, progress_bar=True,tb_log_name=run_name,)
    model.save(args.save)
    writer.close()
    print("Saved final:", args.save)


if __name__ == "__main__":
    main()