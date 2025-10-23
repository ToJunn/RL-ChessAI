import random
import numpy as np
import chess
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass

@dataclass
class RewardConfig:
    win: float = 1.0
    loss: float = -1.0
    draw: float = 0.0
    length_penalty: float = 5e-4
    use_pbrs: bool = True
    beta: float = 0.5
    gamma: float = 0.99
    center_coef: float = 0.002
    mobility_coef: float = 0.01
    repetition_penalty: float = 0.01

class RewardEngine:
    def __init__(self, cfg: RewardConfig):
        self.cfg = cfg
        self.acc_length = 0.0

    def reset(self):
        self.acc_length = 0.0

    def material(self, board: chess.Board) -> float:
        vals = {chess.PAWN:1, chess.KNIGHT:3, chess.BISHOP:3, chess.ROOK:5, chess.QUEEN:9}
        s = 0.0
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if not p: continue
            v = vals.get(p.piece_type, 0)
            s += v if p.color == chess.WHITE else -v
        return s

    def phi(self, board: chess.Board) -> float:
        return self.cfg.beta * (self.material(board) / 39.0)

    def step_reward(self, before: chess.Board, mv: chess.Move, after: chess.Board):
        # length penalty (simple)
        step_pen = self.cfg.length_penalty
        remaining = max(0.0, 0.08 - self.acc_length)
        pen = min(step_pen, remaining)
        self.acc_length += pen
        r = -pen

        if self.cfg.use_pbrs:
            r += (self.cfg.gamma * self.phi(after) - self.phi(before))

        # center control
        center = {chess.D4, chess.E4, chess.D5, chess.E5}
        mover = not after.turn
        cc = 0
        for sq in center:
            if after.piece_at(sq) and after.piece_at(sq).color == mover:
                cc += 1
            if after.is_attacked_by(mover, sq):
                cc += 1
        r += self.cfg.center_coef * cc

        # mobility
        r += self.cfg.mobility_coef * (len(list(after.legal_moves)) - len(list(before.legal_moves))) / 100.0

        # repetition
        if after.is_repetition(3):
            r -= self.cfg.repetition_penalty

        return r

    def terminal_reward(self, board: chess.Board, last_mover_won: bool):
        if board.is_checkmate():
            return self.cfg.win if last_mover_won else self.cfg.loss
        if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
            return self.cfg.draw
        return 0.0

def board_to_planes(board: chess.Board) -> np.ndarray:
    planes = np.zeros((8,8,12), dtype=np.int8)
    type2idx = {chess.PAWN:0, chess.KNIGHT:1, chess.BISHOP:2, chess.ROOK:3, chess.QUEEN:4, chess.KING:5}
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if not p:
            continue
        r, c = divmod(sq, 8)
        base = 0 if p.color == chess.WHITE else 6
        planes[r, c, base + type2idx[p.piece_type]] = 1
    return planes

def legal_moves_mask(board: chess.Board) -> np.ndarray:
    mask = np.zeros(64*64, dtype=np.int8)
    for mv in board.legal_moves:
        idx = 64 * mv.from_square + mv.to_square
        mask[idx] = 1
    return mask

def decode_action_to_move(board: chess.Board, action_idx: int) -> chess.Move | None:
    from_sq = action_idx // 64
    to_sq = action_idx % 64
    mv = chess.Move(from_sq, to_sq)
    # handle promotion not encoded here; will be illegal if required
    return mv if mv in board.legal_moves else None

class ChessEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, reward_cfg: RewardConfig | None = None, illegal_action_mode: str = "penalize_and_random", max_moves: int = 512):
        super().__init__()
        self.board = chess.Board()
        self.rew = RewardEngine(reward_cfg or RewardConfig())
        self.action_space = spaces.Discrete(64*64)
        self.observation_space = spaces.Box(low=0, high=1, shape=(8,8,12), dtype=np.int8)
        self.illegal_action_mode = illegal_action_mode
        self.max_moves = max_moves
        self.reset()

    def _obs(self):
        return board_to_planes(self.board).astype(np.int8)

    def _mask(self):
        return legal_moves_mask(self.board)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.board.reset()
        self.rew.reset()
        self.move_count = 0
        obs = self._obs()
        info = {"action_mask": self._mask()}
        return obs, info

    def step(self, action: int):
        info = {}
        mv = decode_action_to_move(self.board, action)
        illegal_pen = 0.0
        if mv is None:
            if self.illegal_action_mode == "reject":
                obs = self._obs()
                info["action_mask"] = self._mask()
                return obs, -0.01, False, False, info
            legal = list(self.board.legal_moves)
            if legal:
                mv = random.choice(legal)
            illegal_pen = -0.01

        board_before = self.board.copy()
        last_player = self.board.turn  # True=white
        self.board.push(mv)
        self.move_count += 1
        board_after = self.board

        terminated = False
        term_rew = 0.0
        if (board_after.is_checkmate() or board_after.is_stalemate() or
            board_after.is_insufficient_material() or board_after.is_seventyfive_moves() or
            board_after.is_fivefold_repetition()):
            terminated = True
            last_mover_won = board_after.is_checkmate()
            term_rew = self.rew.terminal_reward(board_after, last_mover_won)

        step_rew = self.rew.step_reward(board_before, mv, board_after)
        reward = step_rew + term_rew + illegal_pen

        obs = self._obs()
        truncated = self.move_count >= self.max_moves
        info["action_mask"] = self._mask()
        return obs, float(reward), terminated, truncated, info

    def render(self):
        return self.board.unicode()  # textual

    def close(self):
        pass