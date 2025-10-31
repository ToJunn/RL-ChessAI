from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import chess
import chess.pgn

# ==========================
# Constants & helpers
# ==========================
BASE_ACTION_DIM = 64 * 64  # from*64 + to
PROMO_KINDS = (None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)  # id: 0..4
DEFAULT_CENTER_SQUARES = (chess.E4, chess.D4, chess.E5, chess.D5)

PIECE_VALUES_CAPTURE_KEY = {
    chess.PAWN: "capture_p",
    chess.KNIGHT: "capture_n",
    chess.BISHOP: "capture_b",
    chess.ROOK: "capture_r",
    chess.QUEEN: "capture_q",
    chess.KING: "capture_k",
}

def move_index(from_sq: int, to_sq: int) -> int:
    return from_sq * 64 + to_sq

# (Hàm index_to_move_4096 BỊ XÓA, vì nó cần 'self' để check auto_queen)

def index_to_move_64x64x5(idx: int, board: chess.Board) -> Optional[chess.Move]:
    """Hàm này giữ nguyên, dùng cho không gian action 20480"""
    from_to = idx // 5
    promo_id = idx % 5
    src, dst = divmod(from_to, 64)
    promo = PROMO_KINDS[promo_id]
    mv = chess.Move(src, dst, promotion=promo)
    if mv in board.legal_moves:
        return mv
    return None

# ==========================
# Configs
# ==========================
@dataclass
class RewardConfig:
    step: float = -0.001
    illegal: float = -0.01
    win: float = 1.0
    loss: float = -1.0
    draw: float = 0.0
    # capture rewards
    capture_p: float = 0.01
    capture_n: float = 0.03
    capture_b: float = 0.03
    capture_r: float = 0.05
    capture_q: float = 0.09
    capture_k: float = 0.10
    # positional (NEW)
    center_occupy_pawn: float = 0.01  # Thưởng lớn 1 lần khi Tốt chiếm trung tâm
    center_attack: float = 0.002      # Thưởng nhỏ khi tấn công trung tâm
    center_reward_decay_ply: int = 40 # Ngừng thưởng trung tâm sau 40 ply
    shaping_clip: float = 0.02
    scale: float = 1.0


@dataclass
class TerminationConfig:
    max_fullmoves: int = 80
    max_plies: Optional[int] = None
    repeat_draw_threshold: int = 7
    use_fide_stalemate: bool = False
    use_fide_insufficient: bool = False
    use_fide_50move: bool = False


@dataclass
class RepetitionPenaltyConfig:
    # (Giữ nguyên như v3.1)
    base: float = 0.001
    grace: int = 2
    cap: float = 0.2
    recent_window: int = 12
    backtrack: float = 0.01
    forbid_immediate_backtrack: bool = True
    twofold_extra: float = 0.003


@dataclass
class NoveltyConfig:
    # (Giữ nguyên như v3.1)
    enabled: bool = True
    mode: str = "binary"
    recent_window: int = 16
    bonus: float = 0.002
    count_k: float = 0.01
    cap_per_game: float = 0.2


@dataclass
class ObservationConfig:
    include_meta: bool = True
    flip_perspective: bool = False
    obs_stack_size: int = 4  # NEW: Số frame lịch sử


# ==========================
# Observation encoder
# ==========================

def _get_obs_frame(board: chess.Board,
                   cfg: ObservationConfig,
                   last_move: Optional[chess.Move]) -> np.ndarray:
    """
    Tạo ra MỘT frame observation (C,H,W) tại một thời điểm.
    (Đây là code cũ của _board_to_planes)
    """
    def _rc(sq: int) -> Tuple[int, int]:
        r, c = divmod(sq, 8)
        return r, c

    def _set(planes: np.ndarray, idx: int, sq: int, val: float = 1.0):
        r, c = _rc(sq)
        planes[idx, r, c] = val

    num_meta_planes = 9 if cfg.include_meta else 0
    num_base_planes = 12 + num_meta_planes
    planes = np.zeros((num_base_planes, 8, 8), dtype=np.float32)

    # 12 piece planes
    order = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    for i, pt in enumerate(order):
        for sq in board.pieces(pt, chess.WHITE):
            _set(planes, i, sq)
        for sq in board.pieces(pt, chess.BLACK):
            _set(planes, 6 + i, sq)

    if cfg.include_meta:
        planes[12] = 1.0 if board.turn == chess.WHITE else 0.0
        planes[13] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
        planes[14] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
        planes[15] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
        planes[16] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
        if board.ep_square is not None:
            _set(planes, 17, board.ep_square)
        if last_move is not None:
            _set(planes, 18, last_move.from_square)
            _set(planes, 19, last_move.to_square)
        hm = min(board.halfmove_clock, 100) / 100.0
        planes[20] = hm

    if cfg.flip_perspective and board.turn == chess.BLACK:
        planes = np.rot90(planes, k=2, axes=(1, 2)).copy()

    return planes


# ==========================
# Env
# ==========================
class ChessEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 5}

    def __init__(
        self,
        *,
        illegal_action_mode: str = "reject",
        enable_promotion_actions: bool = False,
        enable_auto_queen: bool = True, # NEW: Tự động phong Hậu nếu dùng 4096 space
        reward_config: Optional[Dict[str, float]] = None,
        repetition_penalty: Optional[Dict[str, float]] = None,
        novelty_config: Optional[Dict[str, Any]] = None,
        termination_config: Optional[Dict[str, Any]] = None,
        observation_config: Optional[Dict[str, Any]] = None,
        save_pgn_path: Optional[str] = None,
        start_fen: Optional[str] = None,
        seed: Optional[int] = None,
        center_squares: Tuple[int, int, int, int] = DEFAULT_CENTER_SQUARES,
    ) -> None:
        super().__init__()
        assert illegal_action_mode in ("reject", "auto")
        self.illegal_action_mode = illegal_action_mode

        # Configs
        self.rcfg = RewardConfig(**(reward_config or {}))
        self.tcfg = TerminationConfig(**(termination_config or {}))
        self.pcfg = RepetitionPenaltyConfig(**(repetition_penalty or {}))
        self.ncfg = NoveltyConfig(**(novelty_config or {}))
        self.ocfg = ObservationConfig(**(observation_config or {}))

        # Board
        self.board = chess.Board()
        self.start_fen = start_fen
        self.last_move: Optional[chess.Move] = None

        # Action/Obs spaces
        self.enable_promotion_actions = enable_promotion_actions
        self.enable_auto_queen = enable_auto_queen
        self.action_space = spaces.Discrete(
            BASE_ACTION_DIM * 5 if enable_promotion_actions else BASE_ACTION_DIM
        )

        # NEW: Observation stacking
        self.ocfg.obs_stack_size = max(1, int(self.ocfg.obs_stack_size))
        self._base_obs_c = 12 + (9 if self.ocfg.include_meta else 0)
        self._obs_frame_shape = (self._base_obs_c, 8, 8)
        self._empty_obs_frame = np.zeros(self._obs_frame_shape, dtype=np.float32)
        
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(self._base_obs_c * self.ocfg.obs_stack_size, 8, 8),
            dtype=np.float32
        )
        self._obs_history: collections.deque[np.ndarray] = collections.deque(
            maxlen=self.ocfg.obs_stack_size
        )

        # Repetition tracking
        self._rep_counts: Dict[str, int] = {}
        self._rep_history: List[str] = []
        self._repeat_penalty_cum: float = 0.0

        # Novelty tracking
        self._novelty_cum: float = 0.0

        # Center control
        self.center_ctrl_cooldown: int = 3
        self.center_squares = center_squares
        self._center_last_reward_ply = { (chess.WHITE, s): -999 for s in self.center_squares }
        self._center_last_reward_ply.update({ (chess.BLACK, s): -999 for s in self.center_squares })
        self._attack_cache_ply_id: int = -1
        self._attack_cache: Dict[Tuple[bool, int], bool] = {}

        # Logging
        self.save_pgn_path = save_pgn_path
        self._moves_for_pgn: List[chess.Move] = []
        self.last_reward_breakdown: Dict[str, float] = {}

        # Internal
        self._ply: int = 0
        self.np_random, _ = gym.utils.seeding.np_random(seed)

    # ---------- Small helpers ----------
    def _is_immediate_backtrack(self, mv: chess.Move) -> bool:
        lm = self.last_move
        return lm is not None and mv.from_square == lm.to_square and mv.to_square == lm.from_square

    def get_action_mask(self) -> np.ndarray:
        if self.enable_promotion_actions:
            mask = np.zeros(BASE_ACTION_DIM * 5, dtype=np.float32)
            for mv in self.board.legal_moves:
                promo_id = 0
                if mv.promotion in PROMO_KINDS:
                    promo_id = PROMO_KINDS.index(mv.promotion)
                idx = (move_index(mv.from_square, mv.to_square) * 5) + promo_id
                mask[idx] = 1.0
        else:
            mask = np.zeros(BASE_ACTION_DIM, dtype=np.float32)
            for mv in self.board.legal_moves:
                mask[move_index(mv.from_square, mv.to_square)] = 1.0

        if getattr(self.pcfg, "forbid_immediate_backtrack", False) and self.last_move is not None:
            # (Logic này giữ nguyên)
            forbid_from = self.last_move.to_square
            forbid_to   = self.last_move.from_square
            if self.enable_promotion_actions:
                base_idx = move_index(forbid_from, forbid_to) * 5
                mask[base_idx:base_idx+5] = 0.0
            else:
                mask[move_index(forbid_from, forbid_to)] = 0.0
        return mask

    def _index_to_move_4096(self, idx: int) -> Optional[chess.Move]:
        """
        NEW: Hàm này giờ là một method, và nó check self.enable_auto_queen
        """
        src, dst = divmod(idx, 64)
        mv = chess.Move(src, dst)
        
        if mv in self.board.legal_moves:
            return mv
            
        # Check if promotion is needed
        is_promo_sq = chess.square_rank(dst) in (0, 7)
        piece = self.board.piece_type_at(src)
        is_pawn = (piece == chess.PAWN)
        
        if is_pawn and is_promo_sq:
            if self.enable_auto_queen:
                # Chỉ thử phong Hậu
                mvp = chess.Move(src, dst, promotion=chess.QUEEN)
                if mvp in self.board.legal_moves:
                    return mvp
            # Nếu enable_auto_queen=False, hoặc phong Hậu không hợp lệ,
            # nước đi này bị coi là illegal (return None)
            
        return None

    def _decode_action(self, action: int) -> Optional[chess.Move]:
        if self.enable_promotion_actions:
            return index_to_move_64x64x5(action, self.board)
        # Sử dụng method mới
        return self._index_to_move_4096(action)

    def _get_current_frame(self) -> np.ndarray:
        """Helper lấy frame observation HIỆN TẠI"""
        return _get_obs_frame(self.board, self.ocfg, self.last_move)

    def _obs(self) -> np.ndarray:
        """NEW: Ghép các frame trong lịch sử lại"""
        # Đảm bảo deque luôn đầy
        while len(self._obs_history) < self.ocfg.obs_stack_size:
             self._obs_history.appendleft(self._empty_obs_frame.copy())
        # Ghép (N, C, H, W) -> (N*C, H, W)
        return np.concatenate(list(self._obs_history), axis=0)

    def _random_legal_action(self) -> Optional[int]:
        # (Giữ nguyên)
        mask = self.get_action_mask()
        legal = np.where(mask > 0)[0]
        if len(legal) == 0:
            return None
        return int(self.np_random.choice(legal))

    # ---------- Reward shaping ----------
    def _capture_bonus(self, mv: chess.Move) -> float:
        # (Giữ nguyên)
        if not self.board.is_capture(mv):
            return 0.0
        if self.board.is_en_passant(mv):
            cap_type = chess.PAWN
        else:
            tgt = self.board.piece_at(mv.to_square)
            cap_type = tgt.piece_type if tgt else None
        if cap_type is None: return 0.0
        key = PIECE_VALUES_CAPTURE_KEY.get(cap_type)
        return getattr(self.rcfg, key) if key else 0.0

    def _attacked_cached(self, color_bool: bool, square: int) -> bool:
        # (Giữ nguyên)
        if self._attack_cache_ply_id != self._ply:
            self._attack_cache.clear()
            self._attack_cache_ply_id = self._ply
        key = (color_bool, square)
        if key in self._attack_cache:
            return self._attack_cache[key]
        has = len(self.board.attackers(color_bool, square)) > 0
        self._attack_cache[key] = has
        return has

    def _center_attack_bonus(self, mover: bool) -> float:
        """
        NEW: Chỉ thưởng cho "tấn công" và có decay theo thời gian
        """
        # Ngừng thưởng nếu đã qua giai đoạn khai cuộc/trung cuộc
        if self._ply >= self.rcfg.center_reward_decay_ply:
            return 0.0
            
        bonus = 0.0
        for sq in self.center_squares:
            if self._attacked_cached(mover, sq):
                last_ply = self._center_last_reward_ply[(mover, sq)]
                if self._ply - last_ply >= self.center_ctrl_cooldown:
                    bonus += self.rcfg.center_attack # Dùng config mới
                    self._center_last_reward_ply[(mover, sq)] = self._ply
        return bonus

    def _apply_repetition_penalty(self, fen_key: str) -> float:
        # (Giữ nguyên)
        recent = self._rep_history[:-1][-self.pcfg.recent_window:]
        cnt_recent = sum(1 for k in recent if k == fen_key)
        over = max(0, cnt_recent - self.pcfg.grace)
        if over <= 0: return 0.0
        raw_pen = float(self.pcfg.base) * float(over)
        allow = max(0.0, float(self.pcfg.cap) - float(self._repeat_penalty_cum))
        actual = min(allow, raw_pen)
        self._repeat_penalty_cum += actual
        return -actual

    def _apply_novelty_bonus(self, fen_key: str) -> float:
        # (Giữ nguyên)
        if not self.ncfg.enabled: return 0.0
        recent_window = max(1, int(self.ncfg.recent_window))
        recent = self._rep_history[:-1][-recent_window:]
        freq = sum(1 for k in recent if k == fen_key)
        if self.ncfg.mode == "count":
            bonus_raw = self.ncfg.count_k / float(np.sqrt(1.0 + freq))
        else:
            bonus_raw = self.ncfg.bonus if freq == 0 else 0.0
        allow = max(0.0, float(self.ncfg.cap_per_game) - float(self._novelty_cum))
        actual = min(allow, max(0.0, float(bonus_raw)))
        self._novelty_cum += actual
        return actual

    # ---------- Termination checks ----------
    def _terminal_info(self) -> Tuple[bool, float, Optional[str], Optional[str]]:
        # (Giữ nguyên)
        if self.board.is_checkmate():
            res = "1-0" if not self.board.turn else "0-1"
            return True, (self.rcfg.win), res, "checkmate"
        if self.tcfg.max_plies is not None and self._ply >= int(self.tcfg.max_plies):
            return True, self.rcfg.draw, "1/2-1/2", "max_plies"
        if self.board.fullmove_number > self.tcfg.max_fullmoves:
            return True, self.rcfg.draw, "1/2-1/2", "max_fullmoves"
        if self._rep_counts.get(self.board.shredder_fen(), 0) > self.tcfg.repeat_draw_threshold:
            return True, self.rcfg.draw, "1/2-1/2", "repeat_draw"
        if self.tcfg.use_fide_stalemate and self.board.is_stalemate():
            return True, self.rcfg.draw, "1/2-1/2", "stalemate"
        if self.tcfg.use_fide_insufficient and self.board.is_insufficient_material():
            return True, self.rcfg.draw, "1/2-1/2", "insufficient_material"
        if self.tcfg.use_fide_50move and self.board.is_fifty_moves():
            return True, self.rcfg.draw, "1/2-1/2", "fifty_move"
        return False, 0.0, None, None

    # ---------- Gym API ----------
    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.board.reset()
        start_fen = options.get("start_fen") if isinstance(options, dict) else None
        if start_fen:
            try: self.board.set_fen(start_fen)
            except Exception: pass
        elif getattr(self, "start_fen", None):
            try: self.board.set_fen(self.start_fen)
            except Exception: pass

        self.last_move = None
        self._ply = 0
        self._repeat_penalty_cum = 0.0
        self._novelty_cum = 0.0
        self._rep_counts.clear()
        self._rep_history.clear()
        self._moves_for_pgn.clear()
        self._attack_cache.clear(); self._attack_cache_ply_id = -1
        self.last_reward_breakdown = {}

        fen_key = self.board.shredder_fen()
        self._rep_counts[fen_key] = 1
        self._rep_history.append(fen_key)
        
        # NEW: Khởi tạo observation stack
        self._obs_history.clear()
        current_frame = self._get_current_frame()
        for _ in range(self.ocfg.obs_stack_size):
            self._obs_history.append(current_frame.copy())

        obs = self._obs()
        info = {"action_mask": self.get_action_mask()}
        return obs, info

    def step(self, action: int):
        info: Dict[str, Any] = {}
        rb = {
            "step": 0.0,
            "capture": 0.0,
            "center_occupy": 0.0, # NEW
            "center_attack": 0.0, # NEW
            "backtrack": 0.0,
            "repetition": 0.0,
            "novelty": 0.0,
            "scaled_clipped": 0.0,
            "terminal": 0.0,
        }
        reward = 0.0
        terminated = False
        truncated = False
        mover = self.board.turn

        mv = self._decode_action(action)
        if mv is None:
            if self.illegal_action_mode == "reject":
                obs = self._obs() # Vẫn trả về obs stack hiện tại
                info["action_mask"] = self.get_action_mask()
                rb["step"] = self.rcfg.illegal
                self.last_reward_breakdown = rb
                return obs, float(self.rcfg.illegal), False, False, info
            # auto:
            legal_idx = self._random_legal_action()
            if legal_idx is None:
                terminated = True
                rb["terminal"] = self.rcfg.draw
                reward += self.rcfg.draw
                info["result"] = "1/2-1/2"
                # Vẫn phải cập nhật obs stack trước khi return
                new_frame = self._get_current_frame()
                self._obs_history.append(new_frame)
                obs = self._obs()
                info["action_mask"] = self.get_action_mask()
                info["reward_breakdown"] = rb
                self.last_reward_breakdown = rb
                return obs, reward, terminated, truncated, info
            reward += self.rcfg.illegal
            rb["step"] += self.rcfg.illegal
            mv = self._decode_action(legal_idx)

        if self._is_immediate_backtrack(mv):
            rb["backtrack"] -= float(self.pcfg.backtrack)
            
        cap_bonus = self._capture_bonus(mv)
        rb["capture"] += cap_bonus

        # NEW: Thưởng chiếm trung tâm bằng Tốt (trước khi push)
        if (self._ply < self.rcfg.center_reward_decay_ply and
            mv.to_square in self.center_squares):
            piece_type = self.board.piece_type_at(mv.from_square)
            if piece_type == chess.PAWN:
                 rb["center_occupy"] = self.rcfg.center_occupy_pawn

        # Apply move
        self.board.push(mv)
        self._moves_for_pgn.append(mv)
        self.last_move = mv
        self._ply += 1

        # NEW: Thưởng tấn công trung tâm (sau khi push)
        center_bonus = self._center_attack_bonus(mover)
        rb["center_attack"] += center_bonus

        # Base shaping
        base = (self.rcfg.step + rb["capture"] + 
                rb["center_occupy"] + rb["center_attack"] + 
                rb["backtrack"]) * self.rcfg.scale
        base = float(np.clip(base, -self.rcfg.shaping_clip, self.rcfg.shaping_clip))
        rb["scaled_clipped"] = base
        reward += base

        # Repetition bookkeeping
        fen_key = self.board.shredder_fen()
        self._rep_counts[fen_key] = self._rep_counts.get(fen_key, 0) + 1
        self._rep_history.append(fen_key)

        # Novelty / Repetition
        novelty = self._apply_novelty_bonus(fen_key)
        rb["novelty"] += novelty
        reward += novelty
        rep_pen = self._apply_repetition_penalty(fen_key)
        if self.board.is_repetition(2) or self.board.can_claim_threefold_repetition():
            rep_pen -= float(self.pcfg.twofold_extra)
        rb["repetition"] += rep_pen
        reward += rep_pen

        # Terminal?
        terminated, term_rew, res, done_reason = self._terminal_info()
        if terminated:
            reward += term_rew
            rb["terminal"] += term_rew
            if res is not None: info["result"] = res
            if done_reason is not None: info["done_reason"] = done_reason

        # NEW: Cập nhật obs stack trước khi return
        new_frame = self._get_current_frame()
        self._obs_history.append(new_frame)
        obs = self._obs()

        info["action_mask"] = self.get_action_mask()
        info["reward_breakdown"] = rb
        self.last_reward_breakdown = rb

        if terminated and getattr(self, "save_pgn_path", None):
            # (Logic ghi PGN giữ nguyên)
            try:
                game = chess.pgn.Game()
                node = game
                b = chess.Board()
                for m in self._moves_for_pgn:
                    node = node.add_variation(m)
                    b.push(m)
                with open(self.save_pgn_path, "a", encoding="utf-8") as f:
                    print(game, file=f)
            except Exception:
                pass

        return obs, reward, terminated, truncated, info

    def render(self):
        return str(self.board)

__all__ = [
    "ChessEnv",
    "move_index",
    "BASE_ACTION_DIM",
]