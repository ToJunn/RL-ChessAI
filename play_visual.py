from __future__ import annotations

import argparse
import json
import os
import sys
import math
from typing import Dict, Optional, Tuple, List

import numpy as np
import pygame as pg
import torch as th
import chess
import chess.engine 

# --- Local imports (v4) ---
from envv4 import ChessEnv, move_index, BASE_ACTION_DIM, PROMO_KINDS, index_to_move_64x64x5
from modelv2 import ActorCriticV2, ExtractorConfig
from sb3_contrib import MaskablePPO

# -------------------- Colors & Theme (Giữ nguyên) --------------------
COL_BG = (24, 26, 27)
COL_PANEL = (32, 34, 36)
TILE_LIGHT = (240, 217, 181)
TILE_DARK  = (181, 136, 99)
HL_MOVE    = (246, 246, 105, 120)
HL_CHECK   = (255, 80, 80, 140)
COL_TEXT   = (233, 233, 235)
BORDER     = (60, 60, 62)

# -------------------- Utility: Elo/Value (Cập nhật) --------------------
def value_to_elo(v: float, cap: int = 800) -> float:
    """Quy đổi Value [-1, 1] (dự đoán thắng) sang Elo."""
    v = float(max(-1.0, min(1.0, v)))
    s = 0.5 * (v + 1.0)
    s = float(np.clip(s, 1e-6, 1-1e-6))
    elo = -400.0 * np.log10(1.0 / s - 1.0)
    return float(max(-cap, min(cap, elo)))

def centipawn_to_elo(cp: int, cap: int = 800) -> float:
    """Quy đổi điểm centipawn (từ Stockfish) sang Elo."""
    # Công thức Lichess: 2 / (1 + exp(-0.004 * cp)) - 1
    # Sau đó chuyển value [-1, 1] này sang elo
    v = 2 / (1 + math.exp(-0.00368 * cp)) - 1
    return value_to_elo(v, cap)

# -------------------- Piece images (Giữ nguyên) --------------------
SYMBOL_TO_NAME = {
    'P': 'white_pawn',   'N': 'white_knight', 'B': 'white_bishop', 'R': 'white_rook', 'Q': 'white_queen', 'K': 'white_king',
    'p': 'black_pawn',   'n': 'black_knight', 'b': 'black_bishop', 'r': 'black_rook', 'q': 'black_queen', 'k': 'black_king',
}

class PieceImages:
    # (Toàn bộ class PieceImages giữ nguyên)
    def __init__(self, cell: int, img_dir: str = './img', mapping_json: Optional[str] = None):
        self.cell = int(cell)
        self.imgs: Dict[str, pg.Surface] = {}
        mapping = SYMBOL_TO_NAME.copy()
        if mapping_json and os.path.isfile(mapping_json):
            try:
                user_map = json.load(open(mapping_json, 'r', encoding='utf-8'))
                for k, v in user_map.items():
                    mapping[k] = v
            except Exception: pass
        for sym, basename in mapping.items():
            path = basename
            if not os.path.isfile(path):
                path = os.path.join(img_dir, f"{basename}.png")
            try:
                img = pg.image.load(path).convert_alpha()
                img = pg.transform.smoothscale(img, (self.cell, self.cell))
                self.imgs[sym] = img
            except Exception as e:
                surf = pg.Surface((self.cell, self.cell), pg.SRCALPHA)
                surf.fill((80, 80, 80, 200))
                self.imgs[sym] = surf
                print(f"[WARN] Could not load piece image: {path} -> {e}")

    def draw(self, screen: pg.Surface, symbol: str, x: int, y: int):
        img = self.imgs.get(symbol)
        if img:
            screen.blit(img, (x, y))

# -------------------- Agents (Cập nhật) --------------------
class PPOAgent:
    def __init__(self, path: str, device: str = 'cpu', dummy_env = None):
        self.model = MaskablePPO.load(path, device=device, env=dummy_env)
        self.device = device
    
    @th.no_grad()
    def choose(self, obs: np.ndarray, mask: np.ndarray, board: chess.Board) -> Tuple[int, float]:
        a, _ = self.model.predict(obs, deterministic=True, action_masks=mask)
        try:
            obs_t = th.as_tensor(obs, dtype=th.float32).unsqueeze(0).to(self.device)
            v = self.model.policy.predict_values(obs_t).detach().cpu().item()
        except Exception:
            v = 0.0
        return int(a), v

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
            print(f"LỖI TẢI A3C MODEL: {e}")
            print(">>> Hãy đảm bảo các tham số --base_channels, --proj_dim, --obs_stack_size KHỚP với file checkpoint!")
            raise e
        self.net.eval()
        
    @th.no_grad()
    def choose(self, obs: np.ndarray, mask: np.ndarray, board: chess.Board) -> Tuple[int, float]:
        obs_t = th.as_tensor(obs, dtype=th.float32).unsqueeze(0).to(self.device)
        logits, value = self.net(obs_t)
        logits = logits.squeeze(0)
        mask_t = th.from_numpy(mask).to(self.device)
        logits = logits.masked_fill(mask_t <= 0, float('-inf'))
        a = int(th.argmax(logits).item())
        v = float(value.squeeze().item())
        return a, v

class StockfishAgent:
    def __init__(self, path: str, elo: int, movetime: int, device: str = 'cpu'):
        self.movetime_sec = movetime / 1000.0
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(path)
        except FileNotFoundError:
            print(f"LỖI: Không tìm thấy Stockfish tại '{path}'")
            print("Hãy tải Stockfish và đặt đường dẫn --stockfish_path chính xác.")
            sys.exit(1)
        
        # Đặt Elo
        try:
            self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
            print(f"Stockfish Elo set to: {elo}")
        except Exception:
            skill = max(0, min(20, int((elo - 1000) / 100)))
            self.engine.configure({"Skill Level": skill})
            print(f"Stockfish Skill Level set to: {skill} (ước tính ~{elo} Elo)")

    def choose(self, obs: np.ndarray, mask: np.ndarray, board: chess.Board) -> Tuple[chess.Move, float]:
        """Trả về (move, value)"""
        limit = chess.engine.Limit(time=self.movetime_sec)
        
        # Yêu cầu Stockfish phân tích VÀ tìm nước đi
        info = self.engine.analyse(board, limit, info=chess.engine.INFO_ALL)
        
        move = info.get("pv", [None])[0] # Lấy nước đi tốt nhất
        score = info.get("score")

        if move is None or score is None:
            # Fallback nếu có lỗi
            result = self.engine.play(board, limit)
            move = result.move
            score = self.engine.analyse(board, limit)["score"]

        # Chuyển đổi điểm (score) thành value
        v = 0.0
        pov_score = score.relative
        if pov_score.is_mate():
            v = 1.0 if pov_score.moves > 0 else -1.0
        else:
            cp = pov_score.cp
            if cp is not None:
                # Dùng hàm mới
                v_elo = centipawn_to_elo(cp)
                # Chuyển đổi ngược elo -> value
                v = (value_to_elo(v_elo/800.0) / 800.0) # Ước tính
                # Đơn giản hơn:
                v = (2 / (1 + math.exp(-0.00368 * cp)) - 1)

        return move, v

    def quit(self):
        self.engine.quit()

# -------------------- Drawing helpers (Giữ nguyên) --------------------
def _alpha_surface(w: int, h: int) -> pg.Surface:
    return pg.Surface((w, h), pg.SRCALPHA)

def draw_board(screen: pg.Surface, board, cell: int, margin: int, pieces: PieceImages, last_move, flipped: bool) -> None:
    # (Toàn bộ code draw_board giữ nguyên)
    board_px = 2*margin + cell*8
    pg.draw.rect(screen, BORDER, (0, 0, board_px, board_px), border_radius=8)
    pg.draw.rect(screen, (0,0,0), (2, 2, board_px-4, board_px-4), border_radius=8)
    for r in range(8):
        for c in range(8):
            rr = 7 - r if not flipped else r
            cc = c if not flipped else 7 - c
            x = margin + cc * cell
            y = margin + r * cell
            color = TILE_LIGHT if (rr + cc) % 2 == 0 else TILE_DARK
            pg.draw.rect(screen, color, (x, y, cell, cell))
    if last_move is not None:
        ov = _alpha_surface(board_px, board_px)
        for s in [last_move.from_square, last_move.to_square]:
            sr, sc = divmod(s, 8)
            if flipped: sr, sc = 7 - sr, 7 - sc
            x = margin + sc * cell
            y = margin + (7 - sr) * cell if not flipped else margin + sr * cell
            pg.draw.rect(ov, HL_MOVE, (x, y, cell, cell))
        screen.blit(ov, (0, 0))
    if board.is_check():
        king_sq = board.king(board.turn)
        if king_sq is not None:
            sr, sc = divmod(king_sq, 8)
            if flipped: sr, sc = 7 - sr, 7 - sc
            x = margin + sc * cell
            y = margin + (7 - sr) * cell if not flipped else margin + sr * cell
            ov = _alpha_surface(board_px, board_px)
            pg.draw.rect(ov, HL_CHECK, (x, y, cell, cell))
            screen.blit(ov, (0,0))
    for sq in range(64):
        p = board.piece_at(sq)
        if not p: continue
        sym = p.symbol()
        sr, sc = divmod(sq, 8)
        rr, cc = (7 - sr, 7 - sc) if flipped else (sr, sc)
        x = margin + cc * cell
        y = margin + (7 - rr) * cell if not flipped else margin + rr * cell
        pieces.draw(screen, sym, x, y)


def draw_side_panel(screen: pg.Surface, font: pg.font.Font, lines: List[str], width: int, height: int):
    # (Toàn bộ code draw_side_panel giữ nguyên)
    pg.draw.rect(screen, COL_PANEL, (width - 320, 0, 320, height))
    x = width - 310
    y = 12
    for ln in lines[-34:]:
        t = font.render(ln, True, COL_TEXT)
        screen.blit(t, (x, y))
        y += 20

# -------------------- Main loop (CẬP NHẬT) --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ppo_path', type=str, default='')
    ap.add_argument('--a3c_path', type=str, default='')
    # THÊM 'stockfish'
    ap.add_argument('--white', type=str, choices=['ppo','a3c','stockfish','random'], default='ppo')
    ap.add_argument('--black', type=str, choices=['ppo','a3c','stockfish','random'], default='a3c')
    
    ap.add_argument('--device', type=str, default='auto')
    ap.add_argument('--fps', type=int, default=2)
    ap.add_argument('--size', type=int, default=980)
    ap.add_argument('--margin', type=int, default=28)
    ap.add_argument('--flip', action='store_true')
    ap.add_argument('--strict_load', action='store_true')
    ap.add_argument('--start_fen', type=str, default='')
    ap.add_argument('--img_map', type=str, default='')
    
    # THÊM args cho Stockfish
    ap.add_argument('--stockfish_path', type=str, default='./stockfish/stockfish-windows-x86-64-avx2.exe')
    ap.add_argument('--stockfish_elo', type=int, default=1400)
    ap.add_argument('--stockfish_movetime', type=int, default=500, help="ms")
    
    # extractor (Args giữ nguyên)
    ap.add_argument('--proj_dim', type=int, default=512)
    ap.add_argument('--base_channels', type=int, default=64)
    ap.add_argument('--n_res_blocks', type=int, default=3)
    ap.add_argument('--se', action='store_true')
    ap.add_argument('--dropout', type=float, default=0.0)
    
    # BẮT BUỘC
    ap.add_argument('--obs_stack_size', type=int, default=0, help="Bắt buộc (ví dụ: 4)")
    
    args = ap.parse_args()

    if args.obs_stack_size <= 0 and (args.white in ('ppo', 'a3c') or args.black in ('ppo', 'a3c')):
        print("LỖI: Bạn phải cung cấp --obs_stack_size (ví dụ: --obs_stack_size 4) khi dùng model PPO hoặc A3C.")
        sys.exit(1)

    device = 'cuda' if (args.device=='auto' and th.cuda.is_available()) else (args.device if args.device!='auto' else 'cpu')

    # FIX: Khởi tạo Dummy Env TRƯỚC TIÊN
    obs_config = {
        "flip_perspective": args.flip, # Dùng flip từ args
        "obs_stack_size": args.obs_stack_size if args.obs_stack_size > 0 else 1 # Fallback
    }
    dummy_env = ChessEnv(
        illegal_action_mode="reject",
        enable_promotion_actions=True,  # BẮT BUỘC
        enable_auto_queen=False,        # BẮT BUỘC
        observation_config=obs_config,
        start_fen=(args.start_fen or None)
    )
    # Thêm hàm helper
    def _decode_action_index(self, move: chess.Move) -> int:
        promo_id = PROMO_KINDS.index(move.promotion) if move.promotion in PROMO_KINDS else 0
        return (move_index(move.from_square, move.to_square) * 5) + promo_id
    ChessEnv._decode_action_index = _decode_action_index
    def _safe_decode_action(self, action_idx: int) -> Optional[chess.Move]:
        if self.enable_promotion_actions:
            return index_to_move_64x64x5(action_idx, self.board)
        return self._index_to_move_4096(action_idx)
    ChessEnv._safe_decode_action = _safe_decode_action


    print(f"Loading models with Obs Space: {dummy_env.observation_space.shape}, Action Space: {dummy_env.action_space.n}")

    env = ChessEnv(
        illegal_action_mode="reject",
        enable_promotion_actions=True,
        enable_auto_queen=False,
        observation_config=obs_config,
        start_fen=(args.start_fen or None)
    )

    # Agents (Thêm Stockfish)
    extractor_cfg = ExtractorConfig(base_channels=args.base_channels, n_res_blocks=args.n_res_blocks,
                                    use_se=args.se, dropout=args.dropout, proj_dim=args.proj_dim)
    
    agents = {}
    if args.white == 'ppo':
        assert args.ppo_path, 'Need --ppo_path for PPO agent'
        agents['white'] = PPOAgent(args.ppo_path, device=device, dummy_env=dummy_env)
    elif args.white == 'a3c':
        assert args.a3c_path, 'Need --a3c_path for A3C agent'
        agents['white'] = A3CAgent(args.a3c_path, dummy_env.observation_space, dummy_env.action_space.n, 
                               device=device, extractor_cfg=extractor_cfg, strict=args.strict_load)
    elif args.white == 'stockfish':
        agents['white'] = StockfishAgent(args.stockfish_path, args.stockfish_elo, args.stockfish_movetime, device=device)
    else:
        agents['white'] = None # Random

    if args.black == 'ppo':
        assert args.ppo_path, 'Need --ppo_path for PPO agent'
        agents['black'] = PPOAgent(args.ppo_path, device=device, dummy_env=dummy_env)
    elif args.black == 'a3c':
        assert args.a3c_path, 'Need --a3c_path for A3C agent'
        agents['black'] = A3CAgent(args.a3c_path, dummy_env.observation_space, dummy_env.action_space.n, 
                               device=device, extractor_cfg=extractor_cfg, strict=args.strict_load)
    elif args.black == 'stockfish':
        agents['black'] = StockfishAgent(args.stockfish_path, args.stockfish_elo, args.stockfish_movetime, device=device)
    else:
        agents['black'] = None # Random

    # Pygame init (Giữ nguyên)
    pg.init()
    margin = args.margin
    board_px = args.size - 320
    cell = (board_px - 2*margin) // 8
    board_px = 2*margin + cell*8
    win_w = board_px + 320
    win_h = board_px

    screen = pg.display.set_mode((win_w, win_h))
    pg.display.set_caption('AI Chess Visualizer (PPO, A3C, Stockfish)')
    info_font = pg.font.SysFont('Consolas', 16)
    pieces = PieceImages(cell, img_dir='./img', mapping_json=(args.img_map or None))
    clock = pg.time.Clock()
    paused = False
    flipped = args.flip

    def agent_name(side: str):
        return {'ppo':'PPO','a3c':'A3C','stockfish':f'Stockfish (Elo {args.stockfish_elo})','random':'Random'}[side]
    header = [
        f"White: {agent_name(args.white)}",
        f"Black: {agent_name(args.black)}",
        f"FPS: {args.fps} | SPACE pause  R reset  F flip",
        "",
    ]
    lines: List[str] = header.copy()
    obs, info = env.reset()
    last_move = None
    step_request = False

    try:
        while True:
            # Event handling (Giữ nguyên)
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    raise SystemExit
                elif event.type == pg.KEYDOWN:
                    if event.key in (pg.K_ESCAPE, pg.K_q):
                        raise SystemExit
                    elif event.key == pg.K_SPACE:
                        paused = not paused
                    elif event.key == pg.K_f:
                        flipped = not flipped
                    elif event.key == pg.K_r:
                        obs, info = env.reset()
                        last_move = None
                        lines = header.copy()
                        paused = False
                    elif event.key == pg.K_n and paused:
                        step_request = True

            # Vẽ (Giữ nguyên)
            screen.fill(COL_BG)
            draw_board(screen, env.board, cell, margin, pieces, last_move, flipped)
            draw_side_panel(screen, info_font, lines, win_w, win_h)

            # Step (CẬP NHẬT logic)
            clock.tick(max(1, 30 if paused else args.fps))
            
            if not paused or step_request:
                if step_request:
                    step_request = False
                    paused = True

                if env.board.is_game_over(claim_draw=True):
                    res = env.board.result(claim_draw=True)
                    if not lines or not lines[-1].startswith("Result:"):
                        lines.append("")
                        lines.append(f"Result: {res}  (R to reset)")
                else:
                    mask = env.get_action_mask()
                    pre_board = env.board.copy()
                    
                    action_idx: Optional[int] = None
                    move: Optional[chess.Move] = None
                    v: float = 0.0
                    
                    current_agent = agents['white'] if env.board.turn else agents['black']
                    agent_type = args.white if env.board.turn else args.black

                    if agent_type in ('ppo', 'a3c'):
                        action_idx, v = current_agent.choose(obs, mask, env.board)
                        move = env._safe_decode_action(action_idx)
                        elo_str = f"({value_to_elo(v):+0.0f} Elo)" # Dự đoán
                    elif agent_type == 'stockfish':
                        move, v = current_agent.choose(obs, mask, env.board)
                        action_idx = env._decode_action_index(move)
                        elo_str = f"({value_to_elo(v):+0.0f} Elo)" # Phân tích
                    else: # Random
                        legal = np.where(mask > 0)[0]
                        action_idx = int(np.random.choice(legal)) if len(legal)>0 else 0
                        move = env._safe_decode_action(action_idx)
                        elo_str = "(Random)"
                    
                    # Thực thi
                    if action_idx is None or move is None:
                        print("Lỗi: Agent không trả về nước đi hợp lệ.")
                        paused = True
                    else:
                        obs, reward, term, trunc, step_info = env.step(action_idx)
                        last_move = move # Dùng move từ Stockfish/PPO
                        
                        try: san = pre_board.san(last_move)
                        except Exception: san = str(last_move)
                        
                        if pre_board.turn: # Lượt Trắng vừa đi
                            lines.append(f"{pre_board.fullmove_number}. {san}  {elo_str}")
                        else: # Lượt Đen vừa đi
                            if lines and lines[-1].startswith(f"{pre_board.fullmove_number}. "):
                                lines[-1] += f"   {san}  {elo_str}"
                            else:
                                lines.append(f"{pre_board.fullmove_number}. ... {san}  {elo_str}")

                        if term or trunc:
                            res = step_info.get('result', env.board.result(claim_draw=True))
                            lines.append("")
                            lines.append(f"Result: {res}  (R to reset)")

            pg.display.flip()

    except SystemExit:
        print("Đang thoát...")
    except Exception as e:
        print('LỖI:', e)
        import traceback
        traceback.print_exc()
    finally:
        # Dọn dẹp
        if agents.get('white') and isinstance(agents['white'], StockfishAgent):
            agents['white'].quit()
        if agents.get('black') and isinstance(agents['black'], StockfishAgent):
            agents['black'].quit()
        pg.quit()
        sys.exit(0)


if __name__ == '__main__':
    main()