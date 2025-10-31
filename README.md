# ♟️ RL-ChessAI — Reinforcement Learning Chess Engine

**RL-ChessAI** là dự án nghiên cứu và huấn luyện tác nhân AI chơi cờ vua sử dụng các thuật toán **Reinforcement Learning** hiện đại, bao gồm **PPO**, **A3C**, và **MCTS (AlphaZero-style)**.  
Hệ thống được phát triển trong khuôn khổ đồ án tốt nghiệp tại **FPT University HCMC**.

---

## 🧠 Mục tiêu
Xây dựng một mô hình AI có thể:
- Tự học chơi cờ vua qua **self-play**;
- Cải thiện ELO qua nhiều vòng huấn luyện;
- Đối đầu và so sánh hiệu năng với **Stockfish** ở mức Elo xác định;
- Có thể **quan sát trực quan** quá trình thi đấu bằng giao diện Pygame.

---

## 📂 Cấu trúc thư mục

```
RL-ChessAI/
│
├── envv4.py             # Môi trường cờ vua Gymnasium với reward shaping, stacking, repetition & novelty bonus
├── modelv2.py           # Kiến trúc CNN-ResNet + Squeeze-Excite cho Actor-Critic / PPO
│
├── train_ppo.py         # Huấn luyện PPO với MaskablePPO (SB3-Contrib)
├── train_a3c.py         # Huấn luyện A3C song song với multiprocessing
├── train_mcts.py        # Huấn luyện AlphaZero-style MCTS + Replay Buffer
│
├── evaluate_elo.py      # Đánh giá ELO so với Stockfish (dựa trên kết quả ván đấu)
├── play_visual.py       # Giao diện Pygame hiển thị trận đấu giữa các agent (PPO, A3C, Stockfish, Random)
│
├── checkpoints/         # Lưu model đã train (.pt / .zip)
├── logs/                # TensorBoard logs
└── img/                 # Hình quân cờ để visualize
```

---

## ⚙️ Yêu cầu môi trường

```bash
pip install torch torchvision torchaudio
pip install stable-baselines3 sb3-contrib gymnasium pygame tensorboard chess
```

> ⚠️ Nếu muốn đánh giá với **Stockfish**, tải engine:
> - [Stockfish official site](https://stockfishchess.org/download/)
> - hoặc đặt đường dẫn trong tham số `--stockfish_path`.

---

## 🚀 Huấn luyện

### 🧩 PPO (MaskablePPO – Stable-Baselines3)
```bash
python train_ppo.py   --timesteps 1000000   --n_envs 4   --obs_stack_size 4   --proj_dim 384   --base_channels 48   --n_res_blocks 3   --dropout 0.05   --ent_coef 0.003   --eval_freq 25000   --eval_eps 32   --learning_rate 1e-4   --lr_schedule linear   --logdir ./logs/ppo_run   --save ./checkpoints/ppo_model.zip   --best_path ./checkpoints/ppo_best.zip
```

---

### 🧩 A3C (Asynchronous Advantage Actor-Critic)
```bash
python train_a3c.py   --timesteps 1000000   --n_envs 8   --obs_stack_size 4   --proj_dim 512   --base_channels 64   --n_res_blocks 3   --entropy_coef 0.01   --learning_rate 1e-4   --eval_freq 25000   --eval_eps 32   --save ./checkpoints/a3c_model.pt   --best_path ./checkpoints/a3c_best.pt
```

---

### ♜ MCTS / AlphaZero Self-Play
```bash
python train_mcts.py   --n_workers 8   --games_per_worker 8   --n_sim 200   --proj_dim 512   --obs_stack_size 4   --train_steps 100000   --eval_freq 5000   --save ./checkpoints/az_model.pt   --best_path ./checkpoints/az_best.pt
```

---

## 🧮 Đánh giá ELO với Stockfish

```bash
python evaluate_elo.py   --model_path ./checkpoints/ppo_best.zip   --agent_type ppo   --stockfish_path ./stockfish/stockfish.exe   --stockfish_elo 1400   --games 50   --movetime 500   --obs_stack_size 4
```

> Kết quả in ra gồm: **Win / Draw / Loss**, **Score %**, và **ước lượng ELO** của agent.

---

## 🖥️ Giao diện trực quan

Xem trận đấu giữa **PPO**, **A3C**, hoặc **Stockfish** trong thời gian thực:

```bash
python play_visual.py   --ppo_path ./checkpoints/ppo_best.zip   --a3c_path ./checkpoints/a3c_best.pt   --white ppo   --black stockfish   --stockfish_path ./stockfish/stockfish.exe   --stockfish_elo 1400   --obs_stack_size 4   --fps 2
```

**Phím tắt:**
| Phím | Chức năng |
|------|------------|
| `SPACE` | Tạm dừng / tiếp tục |
| `R` | Reset ván đấu |
| `F` | Lật bàn cờ |
| `N` | Bước từng lượt khi tạm dừng |
| `ESC` / `Q` | Thoát |

---

## 🧩 Cấu hình phần thưởng (Reward Shaping)

Tích hợp trong `envv4.py`:

| Thành phần | Mô tả | Mặc định |
|-------------|--------|----------|
| `win/loss/draw` | Thắng +1, Thua −1, Hòa 0 | 1 / −1 / 0 |
| `capture_*` | Thưởng khi ăn quân (p,r,n,b,q,k) | 0.01–0.10 |
| `center_occupy_pawn` | Thưởng khi Tốt chiếm ô trung tâm | 0.01 |
| `center_attack` | Thưởng khi tấn công trung tâm | 0.002 |
| `repetition_penalty` | Phạt lặp lại vị trí | 0.001–0.2 |
| `novelty_bonus` | Thưởng nước đi mới | 0.002 |
| `step_penalty` | Phạt nhỏ mỗi lượt | −0.001 |

---

## 🧩 Kiến trúc mô hình (`modelv2.py`)
- **Backbone:** Residual CNN + SE Block  
- **Feature Extractor:** `ChessCNNExtractorV2`
- **Actor-Critic Head:** `ActorCriticV2`
- **Framework:** PyTorch + SB3
- **Hỗ trợ:** CoordConv, LayerNorm, Dropout, Projection head (512-dim)

---

## 🧠 Giám sát huấn luyện
Chạy TensorBoard:
```bash
tensorboard --logdir ./logs
```

---

## 📊 ELO ước lượng nội bộ
Công thức tính trong `evaluate_elo.py`:
\[
ELO = -400 \times \log_{10}\left(\frac{1}{S} - 1\right)
\]
với \(S\) là tỷ lệ thắng (win + 0.5 × draw).

---

## 👥 Tác giả
**Students – FPT University HCM**

| Thành viên | Vai trò |
|-------------|----------|
| Lê Minh Hùng | Leader, RL System Design |
| Nguyễn Quý Toàn (ToJunn) | Research & Implementation |
| Lê Quang Thật | Training & Evaluation |

---

## 📜 Giấy phép
Dự án dành cho mục đích **nghiên cứu học thuật**, không sử dụng thương mại.  
