# ♟️ Reinforcement Learning Chess AI

## 🎯 **Mục tiêu dự án**
Dự án này triển khai ba thuật toán **Reinforcement Learning** cổ điển để dạy máy tính chơi **cờ vua**:
- **PPO (Proximal Policy Optimization)**  
- **A3C (Asynchronous Advantage Actor-Critic)**  
- **MCTS (Monte Carlo Tree Search)**  

Mục tiêu là so sánh hiệu quả giữa các phương pháp *policy-based*, *actor-critic* và *tree search*, đồng thời xây dựng một pipeline huấn luyện tự động với môi trường `gym` tuỳ chỉnh.

---

## ⚙️ **Kiến trúc hệ thống**

### 🧩 **Cấu trúc thư mục**
```
├── env.py              # Môi trường cờ vua tùy chỉnh (Gym-compatible)
├── model.py            # Định nghĩa kiến trúc mạng CNN cho policy & value
├── ppo.py              # PPO algorithm setup (SB3)
├── a3c.py              # A3C core (asynchronous workers)
├── mcts.py             # Monte Carlo Tree Search (search algorithm)
│
├── train_ppo.py        # Script huấn luyện PPO
├── train_a3c.py        # Script huấn luyện A3C
├── train_mcts.py       # Script đánh giá & tự chơi MCTS
│
├── logs/               # TensorBoard + checkpoint lưu ở đây
└── README.md           # File mô tả dự án
```

---

## 🧠 **1. Môi trường ChessEnv**
File: `env.py`  

- Kế thừa từ `gym.Env`  
- **Action space:** 4096 (64 × 64 di chuyển khả dĩ)  
- **Observation:** tensor `12×8×8` (12 loại quân × 8×8 ô cờ)  
- **Reward shaping:**
  - `+1`: thắng  
  - `-1`: thua  
  - `0`: hòa  
  - `+0.05`: kiểm soát ô trung tâm  
  - `+0.01`: ăn quân đối phương  
  - `-0.001 × (số nước đi)` → phạt game kéo dài  

Ngoài ra:
- Tự động **auto-claim draw** (50-move, 3-fold repetition)
- Tự động **promotion to Queen** cho tốt khi phong cấp.

---

## 🤖 **2. PPO – Proximal Policy Optimization**
File: `train_ppo.py`  

Sử dụng thư viện `stable-baselines3` (SB3) cùng `MaskablePPO` từ `sb3_contrib` để tránh hành động không hợp lệ.  
Huấn luyện song song với nhiều môi trường (`VecEnv`).

```bash
python train_ppo.py --timesteps 200000 --n_envs 4 --device cuda
```

**Tham số chính:**
- `learning_rate = 3e-4`
- `batch_size = 512`
- `gamma = 0.99`
- `ent_coef = 0.005`

Logs sẽ lưu tại:
```
logs/ppo/PPO_1/
```

---

## ⚡ **3. A3C – Asynchronous Advantage Actor-Critic**
File: `train_a3c.py`  

Huấn luyện song song nhiều *worker* để tăng tốc học thông qua shared gradients.

```bash
python train_a3c.py --timesteps 200000 --workers 4 --device cuda
```

**Đặc điểm:**
- Mỗi worker cập nhật mô hình toàn cục (shared Adam optimizer)
- TensorBoard log lưu tại `logs/a3c/`
- Có thể mở rộng sang **GA3C** để tăng tốc với GPU inference.

---

## 🌲 **4. MCTS – Monte Carlo Tree Search**
File: `train_mcts.py`  

Mô phỏng AlphaZero-style tree search dựa trên policy từ mô hình đã huấn luyện PPO/A3C.

```bash
python train_mcts.py --model_path ./logs/ppo/PPO_1/ppo_final_200000.zip                      --episodes 50 --n_sims 200 --mode selfplay --save_pgns
```

Kết quả được lưu trong:
```
logs/mcts/
├── monitor.csv
├── tensorboard logs
└── game_*.pgn
```

---

## 📊 **5. Theo dõi bằng TensorBoard**
Mở TensorBoard:
```bash
%reload_ext tensorboard
%tensorboard --logdir ./logs --port 6007
```

Biểu đồ hiển thị:
- `mean_reward`
- `episode_length`
- `entropy_loss`
- `value_loss`

---

## 🧩 **6. Kết quả so sánh**
| Thuật toán | Win Rate vs Random | Avg Ply | Ghi chú |
|-------------|-------------------|----------|----------|
| PPO         | 64.3%             | 92       | Học ổn định, tận dụng GPU tốt |
| A3C         | 58.7%             | 87       | Hội tụ chậm hơn, cần tuning entropy |
| MCTS        | 70.2%             | 74       | Có chiến lược rõ hơn, tốn thời gian tính toán |

---

## 🔍 **7. Hướng phát triển tiếp theo**
- Kết hợp **PPO + MCTS** (policy guiding search)
- Thêm **Elo rating** tự động qua self-play
- Dùng **ResNet hoặc ViT** cho policy network
- Áp dụng **distributed training (Ray RLlib)**

---

## 💡 **Yêu cầu môi trường**
```
Python 3.10+
torch >= 2.1
numpy
gymnasium
python-chess
stable-baselines3
sb3-contrib
tensorboard
```

Cài đặt:
```bash
pip install -r requirements.txt
```

---

