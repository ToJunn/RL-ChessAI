import os
import time
import argparse
import torch
from env import ChessEnv
from ppo import make_ppo

# Đã sửa: Import ActionMasker từ đường dẫn chung (wrappers)
from sb3_contrib.common.wrappers import ActionMasker 
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

def make_env_fn(seed: int = 0):
    """
    Tạo một hàm closure để khởi tạo môi trường cờ vua với các wrapper cần thiết.
    """
    def _thunk():
        env = ChessEnv()
        env.reset(seed=seed)
        env = Monitor(env)  # record episode stats
        # ActionMasker expects a callable that returns the mask for that env instance
        env = ActionMasker(env, lambda e: e._mask())
        return env
    return _thunk

def main():
    parser = argparse.ArgumentParser(description="PPO Training Script for Chess.")
    
    parser.add_argument("--timesteps", 
                        type=int, 
                        default=200_000, 
                        help="Total number of timesteps to train the model.")
                        
    parser.add_argument("--logdir", 
                        type=str, 
                        default="./logs/ppo/PPO_1", 
                        help="Directory for saving logs and tensorboard data.")
                        
    parser.add_argument("--seed", 
                        type=int, 
                        default=0, 
                        help="Random seed for reproducibility.")
                        
    parser.add_argument("--device", 
                        type=str, 
                        default="auto", 
                        help="Device to use for training (e.g., 'auto', 'cuda', 'cpu', 'cuda:0').")
                        
    parser.add_argument("--n_envs", 
                        type=int, 
                        default=4, # Tinh chỉnh default = 4 để so sánh với A3C workers
                        help="Number of parallel environments (vectorized environments).")
                        
    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)

    # Xác định thiết bị
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    print("Using device:", device)
    print(f"Using {args.n_envs} parallel environments (n_envs).")

    # Tạo các môi trường song song
    env_fns = [make_env_fn(args.seed + i) for i in range(args.n_envs)]
    vec_env = DummyVecEnv(env_fns) 
    vec_env = VecMonitor(vec_env, os.path.join(args.logdir, "monitor.csv")) 

    # Tạo model (make_ppo sử dụng kiến trúc chung ChessCNN)
    model = make_ppo(vec_env, tb_log_dir=args.logdir, device=device, total_timesteps=args.timesteps, seed=args.seed)

    print("Bắt đầu huấn luyện...")
    try:
        model.learn(total_timesteps=args.timesteps)
    finally:
        # Lưu model và đóng môi trường
        out = os.path.join(args.logdir, f"ppo_final_{args.timesteps}.zip")
        model.save(out)
        print("Saved model to", out)
        vec_env.close()

if __name__ == "__main__":
    main()