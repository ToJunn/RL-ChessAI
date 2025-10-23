import argparse
import torch
from a3c import train_a3c

def main():
    parser = argparse.ArgumentParser(description="A3C Training Script for Chess.")
    
    parser.add_argument("--timesteps", 
                        type=int, 
                        default=200_000, 
                        help="Total number of steps (from all workers) for training.")
                        
    parser.add_argument("--workers", 
                        type=int, 
                        default=4, # Đã thống nhất với --n_envs=4 của PPO
                        help="Number of asynchronous worker processes.")
                        
    parser.add_argument("--logdir", 
                        type=str, 
                        default="./logs/a3c", 
                        help="Directory to save logs and the final model.")
                        
    parser.add_argument("--device", 
                        type=str, 
                        default="auto", 
                        help="Device to use for NN computation ('auto', 'cuda', 'cpu', 'cuda:0', etc.).")
    
    args = parser.parse_args()

    # Xử lý tham số device để tự động chọn CUDA nếu có
    device = args.device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Bắt đầu A3C training với {args.workers} workers. Thiết bị tính toán mạng: {device}")
    
    # Gọi hàm huấn luyện chính
    train_a3c(num_workers=args.workers, 
              total_steps=args.timesteps, 
              log_dir=args.logdir, 
              device=device)

if __name__ == "__main__":
    main()