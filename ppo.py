import os
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
# Giả sử file model.py đã được tạo và chứa ChessCNN
from model import ChessCNN 
# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor # Không cần thiết nếu import từ model
import torch as th
import torch.nn as nn

# Loại bỏ định nghĩa class ChessCNN cũ.

def make_ppo(env, tb_log_dir: str, device="auto", total_timesteps=100_000, seed=0):
    """
    Tạo và cấu hình mô hình MaskablePPO sử dụng ChessCNN làm Features Extractor.
    """
    os.makedirs(tb_log_dir, exist_ok=True)
    
    # Sử dụng ChessCNN đã import từ model.py
    policy_kwargs = dict(features_extractor_class=ChessCNN, 
                         features_extractor_kwargs=dict(features_dim=256),
                         net_arch=dict(pi=[256,128], vf=[256,128]))
    
    model = MaskablePPO(
        policy=MaskableActorCriticPolicy,
        env=env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.005,
        vf_coef=0.5,
        tensorboard_log=tb_log_dir,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device, # Sử dụng thiết bị được chỉ định (cuda hoặc cpu)
        seed=seed,
    )
    return model