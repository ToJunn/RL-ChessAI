import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Tuple

# --- Feature Extractor (For PPO/SB3) ---
class ChessCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim=features_dim)
        in_channels = observation_space.shape[2]
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            sample = th.zeros((1, in_channels, 8, 8))
            n_flat = self.net(sample).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flat, 512), nn.ReLU(), nn.Linear(512, features_dim), nn.ReLU())
        self._features_dim = features_dim

    def forward(self, obs: th.Tensor) -> th.Tensor:
        x = obs.float().permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = self.net(x)
        x = self.linear(x)
        return x

# --- Actor-Critic Head (For A3C and MCTS Evaluation) ---
class ActorCriticNet(nn.Module):
    def __init__(self, in_c=12, features_dim=256, action_dim=4096):
        super().__init__()
        # Use the same CNN core
        self.cnn = nn.Sequential(nn.Conv2d(in_c, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64,64,3,padding=1), nn.ReLU())
        with th.no_grad():
            dummy = th.zeros(1, in_c, 8, 8)
            n_flat = self.cnn(dummy).view(1,-1).shape[1]
        
        # FC layer to features_dim (256)
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(n_flat, 512), nn.ReLU(), nn.Linear(512, features_dim), nn.ReLU())
        
        # Policy and Value heads
        self.pi = nn.Linear(features_dim, action_dim)
        self.v  = nn.Linear(features_dim, 1)

    def forward(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        # x: (B, C, H, W)
        z = self.cnn(x)
        h = self.fc(z)
        logits = self.pi(h)
        value = self.v(h).squeeze(-1)
        return logits, value