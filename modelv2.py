from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ==========================
# Utils & init
# ==========================

def kaiming_init(m: nn.Module):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
        if getattr(m, "weight", None) is not None:
            nn.init.ones_(m.weight)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)


def get_activation(name: str) -> nn.Module:
    name = (name or "relu").lower()
    if name == "gelu":
        return nn.GELU()
    return nn.ReLU(inplace=True)


def make_norm(kind: str, num_channels: int) -> nn.Module:
    kind = (kind or "bn").lower()
    if kind == "ln":
        # Channel-last LayerNorm via permute for conv outputs
        return nn.GroupNorm(1, num_channels)  # LN equivalent for conv (stable)
    if kind == "gn":
        groups = max(1, num_channels // 16)
        return nn.GroupNorm(groups, num_channels)
    return nn.BatchNorm2d(num_channels)


class SqueezeExcite(nn.Module):
    def __init__(self, c: int, r: int = 8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c, max(1, c // r)),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, c // r), c),
            nn.Sigmoid(),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        w = self.fc(x).view(x.size(0), x.size(1), 1, 1)
        return x * w


class ResidualBlock(nn.Module):
    def __init__(self, c: int, *, norm: str = "bn", act: str = "relu", use_se: bool = True, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False)
        self.norm1 = make_norm(norm, c)
        self.act1 = get_activation(act)
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False)
        self.norm2 = make_norm(norm, c)
        self.se = SqueezeExcite(c) if use_se else nn.Identity()
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.out_act = get_activation(act)

    def forward(self, x: th.Tensor) -> th.Tensor:
        y = self.conv1(x)
        y = self.norm1(y)
        y = self.act1(y)
        y = self.drop(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.se(y)
        return self.out_act(x + y)


# ==========================
# Feature Extractor (for PPO)
# ==========================
@dataclass
class ExtractorConfig:
    base_channels: int = 64
    n_res_blocks: int = 3
    use_se: bool = True
    dropout: float = 0.0
    add_coord_conv: bool = True  # add 2 coord channels internally (x,y)
    proj_dim: int = 512
    norm: str = "bn"            # bn | ln(gn1) | gn
    act: str = "relu"           # relu | gelu
    head_layernorm: bool = True  # LayerNorm-like stability at head


class ChessCNNExtractorV2(BaseFeaturesExtractor):
    """CNN extractor for (C=21,H=8,W=8) with residual stack + (optional) CoordConv.

    Architecture:
      Stem: Conv(C(+2), Cb) -> Norm -> Act
      Body: n x ResidualBlock(Cb)
      Head: Flatten -> Linear(proj_dim) -> Act -> Dropout -> (optional GroupNorm(1) as LN)
    """

    def __init__(self, observation_space: spaces.Box, cfg: Optional[ExtractorConfig] = None):
        self.cfg = cfg or ExtractorConfig()
        super().__init__(observation_space, self.cfg.proj_dim)

        assert len(observation_space.shape) == 3, "Expected (C,H,W)"
        c, h, w = observation_space.shape
        in_c = c + 2 if self.cfg.add_coord_conv else c

        self.stem = nn.Sequential(
            nn.Conv2d(in_c, self.cfg.base_channels, kernel_size=3, padding=1, bias=False),
            make_norm(self.cfg.norm, self.cfg.base_channels),
            get_activation(self.cfg.act),
        )

        body: Iterable[nn.Module] = [
            ResidualBlock(self.cfg.base_channels, norm=self.cfg.norm, act=self.cfg.act, use_se=self.cfg.use_se, dropout=self.cfg.dropout)
            for _ in range(self.cfg.n_res_blocks)
        ]
        self.body = nn.Sequential(*body)

        # Head
        self.flat = nn.Flatten()
        self.fc = nn.Linear(self.cfg.base_channels * h * w, self.cfg.proj_dim)
        self.fc_act = get_activation(self.cfg.act)
        self.fc_drop = nn.Dropout(self.cfg.dropout) if self.cfg.dropout > 0 else nn.Identity()
        self.fc_ln = nn.GroupNorm(1, self.cfg.proj_dim) if self.cfg.head_layernorm else nn.Identity()

        self.apply(kaiming_init)

        # Precompute coord channels (H,W), cached as buffers (persistent)
        if self.cfg.add_coord_conv:
            x = th.linspace(0, 1, w).view(1, 1, 1, w).repeat(1, 1, h, 1)  # (1,1,H,W)
            y = th.linspace(0, 1, h).view(1, 1, h, 1).repeat(1, 1, 1, w)  # (1,1,H,W)
            self.register_buffer("coord_x", x.clone().contiguous(), persistent=True)
            self.register_buffer("coord_y", y.clone().contiguous(), persistent=True)

    def _concat_coords(self, x: th.Tensor) -> th.Tensor:
        if not self.cfg.add_coord_conv:
            return x
        b = x.size(0)
        cx = self.coord_x.expand(b, -1, -1, -1)
        cy = self.coord_y.expand(b, -1, -1, -1)
        return th.cat([x, cx, cy], dim=1)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        x = self._concat_coords(obs)
        x = self.stem(x)
        x = self.body(x)
        x = self.flat(x)
        x = self.fc(x)
        x = self.fc_act(x)
        x = self.fc_drop(x)
        x = self.fc_ln(x)  # GN(1, C) acts like LN over channels, robust for small batches
        return x


# ==========================
# PPO policy helper
# ==========================

def make_policy_kwargs_v2(
    features_dim: int = 512,
    net_arch_pi: Tuple[int, ...] = (256,),
    net_arch_vf: Tuple[int, ...] = (256,),
    extractor_cfg: Optional[ExtractorConfig] = None,
) -> Dict:
    """Return policy_kwargs for MaskablePPO with ChessCNNExtractorV2.

    Usage:
        policy_kwargs = make_policy_kwargs_v2(
            features_dim=512,
            net_arch_pi=(256, 128),
            net_arch_vf=(256, 128),
            extractor_cfg=ExtractorConfig(base_channels=64, n_res_blocks=3, dropout=0.1)
        )
    """
    cfg = extractor_cfg or ExtractorConfig(proj_dim=features_dim)
    return {
        "features_extractor_class": ChessCNNExtractorV2,
        "features_extractor_kwargs": {"cfg": cfg},
        "net_arch": {"pi": list(net_arch_pi), "vf": list(net_arch_vf)},
    }


# ==========================
# Shared Actor-Critic (A3C/PPO)
# ==========================
class ActorCriticV2(nn.Module):
    def __init__(
        self,
        obs_space: spaces.Box,
        action_dim: int = 4096,
        extractor_cfg: Optional[ExtractorConfig] = None,
        pi_dim: int = 256,
        vf_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.extractor = ChessCNNExtractorV2(obs_space, cfg=extractor_cfg)
        fdim = self.extractor.cfg.proj_dim
        self.pi = nn.Sequential(
            nn.Linear(fdim, pi_dim), get_activation(self.extractor.cfg.act), nn.Dropout(dropout)
        )
        self.vf = nn.Sequential(
            nn.Linear(fdim, vf_dim), get_activation(self.extractor.cfg.act), nn.Dropout(dropout)
        )
        self.logits = nn.Linear(pi_dim, action_dim)
        self.value = nn.Linear(vf_dim, 1)

        self.apply(kaiming_init)

    def forward(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        feat = self.extractor(obs)
        p = self.pi(feat)
        v = self.vf(feat)
        logits = self.logits(p)
        value = self.value(v).squeeze(-1)
        return logits, value

    @th.no_grad()
    def act(
        self,
        obs: th.Tensor,
        mask: Optional[th.Tensor | None],
        device: str = "cpu",
        deterministic: bool = False,
    ) -> int:
        self.eval()
        obs = obs.to(device)
        logits, _ = self.forward(obs)
        logits = logits.squeeze(0)
        if mask is not None:
            mask_t = mask if isinstance(mask, th.Tensor) else th.from_numpy(mask).to(device)
            logits = logits.masked_fill(mask_t <= 0, float("-inf"))
        if deterministic:
            action = int(th.argmax(logits).item())
        else:
            probs = th.softmax(logits, dim=-1)
            action = int(th.distributions.Categorical(probs).sample().item())
        return action


__all__ = [
    "ChessCNNExtractorV2",
    "ExtractorConfig",
    "make_policy_kwargs_v2",
    "ActorCriticV2",
]
