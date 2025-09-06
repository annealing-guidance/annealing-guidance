import torch
import torch.nn as nn
from typing import Union


class ScalarMLP(nn.Module):
    """
    MLP that outputs per-example guidance scales given (timestep, lambda, delta),
    where delta = (noise_pred_uncond - noise_pred_text) and delta_norm = ||delta||_2 per example.

    forward(...) returns guidance scales.
    """

    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        n_layers: int,
        # Embedding sizes
        t_embed_dim: int = 4,
        delta_embed_dim: int = 4,
        lambda_embed_dim: int = 4,
        # Normalizations applied before embedding
        t_embed_normalization: float = 1e3,
        delta_embed_normalization: float = 5.0,
        # Final affine on head output
        w_bias: float = 1.0,
        w_scale: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__()

        input_size = t_embed_dim + delta_embed_dim + lambda_embed_dim

        # Head
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(n_layers):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers += [nn.Linear(hidden_size, output_size)]
        self.combined_head = nn.Sequential(*layers)

        # Config
        self.t_embed_dim = t_embed_dim
        self.delta_embed_dim = delta_embed_dim
        self.lambda_embed_dim = lambda_embed_dim

        self.t_embed_normalization = t_embed_normalization
        self.delta_embed_normalization = delta_embed_normalization

        self.w_bias = w_bias
        self.w_scale = w_scale

    # ---------- helpers ----------

    @staticmethod
    def _ensure_batched(x: Union[float, int, torch.Tensor], B: int, device, dtype) -> torch.Tensor:
        """Make x a length-B tensor on (device, dtype)."""
        x = torch.as_tensor(x, device=device, dtype=dtype)
        return x.expand(B) if x.dim() == 0 else x

    @staticmethod
    def _embed_value(value: torch.Tensor, n_embeds: int) -> torch.Tensor:
        """
        Positional-like embedding: [value, cos(value*1), ..., cos(value*(n_embeds-1))].
        Expects value shape: (B,)
        Returns: (B, n_embeds)
        """
        i = torch.arange(1, n_embeds, device=value.device, dtype=value.dtype)
        cosines = torch.cos(value.unsqueeze(-1) * i)
        return torch.cat([value.unsqueeze(-1), cosines], dim=-1)

    # ---------- forward ----------

    def forward(
        self,
        timestep: Union[float, int, torch.Tensor],
        l: Union[float, int, torch.Tensor],
        noise_pred_uncond: torch.Tensor,
        noise_pred_text: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            timestep: scalar or (B,)
            l:        scalar or (B,)
            noise_pred_uncond: (B, C, H, W)
            noise_pred_text:   (B, C, H, W)

        Returns:
            guidance_scales: (B,) if output_size==1 else (B, output_size)
        """
        # 1) Compute delta and its norm per example
        delta = noise_pred_uncond - noise_pred_text            # (B, C, H, W)
        B = delta.shape[0]
        delta_norm = delta.view(B, -1).norm(dim=1)             # (B,)

        # 2) Unify device/dtype & batch shapes for scalar inputs
        device, dtype = delta_norm.device, delta_norm.dtype
        timestep = self._ensure_batched(timestep, B, device, dtype)  # (B,)
        l        = self._ensure_batched(l,        B, device, dtype)  # (B,)

        # 3) Build features with embeddings
        t_feat = self._embed_value(timestep / self.t_embed_normalization, self.t_embed_dim)           # (B, t_embed_dim)
        d_feat = self._embed_value(delta_norm / self.delta_embed_normalization, self.delta_embed_dim) # (B, delta_embed_dim)
        l_feat = self._embed_value(l, self.lambda_embed_dim)        # (B, lambda_embed_dim)

        features = torch.cat([t_feat, d_feat, l_feat], dim=-1)  # (B, input_size)

        # 4) Head → scales
        guidance_scale = self.combined_head(features)                   # (B, output_size)
        guidance_scale = self.w_scale * guidance_scale + self.w_bias

        # Squeeze to (B,) for single-output models
        if guidance_scale.shape[1] == 1:
            guidance_scale = guidance_scale.squeeze(-1)
        return guidance_scale
