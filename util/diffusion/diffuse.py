import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image


class ForwardDiffusion(nn.Module):
    def __init__(self, method="cosine", T=1000):
        super().__init__()
        betas = self._get_betas(T, method)
        alphas = self._get_alphas(betas)
        self.alpha_bar, self.sqrt_alpha_bar, self.sqrt_one_minus_alpha_bar = (
            self._get_alpha_cumprods(alphas)
        )

    @torch.no_grad
    def _get_betas(self, T, method="", s=0.08):
        beta_start = 0.0001
        beta_end = 0.02

        match method:
            case "quadratic":
                betas = torch.linspace(beta_start**0.5, beta_end**0.5, T) ** 2
            case "cosine":
                ts = torch.linspace(0, T, T + 1)
                alpha_bars = torch.cos(((ts / T) + s) / (1 + s) * torch.pi * 0.5) ** 2
                alpha_bars = alpha_bars / alpha_bars[0]
                betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
            case "sigmoid":
                betas = torch.linspace(-6, 6, T)
                betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
            case "linear":
                betas = torch.linspace(beta_start, beta_end, T)
            case _:
                raise ValueError(
                    f"method must be {['quadratic', 'cosine', 'sigmoid', 'linear']}"
                )

        return betas

    @torch.no_grad
    def _get_alphas(self, betas):
        return 1 - betas

    @torch.no_grad
    def _get_alpha_cumprods(self, alphas):
        alpha_bar = torch.cumprod(alphas, dim=0)
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)

        return alpha_bar, sqrt_alpha_bar, sqrt_one_minus_alpha_bar

    @torch.no_grad
    def _get_index_from_list(self, vals, t):
        b = t.shape[0]
        out = vals.gather(-1, t.cpu())
        out = out.reshape(b, 1, 1, 1).to(t.device)
        return out

    @torch.no_grad
    def forward(self, x_0, t):
        x_t = None
        sqrt_alpha_bar = self._get_index_from_list(self.sqrt_alpha_bar, t)
        sqrt_one_minus_alpha_bar = self._get_index_from_list(
            self.sqrt_one_minus_alpha_bar, t
        )

        eps = torch.randn_like(x_0)
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * eps

        return x_t
