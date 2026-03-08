import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def gn(ch: int, groups: int = 32) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=min(groups, ch), num_channels=ch, eps=1e-6, affine=True)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, groups=32, dropout=0):
        super().__init__()
        self.norm1 = gn(in_ch, groups)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = gn(out_ch, groups)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)


class DownsampleBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size, stride, padding)

    def forward(self, x):
        return self.conv(x)


class UpsampleBlock(nn.Module):
    def __init__(self, channels, groups=32, dropout=0.0):
        super().__init__()
        self.norm1 = gn(channels, groups)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = gn(channels, groups)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def _forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h

    def forward(self, x):
        if self.training:
            return checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)

class AttentionBlock(nn.Module):
    """
    Swin-style window attention block (drop-in replacement).

    Uses sliding window attention instead of global attention.
    Much cheaper for large feature maps.
    """

    def __init__(
        self,
        channels,
        groups=32,
        num_heads=8,
        window_size=8,
        shift=False
    ):
        super().__init__()

        assert channels % num_heads == 0

        self.norm = gn(channels, groups)

        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)

        self.proj = nn.Conv2d(channels, channels, 1)

        self.num_heads = num_heads
        self.window_size = window_size
        self.shift = shift

    def window_partition(self, x):
        B, C, H, W = x.shape
        ws = self.window_size

        x = x.view(
            B,
            C,
            H // ws,
            ws,
            W // ws,
            ws
        )

        windows = x.permute(
            0, 2, 4, 1, 3, 5
        ).reshape(-1, C, ws, ws)

        return windows

    def window_reverse(self, windows, H, W):
        ws = self.window_size
        B = int(windows.shape[0] / (H * W / ws / ws))

        x = windows.view(
            B,
            H // ws,
            W // ws,
            -1,
            ws,
            ws
        )

        x = x.permute(
            0, 3, 1, 4, 2, 5
        ).reshape(B, -1, H, W)

        return x

    def _forward(self, x):

        res = x
        B, C, H, W = x.shape

        x = self.norm(x)

        # shifted windows
        if self.shift:
            shift = self.window_size // 2
            x = torch.roll(x, shifts=(-shift, -shift), dims=(2, 3))

        # partition windows
        windows = self.window_partition(x)

        b, c, ws, _ = windows.shape
        head_dim = c // self.num_heads

        q = self.q(windows).view(b, self.num_heads, head_dim, ws * ws).transpose(2, 3)
        k = self.k(windows).view(b, self.num_heads, head_dim, ws * ws).transpose(2, 3)
        v = self.v(windows).view(b, self.num_heads, head_dim, ws * ws).transpose(2, 3)

        attn = F.scaled_dot_product_attention(q, k, v)

        out = attn.transpose(2, 3).reshape(b, c, ws, ws)
        out = self.proj(out)

        # merge windows
        x = self.window_reverse(out, H, W)

        # reverse shift
        if self.shift:
            shift = self.window_size // 2
            x = torch.roll(x, shifts=(shift, shift), dims=(2, 3))

        return res + x

    def forward(self, x):
        if self.training:
            return checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)


# ---------------------------------------------------------------------------
# Naming scheme — every block gets a stable descriptive key:
#
#   encoder:
#     enc_scale{s}_res{r}         residual block at scale s, position r
#     enc_scale{s}_attn{r}        attention after res block s/r
#     enc_scale{s}_down           downsample at end of scale s
#     enc_bottleneck_pre          bottleneck res before attention
#     enc_bottleneck_attn         bottleneck attention
#     enc_bottleneck_post         bottleneck res after attention
#
#   decoder: (same pattern, dec_ prefix)
#     dec_bottleneck_pre/attn/post
#     dec_scale{s}_res{r}
#     dec_scale{s}_attn{r}
#     dec_scale{s}_up
#
# Adding/removing attn_resolutions or changing num_res only adds/removes
# keys — existing keys are never renamed. strict=False handles this cleanly.
# ---------------------------------------------------------------------------

class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_ch=3,
        base_ch=128,
        ch_mult=(1,2,4),
        num_res=2,
        z_ch=4,
        attn_resolutions=(32,),
        input_res=256,
        dropout=0,
        groups=32
    ):
        super().__init__()

        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        self.scales = nn.ModuleList()

        cur_ch = base_ch
        cur_res = input_res

        for s, m in enumerate(ch_mult):

            stage = nn.ModuleList()
            out_ch = base_ch * m

            for r in range(num_res):

                stage.append(
                    ResBlock(cur_ch, out_ch, groups=groups, dropout=dropout)
                )

                cur_ch = out_ch

                if cur_res in attn_resolutions:
                    stage.append(
                        AttentionBlock(cur_ch, groups=groups)
                    )

            stage.append(
                DownsampleBlock(cur_ch)
            )

            self.scales.append(stage)

            cur_res //= 2

        self.mid = nn.ModuleList([
            ResBlock(cur_ch, cur_ch),
            AttentionBlock(cur_ch, shift=False),
            ResBlock(cur_ch, cur_ch),
            AttentionBlock(cur_ch, shift=True),
            ResBlock(cur_ch, cur_ch),
        ])

        self.out_norm = gn(cur_ch)
        self.out_conv = nn.Conv2d(cur_ch, 2*z_ch, 3, padding=1)

    def forward(self, x):

        x = self.in_conv(x)

        for stage in self.scales:
            for block in stage:
                x = block(x)

        for block in self.mid:
            x = block(x)

        x = self.out_norm(x)
        x = self.out_conv(x)

        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        out_ch=3,
        base_ch=128,
        ch_mult=(1, 2, 4),
        num_res=2,
        z_ch=4,
        attn_resolutions=(32,),
        input_res=256,
        dropout=0,
        groups=32,
    ):
        super().__init__()

        ch_mult_rev = list(ch_mult)[::-1]
        cur_ch = base_ch * ch_mult_rev[0]

        # input projection
        self.in_conv = nn.Conv2d(z_ch, cur_ch, 3, padding=1)

        # -------------------------------------------------
        # bottleneck
        # -------------------------------------------------

        self.mid = nn.ModuleList([
            ResBlock(cur_ch, cur_ch, groups=groups, dropout=dropout),
            AttentionBlock(cur_ch, shift=False),
            ResBlock(cur_ch, cur_ch, groups=groups, dropout=dropout),
            AttentionBlock(cur_ch, shift=True),
            ResBlock(cur_ch, cur_ch, groups=groups, dropout=dropout),  # NEW
        ])

        # -------------------------------------------------
        # multi-scale decoder
        # -------------------------------------------------

        self.scales = nn.ModuleList()

        cur_res = input_res // (2 ** len(ch_mult))

        for i, mult in enumerate(ch_mult_rev):

            stage = nn.ModuleList()

            out_ch_r = base_ch * mult

            # final scale gets extra residual blocks (same as your original)
            n_res = num_res + (2 if i == len(ch_mult_rev) - 1 else 0)

            for r in range(n_res):

                stage.append(
                    ResBlock(cur_ch, out_ch_r, groups=groups, dropout=dropout)
                )

                cur_ch = out_ch_r

                if cur_res in attn_resolutions:
                    stage.append(
                        AttentionBlock(cur_ch, groups=groups)
                    )

            # upsample
            stage.append(
                UpsampleBlock(cur_ch, groups=groups, dropout=dropout)
            )

            self.scales.append(stage)

            cur_res *= 2

        # -------------------------------------------------
        # output
        # -------------------------------------------------

        self.out_norm = gn(cur_ch, groups)
        self.out_conv = nn.Conv2d(cur_ch, out_ch, 3, padding=1)
        
        self.refine = nn.Sequential(
            nn.Conv2d(out_ch, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, out_ch, 3, padding=1),
        )

    def forward(self, x):

        x = self.in_conv(x)

        # bottleneck
        for block in self.mid:
            x = block(x)

        # multi-scale decoding
        for stage in self.scales:
            for block in stage:
                x = block(x)

        x = self.out_norm(x)
        x = F.silu(x)
        x = self.out_conv(x)
        ref = self.refine(x)
        x = x + ref

        return x

# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------

class VAE(nn.Module):
    def __init__(
        self,
        z_ch=4,
        base_ch=128,
        ch_mult=(1, 2, 4),
        enc_num_res=2,
        dec_num_res=3,
        attn_resolutions=(32,),
        input_res=256,
        dropout=0,
        groups=32,
    ):
        super().__init__()
        self.encoder = EncoderBlock(
            in_ch=3,
            base_ch=base_ch,
            ch_mult=ch_mult,
            num_res=enc_num_res,
            z_ch=z_ch,
            attn_resolutions=attn_resolutions,
            input_res=input_res,
            dropout=dropout,
            groups=groups,
        )
        self.decoder = DecoderBlock(
            out_ch=3,
            base_ch=base_ch,
            ch_mult=ch_mult,
            num_res=dec_num_res,
            z_ch=z_ch,
            attn_resolutions=attn_resolutions,
            input_res=input_res,
            dropout=dropout,
            groups=groups,
        )

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        else:
            return mu

    def encode(self, x):
        enc_out = self.encoder(x)
        mu, logvar = torch.chunk(enc_out, 2, dim=1)
        logvar = -10 + F.softplus(logvar + 10)
        logvar = 10 - F.softplus(10 - logvar)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z):
        return self.decoder(z)

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def parameter_summary(self):
        enc = sum(p.numel() for p in self.encoder.parameters())
        dec = sum(p.numel() for p in self.decoder.parameters())
        total = enc + dec

        return (
            f"\n[VAE parameters]\n"
            f"total:    {total:,}\n"
            f"encoder:  {enc:,}\n"
            f"decoder:  {dec:,}\n"
            f"trainable:{self.num_trainable_parameters:,}\n"
        )
    
    def forward(self, x):
        z, mu, logvar = self.encode(x)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar, z