import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

def gn(ch: int, groups: int = 32) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=min(groups, ch), num_channels=ch, eps=1e-6, affine=True)

class TimeStepEmbedding(nn.Module):
    def __init__(self, d_in, d_out, T=1000):
        super().__init__()
        self.embed = nn.Embedding(T, d_in)
        self.transform = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.SiLU(),
            nn.Linear(d_out, d_out)
        )

    def forward(self, t):
        return self.transform(self.embed(t))
    
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim, groups=32, dropout=0.0):
        super().__init__()
        self.norm1 = gn(in_ch, groups)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj  = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, out_ch * 2))
        self.norm2 = gn(out_ch, groups)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def _forward(self, x: torch.Tensor, t_emb) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.t_proj(t_emb).chunk(2, dim=1) # Adaptive Group Norm Dhariwal and Nichol
        h = self.norm2(h) * scale[..., None, None] + shift[..., None, None]
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)

    def forward(self, x, t_emb) -> torch.Tensor:
        if self.training:
            return checkpoint(self._forward, x, t_emb, use_reentrant=False)
        return self._forward(x, t_emb)

class AttentionBlock(nn.Module):
    def __init__(self, channels, groups=32, num_heads=8):
        super().__init__()
        assert channels % num_heads == 0
        self.norm = gn(channels, groups)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.num_heads = num_heads
        self.head_dim = channels // self.num_heads

    def _forward(self, x):
        b, c, h, w = x.shape
        res = x
        x = self.norm(x)
        q = self.q(x).view(b, self.num_heads, self.head_dim, h * w).transpose(2, 3)
        k = self.k(x).view(b, self.num_heads, self.head_dim, h * w).transpose(2, 3)
        v = self.v(x).view(b, self.num_heads, self.head_dim, h * w).transpose(2, 3)
        attn = F.scaled_dot_product_attention(q, k, v)
        out = attn.transpose(2, 3).contiguous().view(b, c, h, w)
        return res + self.proj(out)

    def forward(self, x):
        if self.training:
            return checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)
    
class CrossAttentionBlock(nn.Module):
    def __init__(self, channels, ctx_dim=512, num_heads=8, groups=32):
        super().__init__()
        self.norm_x = gn(channels, groups)
        self.norm_ctx = nn.LayerNorm(ctx_dim)
        self.q = nn.Linear(channels, channels)
        self.k = nn.Linear(ctx_dim, channels)
        self.v = nn.Linear(ctx_dim, channels)
        self.proj = nn.Linear(channels, channels)
        self.num_heads = num_heads
        self.head_dim = channels // self.num_heads
    
    def _forward(self, x, ctx):
        b, c, h, w = x.shape
        flat = h * w
        residual = x
        x = self.norm_x(x).view(b, c, flat).permute(0,2,1) # (B, HW, C)
        ctx = self.norm_ctx(ctx)
        T_seq = ctx.shape[1]
        q = self.q(x).view(b, flat, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(ctx).view(b, T_seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(ctx).view(b, T_seq, self.num_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)
        out = attn.transpose(1, 2).contiguous().view(b, flat, c)
        out = self.proj(out).permute(0, 2, 1).view(b, c, h, w)
        return residual + out
    
    def forward(self, x, ctx):
        if self.training:
            return checkpoint(self._forward, x, ctx, use_reentrant=False)
        return self._forward(x, ctx)
    
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
    
class ResAttnBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim, use_attn=True, ctx_dim=512, num_heads=8, groups=32, dropout=0.0):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, t_dim, groups, dropout)
        self.cross_attn = CrossAttentionBlock(out_ch, ctx_dim, num_heads, groups) if use_attn else None
        self.self_attn = AttentionBlock(out_ch, groups, num_heads) if use_attn else None

    def forward(self, x, t_emb, ctx):
        x = self.res(x, t_emb)
        if self.cross_attn:
            x = self.cross_attn(x, ctx)
            x = self.self_attn(x)
        return x
    
class UNET(nn.Module):
    def __init__(
        self,
        in_ch = 16,
        base_ch = 128,
        ch_mult = (1,2,4),
        num_res=4,
        attn_res = (8, 16, 32),
        context_dim = 512,
        num_heads = 8,
        dropout = 0.1,
        latent_spatial_dim = 32,
        groups = 32,
    ):
        super().__init__()
        self.ch_mult = ch_mult
        self.num_res = num_res
        self.attn_res = attn_res
        t_dim = base_ch * 4
        self.t_embed = TimeStepEmbedding(base_ch, t_dim)
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        self.enc_blocks = []
        self.enc_downs = []
        self.skip_ch = []

        self.dec_blocks = []
        self.dec_ups = []
        
        cur_ch = base_ch
        cur_res = latent_spatial_dim
        self.skip_ch.append(cur_ch)

        for level, mult in enumerate(ch_mult):
            out_ch = base_ch * mult
            for _ in range(num_res):
                self.enc_blocks.append(ResAttnBlock(
                    cur_ch, 
                    out_ch, 
                    t_dim, 
                    cur_res in attn_res, 
                    context_dim, 
                    num_heads, 
                    groups, 
                    dropout
                ))
                cur_ch = out_ch
                self.skip_ch.append(cur_ch)
            
            if level < len(ch_mult) - 1:
                self.enc_downs.append(DownsampleBlock(cur_ch))
                cur_res //= 2
                self.skip_ch.append(cur_ch)

        self.enc_blocks = nn.ModuleList(self.enc_blocks)
        self.enc_downs  = nn.ModuleList(self.enc_downs)
        
        self.bottleneck_res1 = ResBlock(cur_ch, cur_ch, t_dim, groups, dropout)
        self.bottleneck_cross_attn = CrossAttentionBlock(cur_ch, context_dim, num_heads, groups)
        self.bottleneck_attn = AttentionBlock(cur_ch, groups, num_heads)
        self.bottleneck_res2 = ResBlock(cur_ch, cur_ch, t_dim, groups, dropout)

        for level, mult in reversed(list(enumerate(ch_mult))):
            out_ch = base_ch * mult
            for _ in range(num_res + 1):
                self.dec_blocks.append(ResAttnBlock(
                    cur_ch + self.skip_ch.pop(), 
                    out_ch, 
                    t_dim, 
                    cur_res in attn_res, 
                    context_dim, 
                    num_heads, 
                    groups, 
                    dropout
                ))
                cur_ch = out_ch
                
            if level > 0:
                self.dec_ups.append(UpsampleBlock(cur_ch, groups))
                cur_res *= 2

        self.dec_blocks = nn.ModuleList(self.dec_blocks)
        self.dec_ups    = nn.ModuleList(self.dec_ups)
        self.conv_out = nn.Conv2d(cur_ch, in_ch, 3, padding=1)
    
    def forward(self, x, t, ctx):
        t_emb = self.t_embed(t)
        h = self.in_conv(x)
        skips = [h]
        n_levels = len(self.ch_mult)
        encoder_index = 0
        downsample_index = 0

        for level in range(n_levels):
            for _ in range(self.num_res):
                h = self.enc_blocks[encoder_index](h, t_emb, ctx)
                encoder_index += 1
                skips.append(h)
            if level < n_levels - 1:
                h = self.enc_downs[downsample_index](h)
                downsample_index += 1
                skips.append(h)
        
        h = self.bottleneck_res1(h, t_emb)
        h = self.bottleneck_cross_attn(h, ctx)
        h = self.bottleneck_attn(h)
        h = self.bottleneck_res2(h, t_emb)

        decoder_index = 0
        upsample_index = 0

        for level in reversed(range(n_levels)):
            for _ in range(self.num_res + 1):
                h = torch.cat([h, skips.pop()], dim=1)
                h = self.dec_blocks[decoder_index](h, t_emb, ctx)
                decoder_index += 1
            if level > 0:
                h = self.dec_ups[upsample_index](h)
                upsample_index += 1
        
        return self.conv_out(F.silu(h))
        

        


        
        