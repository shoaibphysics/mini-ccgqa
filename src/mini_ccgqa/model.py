"""Transformer models with GQA, CCGQA, and Dynamic CCGQA.

I independently wrote this implementation and checked its logic
against the underlying methods. No AI code-generation assistance
was used here.
"""

import torch
from torch import nn
import torch.nn.functional as F

from mini_ccgqa.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        mean_square = x.square().mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(mean_square + self.eps)
        return self.weight * normalized.to(dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

        std = (2.0 / (d_model + d_ff)) ** 0.5
        for layer in (self.gate_proj, self.up_proj, self.down_proj):
            nn.init.trunc_normal_(layer.weight,mean=0.0,std=std,a=-3.0 * std,b=3.0 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        values = self.up_proj(x)
        return self.down_proj(gate * values)

class HalfRoPE(nn.Module):
    def __init__(self, head_dim: int, theta: float = 10_000.0):
        super().__init__()

        if head_dim <= 0 or head_dim % 4 != 0:
            raise ValueError("head_dim must be positive and divisible by 4.")
        if theta <= 0:
            raise ValueError("theta must be positive.")

        self.rotary_dim = head_dim // 2

        exponents =  torch.arange(0, self.rotary_dim, 2, dtype=torch.float32)/ self.rotary_dim
        inv_freq = theta ** (-exponents)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, T, head_dim)
        # Shape comments below use head_dim=16.
        seq_len = x.shape[-2]
        pair_count = self.rotary_dim // 2  # 4

        positions = torch.arange(seq_len, device=x.device, dtype=torch.float32)  # (T,)
        angles = (positions[:, None] * self.inv_freq.float()[None, :])  # (T, 4)
        cos = angles.cos().to(x.dtype)[None, None, :, :]  # (1, 1, T, 4)
        sin = angles.sin().to(x.dtype)[None, None, :, :]  # (1, 1, T, 4)
        even = x[..., :pair_count]                  # (B, H, T, 4)
        odd = x[..., pair_count:self.rotary_dim]  # (B, H, T, 4)
        unchanged = x[..., self.rotary_dim:]         # (B, H, T, 8)

        # Elementwise rotation of pairs (0,4), (1,5), (2,6), (3,7).
        rotated_even = even * cos - odd * sin   # (B, H, T, 4)
        rotated_odd = even * sin + odd * cos  # (B, H, T, 4)

        return torch.cat((rotated_even, rotated_odd, unchanged), dim=-1 )  # (B, H, T, 16)


class QKConvolution(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        self.q_width = cfg.q_heads * cfg.head_dim
        self.k_width = cfg.kv_heads * cfg.head_dim

        channels = self.q_width + self.k_width  # 96
        heads = cfg.q_heads + cfg.kv_heads      # 6

        # Independent temporal filter for each channel.
        self.depthwise = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=2,
            groups=channels,
            bias=True)

        # Mix channels and time within each Q or K head.
        self.headwise = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=2,
            groups=heads,
            bias=True)

    def forward(self,q: torch.Tensor,k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # q: (B, T, q_width), k: (B, T, k_width)
        combined = torch.cat((q, k), dim=-1)  # (B, T, 96)
        combined = combined.transpose(1, 2)  # (B, 96, T)

        # Pad once for the two kernel-size-2 convolutions.
        combined = F.pad(combined, (2, 0))   # (B, 96, T+2)

        combined = self.depthwise(combined)  # (B, 96, T+1)
        combined = self.headwise(combined)   # (B, 96, T)

        combined = combined.transpose(1, 2)  # (B, T, 96)

        q_conv = combined[..., :self.q_width]  # (B, T, 64)
        k_conv = combined[..., self.q_width:]  # (B, T, 32)

        return q_conv, k_conv

def shared_qk_means(q: torch.Tensor,k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Shared Q/K means from raw projections; CCA Eq. (9).
    https://arxiv.org/pdf/2510.04476v2#page=5
    """
    # q: (B, T, H, d)
    # k: (B, T, K, d)
    B, T, H, d = q.shape
    K = k.shape[2]

    if K <= 0 or H % K != 0:
        raise ValueError("Query heads must be divisible by positive KV heads.")

    G = H // K  # Query heads per KV head.

    repeated_k = k.repeat_interleave(G, dim=2)  # (B, T, H, d)
    query_mean = 0.5 * (q + repeated_k)         # (B, T, H, d)

    grouped_mean = query_mean.reshape(B, T, K, G, d)
    key_mean = grouped_mean.mean(dim=3)        # (B, T, K, d)

    return query_mean, key_mean

def delay_values(v: torch.Tensor) -> torch.Tensor:
    # v: (B, T, K*d), before splitting into KV heads.
    width = v.shape[-1]

    if width % 2 != 0:
        raise ValueError("Value width must be even.")

    half = width // 2

    current = v[..., :half]   # (B, T, K*d/2)
    delayed = v[..., half:]   # (B, T, K*d/2)

    # Remove the last position, then prepend one zero position.
    delayed = F.pad(delayed[:, :-1, :], (0, 0, 1, 0))  # (B, T, K*d/2)

    return torch.cat((current, delayed), dim=-1)  # (B, T, K*d)

class GQAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        if cfg.q_heads <= 0 or cfg.kv_heads <= 0:
            raise ValueError("Head counts must be positive.")
        if cfg.q_heads % cfg.kv_heads != 0:
            raise ValueError("q_heads must be divisible by kv_heads.")

        if not 0.0 <= cfg.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.dropout_p = cfg.dropout

        self.q_heads = cfg.q_heads
        self.kv_heads = cfg.kv_heads
        self.head_dim = cfg.head_dim
        self.group_size = cfg.q_heads // cfg.kv_heads

        self.q_width = cfg.q_heads * cfg.head_dim
        self.kv_width = cfg.kv_heads * cfg.head_dim

        self.q_proj = nn.Linear(cfg.d_model, self.q_width, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, self.kv_width, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, self.kv_width, bias=False)
        self.out_proj = nn.Linear(self.q_width, cfg.d_model, bias=False)

        # Our chosen truncated-normal initialization, separately per matrix.
        for layer in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            std = (2.0 / (layer.in_features + layer.out_features)) ** 0.5
            nn.init.trunc_normal_(layer.weight,mean=0.0,std=std,a=-3.0 * std,b=3.0 * std)

        self.rope = HalfRoPE(cfg.head_dim, cfg.rope_theta)

        # One direct learned temperature per KV head.
        # Zero initialization follows ZAYA reference code.
        self.key_temperature = nn.Parameter(torch.zeros(cfg.kv_heads))

    def project(self, x: torch.Tensor):
        # B: batch size, T: sequence length
        # H: query heads, K: KV heads, d: head dimension
        B, T, _ = x.shape 

        q = self.q_proj(x).reshape(B, T, self.q_heads, self.head_dim)  # (B, T, H, d)
        k = self.k_proj(x).reshape(B, T, self.kv_heads, self.head_dim)  # (B, T, K, d)
        v = self.v_proj(x).reshape(B, T, self.kv_heads, self.head_dim)  # (B, T, K, d)

        return q, k, v


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape  # (B, T, 128)
        q, k, v = self.project(x)

        # Choose epsilon automatically from Q's floating-point dtype.
        # Used to clamp Q/K norms and prevent division by zero.
        eps = torch.finfo(q.dtype).eps

        # Target vector length after normalization.
        scale = self.head_dim ** 0.5

        q = q * (scale / q.norm(dim=-1, keepdim=True).clamp_min(eps))
        k = k * (scale / k.norm(dim=-1, keepdim=True).clamp_min(eps))

        temperature = self.key_temperature.to(k.dtype)[None, None, :, None]
        k = k * temperature  # (B, T, 2, 16)

        # Move heads before sequence, as HalfRoPE expects.
        q = self.rope(q.transpose(1, 2))  # (B, 4, T, 16)
        k = self.rope(k.transpose(1, 2))  # (B, 2, T, 16)
        v = v.transpose(1, 2)            # (B, 2, T, 16)

        # Share each KV head across its two query heads.
        k = k.repeat_interleave(self.group_size, dim=1)  # (B, 4, T, 16)
        v = v.repeat_interleave(self.group_size, dim=1)  # (B, 4, T, 16)
        # Example [K0, K1] -> [K0, K0, K1, K1]

        y = F.scaled_dot_product_attention(q, k, v,
                                           is_causal=True,
                                           dropout_p=self.dropout_p if self.training else 0.0)  # (B, 4, T, 16)

        y = y.transpose(1, 2).reshape(B, T, self.q_width)  # (B, T, 64)

        return self.out_proj(y)                          # (B, T, 128)


class CCGQAttention(GQAttention):
    """GQA with causal Q/K convolutions, shared means, and value delay.

    Inherits forward() from GQAttention, which calls this class's project().
    Only Q/K/V construction changes; normalization, temperature, RoPE,
    and attention computation are reused.
    """

    def __init__(self, cfg: ModelConfig):
        # super() accesses the parent class, GQAttention.
        # Its __init__(cfg) initializes this same object using our configuration,
        # creating the inherited projections, RoPE, and learned temperatures.
        super().__init__(cfg)
        self.qk_convolution = QKConvolution(cfg)

    def project(self, x: torch.Tensor):
        B, T, _ = x.shape
        H = self.q_heads
        K = self.kv_heads
        d = self.head_dim

        # Original linear projections.
        q = self.q_proj(x)  # (B, T, H*d)
        k = self.k_proj(x)  # (B, T, K*d)
        v = self.v_proj(x)  # (B, T, K*d)

        # Shared means are computed before convolution.
        q_mean, k_mean = shared_qk_means(q.reshape(B, T, H, d),k.reshape(B, T, K, d))  # (B, T, H, d), (B, T, K, d)

        # Convolve Q/K, then add their shared means.
        q, k = self.qk_convolution(q, k)
        q = q.reshape(B, T, H, d) + q_mean
        k = k.reshape(B, T, K, d) + k_mean

        # Delay the second half of the flattened value channels.
        v = delay_values(v).reshape(B, T, K, d)

        return q, k, v


class DynamicCCGQAttention(CCGQAttention):
    """Two-token temporal gates inside CCGQA's first Q/K convolution.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)

        # Two gates per KV group, reading current + previous hidden states.
        # Direct zeros preserve the random initialization of later modules.
        self.gate_weight = nn.Parameter(torch.zeros(2 * cfg.kv_heads, 2 * cfg.d_model))
        self.gate_bias = nn.Parameter(torch.zeros(2 * cfg.kv_heads))

    def project(self, x: torch.Tensor):
        B, T, _ = x.shape
        H, K, d = self.q_heads, self.kv_heads, self.head_dim

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q_mean, k_mean = shared_qk_means(q.reshape(B, T, H, d), k.reshape(B, T, K, d))

        previous = F.pad(x[:, :-1, :], (0, 0, 1, 0))  # (B, T, D)
        features = torch.cat((x, previous), dim=-1)    # (B, T, 2*D)
        logits = F.linear(features, self.gate_weight, self.gate_bias)

        # delta = gate - 1; it starts at zero.
        delta = (2 * torch.sigmoid(logits) - 1).reshape(B, T, K, 2)

        # Match channel order: all query heads, then all key heads.
        delta = torch.cat((delta.repeat_interleave(self.group_size, dim=2), delta), dim=2)
        delta = delta.repeat_interleave(d, dim=2)  # (B, T, (H+K)*d, 2)

        # PyTorch kernel order: previous position, current position.
        past, current = delta.unbind(dim=-1)
        past = F.pad(past.transpose(1, 2), (1, 0))
        current = F.pad(current.transpose(1, 2), (1, 0))

        # Preserve the original padding and convolution biases.
        combined = torch.cat((q, k), dim=-1).transpose(1, 2)
        combined = F.pad(combined, (2, 0))

        conv = self.qk_convolution
        mixed = conv.depthwise(combined)
        weights = conv.depthwise.weight[:, 0, :]  # (channels, 2)

        # Original result + correction = convolution with weights * gates.
        mixed = mixed + weights[None, :, 0, None] * combined[..., :-1] * past
        mixed = mixed + weights[None, :, 1, None] * combined[..., 1:] * current

        combined = conv.headwise(mixed).transpose(1, 2)

        q = combined[..., :self.q_width].reshape(B, T, H, d) + q_mean
        k = combined[..., self.q_width:].reshape(B, T, K, d) + k_mean
        v = delay_values(v).reshape(B, T, K, d)

        return q, k, v



class TransformerBlock(nn.Module):
    """Pre-norm block with selectable attention, SwiGLU, and residual dropout."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()

        self.attention_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        if cfg.attention_type == "gqa":
            self.attention = GQAttention(cfg)
        elif cfg.attention_type == "ccgqa":
            self.attention = CCGQAttention(cfg)
        elif cfg.attention_type == "ccgqa_dynamic":
            self.attention = DynamicCCGQAttention(cfg)
        else:
            raise ValueError(f"Unknown attention_type: {cfg.attention_type!r}")
        self.ffn_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff)

        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention branch, then residual addition.
        x = x + self.dropout(self.attention(self.attention_norm(x)))
        # Feed-forward branch; this call samples a fresh dropout mask.
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x  # (B, T, d_model)

class TransformerLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.layers)])
        self.final_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)

    def forward(self,token_ids: torch.Tensor,targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        # token_ids: (B, T), containing integer vocabulary IDs.
        x = self.token_embedding(token_ids)  # (B, T, d_model)

        for block in self.blocks:
            x = block(x)                    # (B, T, d_model)

        x = self.final_norm(x)              # (B, T, d_model)

        # Reuse the embedding matrix as the output projection.
        logits = F.linear(x, self.token_embedding.weight)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # targets: (B, T), already shifted to the next tokens.
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),  # (B*T, vocab_size)
                targets.reshape(-1),                  # (B*T,)
            )

        return logits, loss