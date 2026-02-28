# ============================================================
# transformer_block.py
# Core Transformer Components (Attention, FFN, LayerNorm)
# ============================================================

import torch
import torch.nn as nn


# ============================================================
# Multi-Head Self-Attention
# ============================================================

class MultiHeadAttention(nn.Module):
    """
    Implements masked multi-head self-attention.

    Features:
        - Query, Key, Value projections
        - Causal masking (prevents attending to future tokens)
        - Multi-head parallel attention
        - Output projection
    """

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()

        # Ensure embedding dimension is divisible by number of heads
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        # Linear projections for Q, K, V
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Final projection after concatenating heads
        self.out_proj = nn.Linear(d_out, d_out)

        # Dropout applied to attention weights
        self.dropout = nn.Dropout(dropout)

        # Causal mask (upper triangular matrix)
        # Prevents attention to future tokens
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, num_tokens, embedding_dim)

        Returns:
            context_vec: (batch_size, num_tokens, embedding_dim)
        """

        b, num_tokens, d_in = x.shape

        # Project inputs to Q, K, V
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Reshape for multi-head attention
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose to (batch, heads, tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute attention scores
        attn_scores = queries @ keys.transpose(2, 3)

        # Apply causal mask
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Scale and normalize
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5,
            dim=-1
        )

        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Restore original embedding dimension
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        # Final linear projection
        context_vec = self.out_proj(context_vec)

        return context_vec


# ============================================================
# Custom Layer Normalization
# ============================================================

class LayerNorm(nn.Module):
    """
    Custom implementation of Layer Normalization.

    Normalizes across embedding dimension.
    """

    def __init__(self, emb_dim):
        super().__init__()

        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)

        return self.scale * norm_x + self.shift


# ============================================================
# GELU Activation
# ============================================================

class GELU(nn.Module):
    """
    Gaussian Error Linear Unit activation function.
    Used in GPT feedforward layers.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (
            1 + torch.tanh(
                torch.sqrt(torch.tensor(2.0 / torch.pi)) *
                (x + 0.044715 * torch.pow(x, 3))
            )
        )


# ============================================================
# Feedforward Network (MLP)
# ============================================================

class FeedForward(nn.Module):
    """
    Two-layer feedforward network:
        Linear → GELU → Linear
    """

    def __init__(self, cfg):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


# ============================================================
# Transformer Block
# ============================================================

class TransformerBlock(nn.Module):
    """
    A single GPT Transformer block:

        LayerNorm
        → Multi-Head Attention
        → Residual Connection
        → LayerNorm
        → Feedforward
        → Residual Connection
    """

    def __init__(self, cfg):
        super().__init__()

        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )

        self.ff = FeedForward(cfg)

        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])

        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):

        # ---- Attention Block ----
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Residual connection

        # ---- Feedforward Block ----
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Residual connection

        return x