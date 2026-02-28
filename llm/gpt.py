# ============================================================
# gpt.py
# GPT Model Definition
# ============================================================

import torch
import torch.nn as nn
from .transformer_block import TransformerBlock, LayerNorm


# ============================================================
# GPT Model
# ============================================================

class GPTModel(nn.Module):
    """
    GPT-style Transformer model.

    Architecture:
        Token Embedding
        + Positional Embedding
        → Dropout
        → N Transformer Blocks
        → Final LayerNorm
        → Linear Output Head (vocab projection)

    Args:
        cfg (dict): Configuration dictionary containing:
            - vocab_size
            - context_length
            - emb_dim
            - n_layers
            - n_heads
            - drop_rate
            - qkv_bias
    """

    def __init__(self, cfg):
        super().__init__()

        # --------------------------------------------------------
        # Token Embedding Layer
        # Maps token IDs → embedding vectors
        # --------------------------------------------------------
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])

        # --------------------------------------------------------
        # Positional Embedding Layer
        # Adds position information to token embeddings
        # --------------------------------------------------------
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])

        # --------------------------------------------------------
        # Embedding Dropout
        # Applied after summing token + positional embeddings
        # --------------------------------------------------------
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # --------------------------------------------------------
        # Transformer Blocks (stacked)
        # --------------------------------------------------------
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # --------------------------------------------------------
        # Final Layer Normalization
        # --------------------------------------------------------
        self.final_norm = LayerNorm(cfg["emb_dim"])

        # --------------------------------------------------------
        # Output Projection Head
        # Maps embedding dimension → vocabulary size
        # Used for next-token prediction or classification head
        # --------------------------------------------------------
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    # ============================================================
    # Forward Pass
    # ============================================================

    def forward(self, in_idx):
        """
        Forward pass of GPT model.

        Args:
            in_idx (Tensor):
                Shape: (batch_size, sequence_length)
                Contains token IDs.

        Returns:
            logits (Tensor):
                Shape: (batch_size, sequence_length, vocab_size)
                Raw prediction scores for each token position.
        """

        # Get batch size and sequence length
        batch_size, seq_len = in_idx.shape

        # --------------------------------------------------------
        # Token Embeddings
        # --------------------------------------------------------
        tok_embeds = self.tok_emb(in_idx)

        # --------------------------------------------------------
        # Positional Embeddings
        # Create position indices [0, 1, 2, ..., seq_len-1]
        # --------------------------------------------------------
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )

        # --------------------------------------------------------
        # Combine token + position embeddings
        # --------------------------------------------------------
        x = tok_embeds + pos_embeds

        # Apply embedding dropout
        x = self.drop_emb(x)

        # Pass through transformer stack
        x = self.trf_blocks(x)

        # Apply final layer normalization
        x = self.final_norm(x)

        # Project to vocabulary logits
        logits = self.out_head(x)

        return logits