
import torch
import torch.nn as nn

class CrossModalAttentionBlock(nn.Module):
    """Single cross-modal attention block with residual + FFN.

    Given:
      query: (B, Tq, D)
      key_value: (B, Tk, D)
    computes attention of query over key/value, then applies FFN.
    """
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, query, key_value):
        attn_out, attn_weights = self.attn(query, key_value, key_value, need_weights=True)
        x = self.norm1(query + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x, attn_weights
