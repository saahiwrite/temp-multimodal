
import torch
import torch.nn as nn
from transformers import AutoModel
from .cross_modal_attention import CrossModalAttentionBlock

class DebertaWavLMClassifier(nn.Module):
    """Multimodal classifier with DeBERTa-v3-base (text) + WavLM features (pre-extracted).

    - Text encoder: DeBERTa-v3-base
    - Audio branch: pre-extracted WavLM hidden states, projected to shared dim
    - Cross-modal attention: text queries audio features
    - Classification: pooled representation -> 6-dim logits (one per communication style)
    """
    def __init__(
        self,
        text_model_name="microsoft/deberta-v3-base",
        wavlm_hidden_dim=768,
        proj_dim=256,
        num_heads=8,
        dropout=0.1,
        freeze_text=False,
    ):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_hidden = self.text_encoder.config.hidden_size

        self.text_proj = nn.Linear(text_hidden, proj_dim)
        self.audio_proj = nn.Linear(wavlm_hidden_dim, proj_dim)

        self.cross_attn = CrossModalAttentionBlock(proj_dim, num_heads=num_heads, dropout=dropout)

        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, 6)  # 6 communication styles
        )

        if freeze_text:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask, features):
        # Text branch
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_hidden = text_out.last_hidden_state  # (B, T_text, H_text)
        text_repr = self.text_proj(text_hidden)   # (B, T_text, D)

        # Audio/features branch: (B, T_feat, H_wavlm)
        audio_repr = self.audio_proj(features)    # (B, T_feat, D)

        # Cross-modal: text queries audio
        fused, attn_weights = self.cross_attn(text_repr, audio_repr)  # (B, T_text, D), (B, heads, T_text, T_feat)

        pooled = fused.mean(dim=1)  # simple average pooling over text tokens
        logits = self.classifier(pooled)  # (B, 6)

        return logits, {"attn_text_audio": attn_weights}
