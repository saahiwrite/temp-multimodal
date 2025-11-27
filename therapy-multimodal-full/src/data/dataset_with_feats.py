
import json, os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

class TherapyChunkDataset(Dataset):
    """Dataset for pre-extracted WavLM features + DeBERTa text.

    Expects metadata JSON with entries:
    {
      "id": ...,
      "text": ...,
      "labels": [6-dim],
      "feature_path": "path/to/features.npy"
    }
    """
    def __init__(self, metadata_path, tokenizer_name="microsoft/deberta-v3-base", debug=False):
        self.metadata_path = Path(metadata_path)
        with open(self.metadata_path, "r") as f:
            self.meta = json.load(f)
        self.debug = debug
        if debug:
            self.meta = self.meta[:200]

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        item = self.meta[idx]
        text = item.get("text", "")
        tok = self.tokenizer(
            text,
            truncation=True,
            padding=False,
            return_tensors="pt"
        )
        input_ids = tok["input_ids"].squeeze(0)
        attention_mask = tok["attention_mask"].squeeze(0)

        feat_path = item.get("feature_path")
        if feat_path is None or not os.path.exists(feat_path):
            # fallback: random features
            feats = np.random.randn(80, 768).astype("float32")
        else:
            feats = np.load(feat_path).astype("float32")
        feats = torch.from_numpy(feats)  # (T, H)

        labels = torch.tensor(item.get("labels", [0]*6), dtype=torch.float32)
        return {
            "id": item.get("id", str(idx)),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "features": feats,
            "labels": labels
        }
