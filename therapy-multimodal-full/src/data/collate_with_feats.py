
import torch

class CollatorWithFeatures:
    """Pad text tokens via tokenizer.pad and features to max length in batch."""
    def __init__(self, tokenizer, max_feat_len=None, pad_value=0.0):
        self.tokenizer = tokenizer
        self.max_feat_len = max_feat_len
        self.pad_value = pad_value

    def __call__(self, batch):
        input_ids = [b["input_ids"] for b in batch]
        attention_mask = [b["attention_mask"] for b in batch]
        padded = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            return_tensors="pt"
        )
        feats = [b["features"] for b in batch]
        lengths = [f.shape[0] for f in feats]
        max_len = max(lengths) if self.max_feat_len is None else min(max(lengths), self.max_feat_len)
        B = len(feats)
        H = feats[0].shape[1]
        padded_feats = torch.full((B, max_len, H), fill_value=self.pad_value, dtype=torch.float32)
        for i, f in enumerate(feats):
            L = min(f.shape[0], max_len)
            padded_feats[i, :L, :] = f[:L, :]
        labels = torch.stack([b["labels"] for b in batch], dim=0)
        ids = [b["id"] for b in batch]
        return {
            "ids": ids,
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
            "features": padded_feats,
            "labels": labels
        }
