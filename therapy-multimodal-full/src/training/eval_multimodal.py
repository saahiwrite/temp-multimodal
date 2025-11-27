
"""Evaluate a saved model + thresholds on a metadata_with_feats.json file."""
import argparse, json, os
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset_with_feats import TherapyChunkDataset
from src.data.collate_with_feats import CollatorWithFeatures
from src.models.multimodal_wavlm_deberta import DebertaWavLMClassifier
from src.utils.metrics import multilabel_metrics

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    ap.add_argument("--thresholds", default="checkpoints/thresholds.json")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ds = TherapyChunkDataset(args.metadata, debug=False)
    collator = CollatorWithFeatures(ds.tokenizer)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    device = torch.device(args.device)
    model = DebertaWavLMClassifier()
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    with open(args.thresholds, "r") as f:
        thresholds = json.load(f)

    ys, preds = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        feats = batch["features"].to(device)
        labels = batch["labels"].cpu().numpy()
        logits, _ = model(input_ids, attention_mask, feats)
        probs = torch.sigmoid(logits).cpu().numpy()
        ys.append(labels); preds.append(probs)
    y = np.vstack(ys); p = np.vstack(preds)

    metrics_05 = multilabel_metrics(y, p, threshold=0.5)
    print("Metrics @0.5:", metrics_05)

    from sklearn.metrics import f1_score
    y_pred_bin = np.stack([(p[:, c] >= thresholds[c]).astype(int) for c in range(p.shape[1])], axis=1)
    wf1 = f1_score(y, y_pred_bin, average="weighted", zero_division=0)
    print("Weighted F1 @tuned thresholds:", wf1)

if __name__ == "__main__":
    main()
