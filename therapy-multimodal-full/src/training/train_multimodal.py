
"""Train multimodal DeBERTa + WavLM-feature classifier.

Key features:
- class-weighted BCEWithLogitsLoss using pos_weight per class
- per-class threshold tuning on validation set
- attention heatmap visualization

Usage (toy):

python src/training/train_multimodal.py \
    --metadata data/toy_chunks_with_feats.json \
    --debug \
    --epochs 2 \
    --batch_size 8 \
    --device cpu
"""
import os, json, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
import torch.nn.functional as F
from tqdm.auto import tqdm

from src.data.dataset_with_feats import TherapyChunkDataset
from src.data.collate_with_feats import CollatorWithFeatures
from src.models.multimodal_wavlm_deberta import DebertaWavLMClassifier
from src.utils.metrics import multilabel_metrics

def compute_pos_weight(y):
    """Compute pos_weight per class: (neg+1)/(pos+1)."""
    pos = y.sum(axis=0)
    neg = y.shape[0] - pos
    pos_weight = (neg + 1.0) / (pos + 1.0)
    return torch.tensor(pos_weight, dtype=torch.float32)

def tune_thresholds(y_val, p_val):
    from sklearn.metrics import f1_score
    C = y_val.shape[1]
    thresholds = []
    for c in range(C):
        best_t, best_f1 = 0.5, 0.0
        for t in np.linspace(0.1, 0.9, 17):
            preds = (p_val[:, c] >= t).astype(int)
            f1 = f1_score(y_val[:, c], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds.append(float(best_t))
    return thresholds

def train_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="train"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        feats = batch["features"].to(device)
        labels = batch["labels"].to(device)

        logits, _ = model(input_ids, attention_mask, feats)
        loss = loss_fn(logits, labels)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        total_loss += loss.item() * input_ids.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, preds, atts = [], [], []
    for batch in tqdm(loader, desc="eval"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        feats = batch["features"].to(device)
        labels = batch["labels"].cpu().numpy()

        logits, extra = model(input_ids, attention_mask, feats)
        probs = torch.sigmoid(logits).cpu().numpy()
        ys.append(labels)
        preds.append(probs)
        if extra.get("attn_text_audio") is not None:
            atts.append(extra["attn_text_audio"].cpu().numpy())
    y = np.vstack(ys)
    p = np.vstack(preds)
    return y, p, atts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", type=str, default="data/toy_chunks_with_feats.json")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save_dir", type=str, default="checkpoints")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    ds = TherapyChunkDataset(args.metadata, debug=args.debug)
    collator = CollatorWithFeatures(ds.tokenizer)

    n = len(ds)
    idxs = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(idxs)
    tr = idxs[: int(0.7 * n)]
    va = idxs[int(0.7 * n): int(0.85 * n)]
    te = idxs[int(0.85 * n):]

    train_ds = Subset(ds, tr)
    val_ds = Subset(ds, va)
    test_ds = Subset(ds, te)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    # compute pos_weight from train labels
    all_labels = []
    for b in train_loader:
        all_labels.append(b["labels"].numpy())
    all_labels = np.vstack(all_labels)
    pos_weight = compute_pos_weight(all_labels)

    device = torch.device(args.device)
    model = DebertaWavLMClassifier()
    model.to(device)

    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    opt = AdamW(model.parameters(), lr=args.lr)

    best_wf1 = 0.0
    best_thresholds = [0.5] * 6

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        tr_loss = train_epoch(model, train_loader, opt, loss_fn, device)
        print("Train loss:", tr_loss)

        y_val, p_val, attn_val = evaluate(model, val_loader, device)
        metrics_05 = multilabel_metrics(y_val, p_val, threshold=0.5)
        print("Val metrics @0.5:", metrics_05)
        thresholds = tune_thresholds(y_val, p_val)
        print("Tuned thresholds:", thresholds)

        # Evaluate with tuned thresholds for weighted F1
        from sklearn.metrics import f1_score
        y_pred_bin = np.stack([(p_val[:, c] >= thresholds[c]).astype(int) for c in range(p_val.shape[1])], axis=1)
        wf1 = f1_score(y_val, y_pred_bin, average="weighted", zero_division=0)
        print("Val weighted F1 @tuned:", wf1)

        if wf1 > best_wf1:
            best_wf1 = wf1
            best_thresholds = thresholds
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pt"))
            with open(os.path.join(args.save_dir, "thresholds.json"), "w") as f:
                json.dump(best_thresholds, f, indent=2)
            print("Saved new best model + thresholds.")

        # Save example attention heatmap from first validation batch
        if len(attn_val) > 0:
            try:
                import matplotlib.pyplot as plt
                att = attn_val[0]  # (B, heads, T_text, T_feat)
                arr = att.mean(axis=(0,1))  # average over batch & heads -> (T_text, T_feat)
                plt.figure(figsize=(6,4))
                plt.imshow(arr, aspect="auto", origin="lower")
                plt.colorbar()
                plt.title("Text→Audio Attention (validation example)")
                outp = os.path.join(args.save_dir, "att_heatmap.png")
                plt.savefig(outp, bbox_inches="tight", dpi=150)
                plt.close()
                print("Saved attention heatmap to", outp)
            except Exception as e:
                print("Could not save attention heatmap:", e)

    # Final test evaluation with best thresholds
    y_test, p_test, _ = evaluate(model, test_loader, device)
    metrics_test_05 = multilabel_metrics(y_test, p_test, threshold=0.5)
    print("Test metrics @0.5:", metrics_test_05)
    y_pred_bin = np.stack([(p_test[:, c] >= best_thresholds[c]).astype(int) for c in range(p_test.shape[1])], axis=1)
    from sklearn.metrics import f1_score
    wf1_test = f1_score(y_test, y_pred_bin, average="weighted", zero_division=0)
    print("Test weighted F1 @tuned:", wf1_test)

if __name__ == "__main__":
    main()
