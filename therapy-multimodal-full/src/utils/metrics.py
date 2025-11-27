
import numpy as np
from sklearn.metrics import f1_score

def multilabel_metrics(y_true, y_pred, threshold=0.5):
    """Compute micro/macro/weighted F1 for multi-label predictions."""
    y_pred_bin = (y_pred >= threshold).astype(int)
    macro_f1 = f1_score(y_true, y_pred_bin, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred_bin, average="weighted", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred_bin, average="micro", zero_division=0)
    return {
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "micro_f1": float(micro_f1)
    }
