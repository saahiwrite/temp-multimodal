
"""Pre-extract WavLM features (or synthetic ones for debug) for each chunk.

This script reads a metadata JSON, iterates over chunks, and writes a new JSON with an
added "feature_path" key pointing to a .npy file containing per-frame features.

Two modes:
- --use_wavlm: use microsoft/wavlm-base-plus via HuggingFace (requires transformers, torchaudio, GPU recommended).
- --debug: ignore audio files and generate random features (fast, good for testing code paths).

Usage (toy debug):

python src/data/feature_extraction.py \
    --metadata data/toy_chunks_metadata.json \
    --out_dir features \
    --out_metadata data/toy_chunks_with_feats.json \
    --debug

Usage (real WavLM features):

python src/data/feature_extraction.py \
    --metadata data/chunks_metadata.json \
    --out_dir features \
    --out_metadata data/chunks_with_feats.json \
    --use_wavlm
"""
import os, json, argparse
from pathlib import Path

import numpy as np

def extract_wavlm_features(audio_path, model, processor, sample_rate=16000, device="cpu"):
    import torchaudio, torch
    wav, sr = torchaudio.load(str(audio_path))
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.to(device)
    inputs = processor(wav.squeeze(0), sampling_rate=sample_rate, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    feats = out.last_hidden_state.squeeze(0).cpu().numpy().astype("float32")  # (T, H)
    return feats

def main(metadata_path, out_dir="features", out_metadata_path=None, use_wavlm=False, debug=False, device="cpu"):
    p = Path(metadata_path)
    with open(p, "r") as f:
        meta = json.load(f)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    new_meta = []

    if use_wavlm and not debug:
        from transformers import AutoModel, AutoProcessor
        import torch
        wavlm_name = "microsoft/wavlm-base-plus"
        processor = AutoProcessor.from_pretrained(wavlm_name)
        model = AutoModel.from_pretrained(wavlm_name).to(device)
        model.eval()
    else:
        processor = None
        model = None

    for item in meta:
        cid = item.get("id")
        if debug or not use_wavlm:
            T = np.random.randint(40, 120)
            H = 768
            feats = (np.random.randn(T, H) * 0.5).astype("float32")
        else:
            audio_path = item.get("audio_path")
            if not audio_path or not Path(audio_path).exists():
                print(f"Skipping {cid}: missing audio_path")
                continue
            feats = extract_wavlm_features(audio_path, model, processor, device=device)

        feat_path = out_dir / f"{cid}.npy"
        np.save(str(feat_path), feats)
        item["feature_path"] = str(feat_path)
        new_meta.append(item)

    if out_metadata_path is None:
        out_metadata_path = p.parent / (p.stem + "_with_feats.json")
    with open(out_metadata_path, "w") as f:
        json.dump(new_meta, f, indent=2)
    print(f"Wrote {len(new_meta)} entries to {out_metadata_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out_dir", default="features")
    ap.add_argument("--out_metadata", default=None)
    ap.add_argument("--use_wavlm", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    main(args.metadata, out_dir=args.out_dir, out_metadata_path=args.out_metadata,
         use_wavlm=args.use_wavlm, debug=args.debug, device=args.device)
