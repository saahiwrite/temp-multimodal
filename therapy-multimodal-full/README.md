
# Multimodal Therapist Communication Analysis

This repo contains a **full working code skeleton** for the methodology described in your proposal:

- Text encoder: **DeBERTa-v3-base**
- Audio encoder: **WavLM-base-plus** (pre-extracted features)
- Cross-modal attention (text ↔ audio)
- Multi-label classification with **class-weighted BCE loss**
- Per-class threshold tuning
- Attention heatmap visualization

## Quick Start (toy mode)

1. Install dependencies (inside a virtualenv is recommended):

   ```bash
   pip install -r requirements.txt
   ```

2. Generate toy metadata (already provided as `data/toy_chunks_metadata.json`) and synthetic WavLM-like features:

   ```bash
   python src/data/feature_extraction.py \
       --metadata data/toy_chunks_metadata.json \
       --out_dir features \
       --out_metadata data/toy_chunks_with_feats.json \
       --debug
   ```

3. Train the multimodal model on features:

   ```bash
   python src/training/train_multimodal.py \
       --metadata data/toy_chunks_with_feats.json \
       --debug \
       --epochs 2 \
       --batch_size 8 \
       --device cpu
   ```

4. Examine outputs in `checkpoints/`:

   - `best_model.pt` – best checkpoint
   - `thresholds.json` – per-class tuned thresholds
   - `att_heatmap.png` – example attention heatmap

## Real Data

Replace `data/toy_chunks_metadata.json` with your real `chunks_metadata.json` from the HOPE dataset pipeline, then:

1. Extract WavLM features for each chunk:

   ```bash
   python src/data/feature_extraction.py \
       --metadata data/chunks_metadata.json \
       --out_dir features \
       --out_metadata data/chunks_with_feats.json \
       --use_wavlm
   ```

2. Train:

   ```bash
   python src/training/train_multimodal.py \
       --metadata data/chunks_with_feats.json \
       --epochs 10 \
       --batch_size 4 \
       --device cuda
   ```

Adjust batch size for GPU memory.

## Metadata format

Each entry in `chunks_metadata.json` or `toy_chunks_metadata.json` should look like:

```json
{
  "id": "session_0001_chunk_000",
  "audio_path": "/absolute/or/relative/path/to/chunk.wav",
  "text": "therapist utterance in this 10-second window...",
  "labels": [0, 1, 0, 0, 1, 0]
}
```

After running `feature_extraction.py`, a new field is added:

```json
"feature_path": "features/session_0001_chunk_000.npy"
```

## Important Notes

- The code is written to be **readable and hackable**, not hyper-optimized.
- For real experiments, make sure:
  - You have a GPU and install CUDA-compatible `torch` and `torchaudio`.
  - You install `transformers` and `accelerate` for WavLM/DeBERTa.
- Threshold tuning currently uses a simple F1 grid-search per class over thresholds in `[0.1, 0.9]`.

