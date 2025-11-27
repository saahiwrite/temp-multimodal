
#!/bin/bash
# Toy debug pipeline:
# 1) extract synthetic features
# 2) train multimodal model

set -e

python src/data/feature_extraction.py \
  --metadata data/toy_chunks_metadata.json \
  --out_dir features \
  --out_metadata data/toy_chunks_with_feats.json \
  --debug

python src/training/train_multimodal.py \
  --metadata data/toy_chunks_with_feats.json \
  --debug \
  --epochs 1 \
  --batch_size 4 \
  --device cpu
