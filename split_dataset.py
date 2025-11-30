#!/usr/bin/env python3
"""
Split dataset into train/val/test by sessions (70/15/15)

This ensures no data leakage by keeping all chunks from the same session together.

Usage:
    python3 split_dataset.py \
        --input data/chunks_with_feats.json \
        --output_dir data/splits \
        --train_ratio 0.70 \
        --val_ratio 0.15 \
        --test_ratio 0.15 \
        --seed 42
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import random
import numpy as np


def split_by_sessions(chunks, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split chunks by sessions to avoid data leakage.
    
    Args:
        chunks: List of chunk dictionaries
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        seed: Random seed for reproducibility
    
    Returns:
        train_chunks, val_chunks, test_chunks
    """
    # Set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Group chunks by video_id (session)
    sessions = defaultdict(list)
    for chunk in chunks:
        # Extract session ID from chunk ID or video_id field
        if 'video_id' in chunk:
            session_id = chunk['video_id']
        else:
            # Extract from ID (e.g., "video_001_chunk_005" -> "video_001")
            chunk_id = chunk.get('id', '')
            session_id = '_'.join(chunk_id.split('_')[:-2]) if '_chunk_' in chunk_id else chunk_id
        
        sessions[session_id].append(chunk)
    
    # Get list of unique sessions
    session_ids = list(sessions.keys())
    num_sessions = len(session_ids)
    
    print(f"\nDataset Statistics:")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Total sessions: {num_sessions}")
    print(f"  Chunks per session: {len(chunks) / num_sessions:.1f} avg")
    
    # Shuffle sessions
    random.shuffle(session_ids)
    
    # Calculate split points
    train_end = int(num_sessions * train_ratio)
    val_end = train_end + int(num_sessions * val_ratio)
    
    # Split sessions
    train_sessions = session_ids[:train_end]
    val_sessions = session_ids[train_end:val_end]
    test_sessions = session_ids[val_end:]
    
    # Collect chunks for each split
    train_chunks = []
    val_chunks = []
    test_chunks = []
    
    for session_id in train_sessions:
        train_chunks.extend(sessions[session_id])
    
    for session_id in val_sessions:
        val_chunks.extend(sessions[session_id])
    
    for session_id in test_sessions:
        test_chunks.extend(sessions[session_id])
    
    # Print split statistics
    print(f"\nSplit Statistics:")
    print(f"  Train: {len(train_sessions)} sessions, {len(train_chunks)} chunks ({len(train_chunks)/len(chunks)*100:.1f}%)")
    print(f"  Val:   {len(val_sessions)} sessions, {len(val_chunks)} chunks ({len(val_chunks)/len(chunks)*100:.1f}%)")
    print(f"  Test:  {len(test_sessions)} sessions, {len(test_chunks)} chunks ({len(test_chunks)/len(chunks)*100:.1f}%)")
    
    # Check label distribution in each split
    print(f"\nLabel Distribution:")
    label_names = ["Neutral", "Reflective", "Empathetic", "Supportive", "Validating", "Transitional"]
    
    for split_name, split_chunks in [("Train", train_chunks), ("Val", val_chunks), ("Test", test_chunks)]:
        label_counts = [0] * 6
        for chunk in split_chunks:
            labels = chunk.get('labels', [0]*6)
            for i, val in enumerate(labels):
                label_counts[i] += val
        
        print(f"\n  {split_name}:")
        for i, name in enumerate(label_names):
            pct = (label_counts[i] / len(split_chunks)) * 100
            print(f"    {name:15} {label_counts[i]:5} ({pct:5.1f}%)")
    
    return train_chunks, val_chunks, test_chunks


def main():
    parser = argparse.ArgumentParser(description='Split dataset into train/val/test by sessions')
    parser.add_argument('--input', required=True, help='Input JSON file with all chunks')
    parser.add_argument('--output_dir', default='data/splits', help='Output directory for splits')
    parser.add_argument('--train_ratio', type=float, default=0.70, help='Training ratio (default: 0.70)')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Validation ratio (default: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Test ratio (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        print(f"Error: Ratios must sum to 1.0 (got {total_ratio})")
        return
    
    print("="*70)
    print("Dataset Splitting by Sessions")
    print("="*70)
    
    # Load data
    print(f"\nLoading data from: {args.input}")
    with open(args.input, 'r') as f:
        chunks = json.load(f)
    
    # Split data
    train_chunks, val_chunks, test_chunks = split_by_sessions(
        chunks,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save splits
    train_file = output_dir / 'train.json'
    val_file = output_dir / 'val.json'
    test_file = output_dir / 'test.json'
    
    print(f"\nSaving splits to: {output_dir}")
    
    with open(train_file, 'w') as f:
        json.dump(train_chunks, f, indent=2)
    print(f"  ✓ Train: {train_file}")
    
    with open(val_file, 'w') as f:
        json.dump(val_chunks, f, indent=2)
    print(f"  ✓ Val:   {val_file}")
    
    with open(test_file, 'w') as f:
        json.dump(test_chunks, f, indent=2)
    print(f"  ✓ Test:  {test_file}")
    
    # Save split info
    split_info = {
        'seed': args.seed,
        'train_ratio': args.train_ratio,
        'val_ratio': args.val_ratio,
        'test_ratio': args.test_ratio,
        'train_chunks': len(train_chunks),
        'val_chunks': len(val_chunks),
        'test_chunks': len(test_chunks),
        'total_chunks': len(chunks)
    }
    
    info_file = output_dir / 'split_info.json'
    with open(info_file, 'w') as f:
        json.dump(split_info, f, indent=2)
    print(f"  ✓ Info:  {info_file}")
    
    print("\n" + "="*70)
    print("✅ Dataset split complete!")
    print("="*70)
    print(f"\nNext steps:")
    print(f"  1. Train on: {train_file}")
    print(f"  2. Validate on: {val_file}")
    print(f"  3. Test on: {test_file}")
    print(f"\nExample training command:")
    print(f"  python3 src/training/train_multimodal.py \\")
    print(f"      --metadata {train_file} \\")
    print(f"      --val_metadata {val_file} \\")
    print(f"      --epochs 10 --batch_size 4 --device cpu")


if __name__ == '__main__':
    main()
