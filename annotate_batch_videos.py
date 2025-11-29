#!/usr/bin/env python3
"""
Batch LLM Annotation for Multiple Videos

Processes ALL videos in a directory at once and combines them into a single
training-ready file.

Usage:
    python3 annotate_batch_videos.py \
        --transcript-dir data/transcripts \
        --audio-base-dir data/chunks \
        --output data/all_chunks_metadata.json \
        --api-key YOUR_CLAUDE_API_KEY

This will:
1. Find all *_therapist.json files in transcript-dir
2. For each video, find corresponding audio chunks
3. Annotate all chunks from all videos
4. Combine into ONE training-ready file
"""

import json
import argparse
import time
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

try:
    import anthropic
except ImportError:
    print("Error: anthropic package not found. Install with: pip install anthropic")
    print("Or for OpenAI: pip install openai")
    exit(1)


# Label order - CRITICAL for binary vector
LABEL_NAMES = ["Neutral", "Reflective", "Empathetic", "Supportive", "Validating", "Transitional"]


ANNOTATION_PROMPT = """You are an expert in psychotherapy and therapeutic communication analysis. 

Analyze the following therapist utterance and identify which communication styles are present. A single utterance can have MULTIPLE styles.

**Therapist Utterance:**
"{text}"

**Communication Styles (Multi-Label):**

1. **Neutral** - Basic conversational statements, administrative talk, or transitional phrases without therapeutic intent
   - Examples: "Hi, how are you?", "Let's get started", "We have 10 minutes left"

2. **Reflective** - Mirroring or paraphrasing the client's words/feelings to demonstrate understanding
   - Examples: "So you're feeling anxious about the meeting", "It sounds like you're saying..."

3. **Empathetic** - Expressing emotional understanding and connection with the client's experience
   - Examples: "That must be really difficult for you", "I can hear how painful this is"

4. **Supportive** - Providing encouragement, reassurance, or positive reinforcement
   - Examples: "You're making great progress", "That took a lot of courage"

5. **Validating** - Acknowledging the legitimacy of the client's feelings or experiences
   - Examples: "Your feelings are completely valid", "Anyone would feel that way"

6. **Transitional** - Guiding the conversation to a new topic or phase of therapy
   - Examples: "Let's explore that further", "Moving on to...", "I want to shift focus to..."

**Important Guidelines:**
- Select ALL applicable styles (most utterances have 1-2, some have 3+)
- Neutral is common but should not exclude other styles
- Be conservative - only select styles that are clearly present
- Consider both explicit content (words) and implicit intent (therapeutic function)

**Response Format:**
Return ONLY a valid JSON array of the applicable style names. No explanation, no markdown, just the JSON array.

Example outputs:
["Neutral"]
["Reflective", "Empathetic"]
["Supportive", "Validating"]
["Neutral", "Transitional"]

Your response:"""


def annotate_chunk_with_claude(text: str, client: anthropic.Anthropic, 
                                model: str = "claude-sonnet-4-20250514") -> List[int]:
    """
    Annotate a single chunk and return BINARY label vector.
    
    Args:
        text: Therapist utterance text
        client: Anthropic client instance
        model: Claude model to use
    
    Returns:
        Binary label vector [0,1,0,1,0,0] format
    """
    try:
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            temperature=0.0,
            messages=[{
                "role": "user",
                "content": ANNOTATION_PROMPT.format(text=text)
            }]
        )
        
        response_text = message.content[0].text.strip()
        
        # Clean up response
        if response_text.startswith("```json"):
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif response_text.startswith("```"):
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        # Parse JSON to get label names
        label_names = json.loads(response_text)
        
        # Convert to binary vector IMMEDIATELY
        binary_labels = [1 if label in label_names else 0 for label in LABEL_NAMES]
        
        return binary_labels
    
    except Exception as e:
        print(f"\nWarning: Error annotating, using default [1,0,0,0,0,0]: {e}")
        return [1, 0, 0, 0, 0, 0]  # Default to Neutral


def find_transcript_files(transcript_dir: str) -> List[Path]:
    """
    Find all therapist transcript JSON files.
    
    Args:
        transcript_dir: Directory containing *_therapist.json files
    
    Returns:
        List of transcript file paths
    """
    transcript_path = Path(transcript_dir)
    
    if not transcript_path.exists():
        print(f"Error: Transcript directory not found: {transcript_dir}")
        return []
    
    # Find all *_therapist.json files
    transcript_files = list(transcript_path.glob("*_therapist.json"))
    
    if not transcript_files:
        print(f"Warning: No *_therapist.json files found in {transcript_dir}")
        # Try without _therapist suffix
        transcript_files = list(transcript_path.glob("*.json"))
    
    return sorted(transcript_files)


def load_transcript_and_create_chunks(transcript_file: Path, audio_base_dir: str) -> List[Dict]:
    """
    Load a single transcript and create chunk entries.
    
    Args:
        transcript_file: Path to VIDEO_ID_therapist.json
        audio_base_dir: Base directory containing video subdirectories with chunks
    
    Returns:
        List of chunk dictionaries
    """
    with open(transcript_file, 'r') as f:
        transcript_data = json.load(f)
    
    # Extract video ID from filename
    # e.g., zTvFhaDP0bM_therapist.json -> zTvFhaDP0bM
    video_id = transcript_file.stem.replace('_therapist', '')
    
    # Get metadata if available
    metadata = transcript_data.get('metadata', {})
    if 'video_id' in metadata:
        video_id = metadata['video_id']
    
    # Audio directory for this video
    audio_dir = Path(audio_base_dir) / video_id
    
    # Find chunks in transcript
    if 'segments' in transcript_data:
        transcript_chunks = transcript_data['segments']
    elif 'chunks' in transcript_data:
        transcript_chunks = transcript_data['chunks']
    elif 'utterances' in transcript_data:
        transcript_chunks = transcript_data['utterances']
    elif 'samples' in transcript_data:
        transcript_chunks = transcript_data['samples']
    else:
        print(f"Warning: No chunks/utterances found in {transcript_file}")
        return []
    
    # Create chunk entries
    chunks = []
    for i, chunk_data in enumerate(transcript_chunks, start=1):
        # Extract text
        text = chunk_data.get('text', '') or chunk_data.get('transcript', '')
        
        if not text:
            continue
        
        chunk = {
            "id": f"{video_id}_chunk_{i:03d}",
            "audio_path": str(audio_dir / f"chunk_therapist_{i:03d}.wav"),
            "text": text.strip(),
            "video_id": video_id,
            "chunk_number": i
        }
        chunks.append(chunk)
    
    return chunks


def process_all_videos(transcript_dir: str, audio_base_dir: str, output_file: str,
                       api_key: str, model: str = "claude-sonnet-4-20250514",
                       save_frequency: int = 50):
    """
    Process all videos and create one combined training file.
    """
    # Find all transcript files
    print(f"Searching for transcript files in: {transcript_dir}")
    transcript_files = find_transcript_files(transcript_dir)
    
    if not transcript_files:
        print("Error: No transcript files found!")
        return
    
    print(f"\nFound {len(transcript_files)} transcript files:")
    for tf in transcript_files:
        print(f"  - {tf.name}")
    
    # Load and prepare all chunks from all videos
    print("\nLoading chunks from all videos...")
    all_chunks = []
    video_stats = {}
    
    for transcript_file in transcript_files:
        chunks = load_transcript_and_create_chunks(transcript_file, audio_base_dir)
        video_id = transcript_file.stem.replace('_therapist', '')
        video_stats[video_id] = len(chunks)
        all_chunks.extend(chunks)
        print(f"  {video_id}: {len(chunks)} chunks")
    
    total_chunks = len(all_chunks)
    print(f"\n{'='*60}")
    print(f"Total: {total_chunks} chunks from {len(transcript_files)} videos")
    print(f"{'='*60}")
    
    if total_chunks == 0:
        print("Error: No chunks to annotate!")
        return
    
    # Estimate cost
    estimated_cost = (total_chunks * 50 * 3) / 1_000_000  # Rough estimate for Claude
    print(f"\nEstimated cost: ~${estimated_cost:.2f}")
    print(f"Estimated time: ~{total_chunks * 1.5 / 60:.0f} minutes")
    
    # Confirm
    response = input("\nProceed with annotation? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Initialize client
    client = anthropic.Anthropic(api_key=api_key)
    
    # Annotate all chunks
    print(f"\n{'='*60}")
    print("Starting batch annotation...")
    print(f"{'='*60}\n")
    
    annotated_count = 0
    errors = []
    
    for i, chunk in enumerate(tqdm(all_chunks, desc="Annotating all videos")):
        try:
            # Get binary labels
            binary_labels = annotate_chunk_with_claude(chunk['text'], client, model)
            chunk['labels'] = binary_labels
            annotated_count += 1
            
            # Save progress periodically
            if (i + 1) % save_frequency == 0:
                save_training_format(all_chunks, output_file)
                print(f"\n✓ Progress saved: {i+1}/{total_chunks} chunks")
            
            # Rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            print(f"\nError on chunk {i} ({chunk['id']}): {e}")
            errors.append((i, chunk['id'], str(e)))
            chunk['labels'] = [1, 0, 0, 0, 0, 0]
    
    # Final save
    save_training_format(all_chunks, output_file)
    
    # Print summary
    print(f"\n{'='*60}")
    print("Batch Annotation Complete!")
    print(f"{'='*60}")
    print(f"Total videos: {len(transcript_files)}")
    print(f"Total chunks: {total_chunks}")
    print(f"Successfully annotated: {annotated_count}")
    print(f"Errors: {len(errors)}")
    
    # Per-video stats
    print(f"\n{'='*60}")
    print("Per-Video Statistics:")
    print(f"{'='*60}")
    for video_id, count in sorted(video_stats.items()):
        print(f"  {video_id}: {count} chunks")
    
    # Label distribution
    print(f"\n{'='*60}")
    print("Overall Label Distribution:")
    print(f"{'='*60}")
    
    label_counts = [0] * 6
    for chunk in all_chunks:
        for idx, val in enumerate(chunk.get('labels', [0]*6)):
            label_counts[idx] += val
    
    for idx, label_name in enumerate(LABEL_NAMES):
        count = label_counts[idx]
        pct = (count / total_chunks) * 100
        print(f"  {label_name:15} {count:5} ({pct:5.1f}%)")
    
    print(f"\n✓ Output saved to: {output_file}")
    print(f"\n✅ ALL {len(transcript_files)} VIDEOS PROCESSED!")
    print("\n📊 Dataset is ready for training - No conversion needed!")
    print(f"\nNext step: Extract WavLM features")
    print(f"  python3 src/data/feature_extraction.py \\")
    print(f"      --metadata {output_file} \\")
    print(f"      --out_dir features \\")
    print(f"      --out_metadata data/chunks_with_feats.json \\")
    print(f"      --use_wavlm --device cpu")


def save_training_format(chunks: List[Dict], output_file: str):
    """Save chunks in training format."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(chunks, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Batch annotate ALL videos at once'
    )
    parser.add_argument('--transcript-dir', required=True,
                       help='Directory containing *_therapist.json files')
    parser.add_argument('--audio-base-dir', required=True,
                       help='Base directory containing video subdirs with chunks')
    parser.add_argument('--output', required=True,
                       help='Output file for combined training data')
    parser.add_argument('--api-key', required=True,
                       help='Anthropic API key')
    parser.add_argument('--model', default='claude-sonnet-4-20250514',
                       help='Claude model')
    parser.add_argument('--save-freq', type=int, default=50,
                       help='Save progress every N chunks')
    
    args = parser.parse_args()
    
    # Validate
    if not Path(args.transcript_dir).exists():
        print(f"Error: Transcript directory not found: {args.transcript_dir}")
        return
    
    if not Path(args.audio_base_dir).exists():
        print(f"Error: Audio base directory not found: {args.audio_base_dir}")
        return
    
    # Process all videos
    process_all_videos(
        args.transcript_dir,
        args.audio_base_dir,
        args.output,
        args.api_key,
        args.model,
        args.save_freq
    )


if __name__ == '__main__':
    main()
