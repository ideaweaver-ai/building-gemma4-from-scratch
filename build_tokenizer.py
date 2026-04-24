"""
Build a Custom 8K Vocabulary Tokenizer for TinyStories
======================================================
Trains a SentencePiece BPE tokenizer on the TinyStories dataset with a
vocabulary of 8,000 tokens — optimized for children's story generation.

Why 8K instead of GPT-2's 50K:
  - TinyStories uses simple English (~5-8K unique subwords)
  - Saves ~16M embedding parameters (19.3M → 3.1M with emb_dim=384)
  - Every token gets meaningful gradient updates during training
  - Sequences stay roughly the same length as GPT-2 tokenization

Output files:
  - tinystories_tokenizer.model  — SentencePiece model file
  - tinystories_tokenizer.vocab  — Human-readable vocabulary
  - train.bin                    — Tokenized training data (memory-mapped)
  - validation.bin               — Tokenized validation data (memory-mapped)

Usage:
  python build_tokenizer.py

After running, update your model config:
  "vocab_size": 8000
And use TinyStoriesTokenizer from this file for encoding/decoding.
"""

import os
import numpy as np
from tqdm.auto import tqdm

VOCAB_SIZE = 8000
TOKENIZER_PREFIX = "tinystories_tokenizer"
MODEL_FILE = f"{TOKENIZER_PREFIX}.model"


# ============================================================================
# Step 1: Download TinyStories and export raw text for SentencePiece training
# ============================================================================

def export_text_for_training(output_file="tinystories_raw.txt", max_samples=500_000):
    """
    Export a subset of TinyStories to a plain text file for tokenizer training.
    500K samples is more than enough to learn the vocabulary distribution.
    """
    from datasets import load_dataset

    if os.path.exists(output_file):
        print(f"{output_file} already exists — skipping export.")
        return output_file

    print("Downloading TinyStories dataset...")
    ds = load_dataset("roneneldan/TinyStories", split="train")

    num_samples = min(max_samples, len(ds))
    print(f"Exporting {num_samples:,} samples to {output_file}...")

    with open(output_file, "w", encoding="utf-8") as f:
        for i in tqdm(range(num_samples), desc="exporting text"):
            text = ds[i]["text"].strip()
            if text:
                f.write(text + "\n")

    print(f"Exported {num_samples:,} stories to {output_file}")
    return output_file


# ============================================================================
# Step 2: Train SentencePiece BPE tokenizer
# ============================================================================

def train_tokenizer(input_file="tinystories_raw.txt"):
    """Train a SentencePiece BPE model with 8K vocabulary on TinyStories text."""
    import sentencepiece as spm

    if os.path.exists(MODEL_FILE):
        print(f"{MODEL_FILE} already exists — skipping training.")
        return

    print(f"Training SentencePiece BPE tokenizer (vocab_size={VOCAB_SIZE})...")
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=TOKENIZER_PREFIX,
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        character_coverage=1.0,       # Full coverage for English
        num_threads=os.cpu_count(),
        split_digits=True,            # Treat each digit separately (like GPT)
        byte_fallback=True,           # Handle unknown chars via UTF-8 bytes
        pad_id=3,                     # Reserve pad token
        unk_id=0,
        bos_id=1,
        eos_id=2,
        max_sentence_length=16384,
    )
    print(f"Tokenizer saved: {MODEL_FILE}")


# ============================================================================
# Step 3: Tokenizer wrapper class
# ============================================================================

class TinyStoriesTokenizer:
    """Lightweight wrapper around the trained SentencePiece model."""

    def __init__(self, model_path=MODEL_FILE):
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

    @property
    def vocab_size(self):
        return self.sp.get_piece_size()

    def encode(self, text):
        """Encode text to token IDs (no special tokens added)."""
        return self.sp.encode(text, out_type=int)

    def decode(self, ids):
        """Decode token IDs back to text."""
        return self.sp.decode(ids)

    def encode_batch(self, texts):
        """Encode a list of texts."""
        return [self.sp.encode(t, out_type=int) for t in texts]


# ============================================================================
# Step 4: Tokenize TinyStories dataset into binary files
# ============================================================================

def prepare_data():
    """Tokenize the full TinyStories dataset using the custom tokenizer."""
    from datasets import load_dataset

    if os.path.exists("train.bin") and os.path.exists("validation.bin"):
        print("train.bin and validation.bin already exist — skipping preparation.")
        return

    tokenizer = TinyStoriesTokenizer(MODEL_FILE)
    print(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

    ds = load_dataset("roneneldan/TinyStories")

    def process(example):
        ids = tokenizer.encode(example['text'])
        return {'ids': ids, 'len': len(ids)}

    tokenized = ds.map(
        process,
        remove_columns=['text'],
        desc="tokenizing",
        num_proc=8,
    )

    # Max token value determines dtype
    max_token = max(
        max(tokenized['train']['len']),  # This is length, not token value
        tokenizer.vocab_size
    )
    dtype = np.uint16 if max_token < 2**16 else np.uint32
    print(f"Using dtype={dtype.__name__} (vocab_size={tokenizer.vocab_size})")

    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = f'{split}.bin'
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            batch = dset.shard(
                num_shards=total_batches, index=batch_idx, contiguous=True
            ).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
        print(f"Wrote {filename}: {arr_len:,} tokens")


# ============================================================================
# Step 5: Print tokenizer statistics
# ============================================================================

def print_stats():
    """Show tokenizer stats and sample encodings."""
    tokenizer = TinyStoriesTokenizer(MODEL_FILE)

    print(f"\n{'='*60}")
    print(f"TinyStories Tokenizer Statistics")
    print(f"{'='*60}")
    print(f"Vocabulary size: {tokenizer.vocab_size:,}")
    print(f"Model file:      {MODEL_FILE}")

    # Sample encodings
    samples = [
        "Once upon a time there was a little girl.",
        "The cat sat on the mat and smiled.",
        "Grandmother told a story about a magical unicorn.",
        "He was very happy because he found a shiny red ball.",
    ]

    print(f"\nSample encodings:")
    for text in samples:
        tokens = tokenizer.encode(text)
        print(f"  [{len(tokens):>3} tokens] {text}")

    # Compare with GPT-2
    try:
        import tiktoken
        gpt2 = tiktoken.get_encoding("gpt2")
        print(f"\nComparison with GPT-2 tokenizer:")
        print(f"  {'Text':<55} {'Custom':>7} {'GPT-2':>7}")
        print(f"  {'-'*55} {'-'*7} {'-'*7}")
        for text in samples:
            custom_len = len(tokenizer.encode(text))
            gpt2_len = len(gpt2.encode_ordinary(text))
            print(f"  {text:<55} {custom_len:>7} {gpt2_len:>7}")
    except ImportError:
        pass

    # Embedding parameter savings
    for emb_dim in [384, 512]:
        gpt2_params = 50_257 * emb_dim
        custom_params = VOCAB_SIZE * emb_dim
        saved = gpt2_params - custom_params
        print(f"\n  Embedding savings (emb_dim={emb_dim}):")
        print(f"    GPT-2:  {gpt2_params:>12,} params")
        print(f"    Custom: {custom_params:>12,} params")
        print(f"    Saved:  {saved:>12,} params ({saved/1e6:.1f}M)")


# ============================================================================
# Main: Run the full pipeline
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Building Custom TinyStories Tokenizer")
    print("=" * 60)

    # Step 1: Export text
    text_file = export_text_for_training()

    # Step 2: Train tokenizer
    train_tokenizer(text_file)

    # Step 3: Print stats
    print_stats()

    # Step 4: Tokenize dataset
    print(f"\n{'='*60}")
    print("Tokenizing full dataset...")
    print(f"{'='*60}")
    prepare_data()

    # Cleanup raw text file (large, no longer needed)
    if os.path.exists("tinystories_raw.txt"):
        size_mb = os.path.getsize("tinystories_raw.txt") / 1e6
        os.remove("tinystories_raw.txt")
        print(f"\nCleaned up tinystories_raw.txt ({size_mb:.0f}MB)")

    print(f"\n{'='*60}")
    print("Done! To use in your model:")
    print(f"{'='*60}")
    print(f'  1. Set "vocab_size": {VOCAB_SIZE} in your config')
    print(f'  2. Use TinyStoriesTokenizer("{MODEL_FILE}") instead of tiktoken')
    print(f"  3. train.bin and validation.bin are ready for training")
