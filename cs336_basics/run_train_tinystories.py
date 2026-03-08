"""Train BPE on TinyStories and serialize results."""
import json
import os
import time
import tracemalloc

from train_bpe import train_bpe

INPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "TinyStoriesV2-GPT4-train.txt")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def main():
    tracemalloc.start()
    start = time.time()

    vocab, merges = train_bpe(
        input_path=INPUT_PATH,
        vocab_size=10_000,
        special_tokens=["<|endoftext|>"],
    )

    elapsed = time.time() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Serialize vocab
    vocab_json = {str(k): list(v) for k, v in vocab.items()}
    with open(os.path.join(OUTPUT_DIR, "ts_vocab.json"), "w") as f:
        json.dump(vocab_json, f)

    # Serialize merges
    merges_list = [[list(a), list(b)] for a, b in merges]
    with open(os.path.join(OUTPUT_DIR, "ts_merges.json"), "w") as f:
        json.dump(merges_list, f)

    # Find longest token
    longest_token = max(vocab.values(), key=len)
    longest_id = [k for k, v in vocab.items() if v == longest_token][0]

    print(f"Training time: {elapsed:.1f}s ({elapsed/3600:.2f} hours)")
    print(f"Peak memory: {peak / 1024 / 1024:.1f} MB ({peak / 1024 / 1024 / 1024:.2f} GB)")
    print(f"Vocab size: {len(vocab)}")
    print(f"Merges: {len(merges)}")
    print(f"Longest token (id={longest_id}, {len(longest_token)} bytes): {longest_token!r}")
    try:
        print(f"Longest token decoded: '{longest_token.decode('utf-8')}'")
    except UnicodeDecodeError:
        print("Longest token is not valid UTF-8")

if __name__ == "__main__":
    main()
