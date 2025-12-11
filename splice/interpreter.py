# splice_build_vocab.py
import argparse
import json
import os
from typing import List

import requests

SPLICE_VOCAB_URL = (
    "https://raw.githubusercontent.com/AI4LIFE-GROUP/SpLiCE/main/"
    "resources/laion_concept_vocabulary.json"
)

def download_splice_vocab(save_path: str) -> List[str]:
    """
    Download the LAION-based 1–2 word concept vocabulary used in SpLiCE.
    The file in the official repo stores a list of concept strings.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"Downloading SpLiCE LAION vocabulary from:\n  {SPLICE_VOCAB_URL}")
    resp = requests.get(SPLICE_VOCAB_URL, timeout=60)
    resp.raise_for_status()
    vocab = resp.json()  # expected: ["dog", "coffee cup", ...]
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"Saved vocabulary with {len(vocab)} concepts to {save_path}")
    return vocab

def load_local_vocab(path: str) -> List[str]:
    """
    Load a local JSON list of concepts.
    Expected format: ["dog", "coffee cup", ...]
    """
    with open(path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    print(f"Loaded {len(vocab)} concepts from {path}")
    return vocab

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=str,
        default="data/splice_laion_vocab.json",
        help="Path to save or read the concept vocabulary JSON",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="If set, download official SpLiCE LAION vocabulary file",
    )
    args = parser.parse_args()

    if args.download:
        download_splice_vocab(args.out)
    else:
        # Example: print basic stats for an existing vocab file
        vocab = load_local_vocab(args.out)
        print("Examples:", vocab[:20])

if __name__ == "__main__":
    main()
