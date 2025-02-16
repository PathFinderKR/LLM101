# src/tokenizer.py

import os
import sys
import json
import argparse
from typing import Dict, Optional
import torch
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from src.utils import load_text


class CharTokenizer:
    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        """
        Initialize the character-level tokenizer.

        Args:
            vocab (dict, optional): A pre-defined vocabulary mapping. If None, it will be built from data.
        """

        self.SPECIAL_TOKENS = [
            "<|begin_of_text|>",    # BOS
            "<|eot_id|>",           # This signifies the end of the message in a turn.
            "<|start_header_id|>",  # Start of header
            "<|end_header_id|>",    # End of header
            "<|end_of_text|>",      # EOS
            "<|UNK|>",              # Unknown token
        ]
        self.ROLES = ["system", "user", "assistant"]

        self.tokenizer_type = "char"

        if vocab is not None:
            self.char2idx = vocab
            self.idx2char = {idx: char for char, idx in vocab.items()}
            self.vocab_size = len(vocab)
        else:
            self.char2idx: Dict[str, int] = {}
            self.idx2char: Dict[int, str] = {}
            self.vocab_size: int = 0

    def build_vocab(self, text: str):
        """
        Build vocabulary from the provided text.

        Args:
            text (str): The text data to build the vocabulary from.
        """
        unique_chars = sorted(set(text))
        start_idx = len(self.SPECIAL_TOKENS)

        # Character to index mapping
        self.char2idx = {char: idx for idx, char in enumerate(self.SPECIAL_TOKENS)}
        for idx, char in enumerate(unique_chars, start=start_idx):
            if char not in self.char2idx:
                self.char2idx[char] = idx

        # Index to character mapping
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}

        self.vocab_size = len(self.char2idx)

    def encode(self, text: str) -> torch.Tensor:
        """
        Encode a string into a tensor of integer token IDs.

        Args:
            text (str): The text to encode.

        Returns:
            torch.Tensor: The encoded tensor.
        """
        ids = []
        for char in text:
            if char in self.char2idx:
                ids.append(self.char2idx[char])
            else:
                ids.append(self.char2idx["<|UNK|>"])
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, tokens: torch.Tensor) -> str:
        """
        Decode a tensor of integer token IDs into a string.

        Args:
            tokens (torch.Tensor): The tensor of token IDs.

        Returns:
            str: The decoded string.
        """
        chars = []
        for idx in tokens:
            if idx in self.idx2char:
                chars.append(self.idx2char[idx])
            else:
                chars.append("<|UNK|>")
        return ''.join(chars)

    def save_vocab(self, file_path: str, type: str):
        """
        Save the vocabulary to a JSON file.

        Args:
            file_path (str): The path to save the vocabulary file.
            type (str): The type of the tokenizer.
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.char2idx, f, ensure_ascii=False, indent=4)
        print(f"Vocabulary saved to {type}_vocab.json.")

    def load_vocab(self, file_path: str):
        """
        Load the vocabulary from a JSON file.

        Args:
            file_path (str): The path to the vocabulary file.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            self.char2idx = json.load(f)
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        self.vocab_size = len(self.char2idx)
        print(f"Vocabulary loaded from {file_path}.")


class BPETokenizer:
    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        pass

    def build_vocab(self, text: str):
        pass

    def encode(self, text: str) -> torch.Tensor:
        pass

    def decode(self, tokens: torch.Tensor) -> str:
        pass

    def save_vocab(self, file_path: str):
        pass

    def load_vocab(self, file_path: str):
        pass


def parse_args():
    """
       Parse command-line arguments.

       Returns:
           argparse.Namespace: Command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Build a vocabulary for the tokenizer.")

    default_file_path = "/data/shakespeare.txt"

    parser.add_argument(
        "--type",
        type=str,
        choices=["char", "bpe"],
        default="char",
        help="The type of tokenizer to use. ('char' or 'bpe')"
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default=default_file_path,
        help=f"Path to the text file to build the vocabulary from. Default: {default_file_path}"
    )
    return parser.parse_args()


def main():
    args = parse_args()


    # Root directory
    root_dir = os.path.dirname(os.path.abspath(__file__)) + "/../"


    # Load the text data
    text = load_text(file_path=root_dir+args.file_path)


    # Initialize the tokenizer
    if args.type == "char":
        tokenizer = CharTokenizer()
    elif args.type == "bpe":
        tokenizer = BPETokenizer()
    else:
        raise ValueError("Invalid tokenizer type. Choose 'char' or 'bpe'.")


    # Build and save the vocabulary
    tokenizer.build_vocab(text=text)
    tokenizer.save_vocab(file_path=root_dir+f"{args.type}_vocab.json", type=args.type)


if __name__ == "__main__":
    main()
