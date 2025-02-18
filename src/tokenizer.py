# src/tokenizer.py

import os
import sys
import json
import argparse
from typing import Dict, Optional, List
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

    def save_vocab(self, file_path: str):
        """
        Save the vocabulary to a JSON file.

        Args:
            file_path (str): The path to save the vocabulary file.
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.char2idx, f, ensure_ascii=False, indent=4)
        print(f"Vocabulary saved to {file_path}.")

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


class ContextCharTokenizer:
    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        """
        Initialize the context-aware character-level tokenizer.

        Args:
            vocab (dict, optional): A pre-defined vocabulary mapping token->id.
                                    If None, the vocab will be built from data.
        """

        # Special tokens (as in your original code)
        self.SPECIAL_TOKENS = [
            "<|begin_of_text|>",    # BOS
            "<|eot_id|>",           # This signifies the end of the message in a turn.
            "<|start_header_id|>",  # Start of header
            "<|end_header_id|>",    # End of header
            "<|end_of_text|>",      # EOS
            "<|UNK|>",              # Unknown token
        ]

        self.tokenizer_type = "context_aware_char"

        if vocab is not None:
            self.token2idx = vocab
            self.idx2token = {idx: token for token, idx in vocab.items()}
            self.vocab_size = len(vocab)
        else:
            self.token2idx: Dict[str, int] = {}
            self.idx2token: Dict[int, str] = {}
            self.vocab_size: int = 0

    def _tokenize_with_context(self, text: str) -> List[str]:
        """
        Split text into context-aware character tokens.

        We:
          1. Split text into words using a simple regex or split on whitespace.
          2. For each word, mark each character with B, M, E (begin/middle/end)
             or S (if it's a single-character word).
          3. Return the list of tokens.

        Args:
            text (str): The input text.

        Returns:
            List[str]: A list of context-aware character tokens.
        """
        # Simple way to split on whitespace and keep punctuation separate if desired.
        # You might adapt this to your use case.
        # Example: re.findall(r"\w+|[^\w\s]+", text) would split out punctuation too.
        words = text.split()

        tokens = []
        for word in words:
            # You could also further handle punctuation as separate "words."
            # For this simple approach, treat them as part of the "word."
            if len(word) == 1:
                # single-character word: c_S
                tokens.append(f"{word}_S")
            else:
                # multi-character word
                # first character: c_B
                tokens.append(f"{word[0]}_B")
                # middle characters: c_M
                for c in word[1:-1]:
                    tokens.append(f"{c}_M")
                # last character: c_E
                tokens.append(f"{word[-1]}_E")

            # Optionally, you might want to treat whitespace as a token
            # or separate token to keep spacing info. That is optional.

        return tokens

    def build_vocab(self, text: str):
        """
        Build or rebuild the tokenizer vocabulary from raw text.

        Args:
            text (str): The text data to build the vocabulary from.
        """
        # 1. Create context-aware char tokens
        tokens = self._tokenize_with_context(text)

        # 2. Collect unique tokens
        unique_tokens = sorted(set(tokens))

        # 3. Initialize token->id with special tokens
        self.token2idx = {tok: i for i, tok in enumerate(self.SPECIAL_TOKENS)}

        # 4. Add context-aware tokens to the vocabulary
        start_idx = len(self.token2idx)
        for i, token in enumerate(unique_tokens, start=start_idx):
            if token not in self.token2idx:
                self.token2idx[token] = i

        # 5. Build idx->token mapping
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}

        # 6. Set vocab size
        self.vocab_size = len(self.token2idx)

    def encode(self, text: str) -> torch.Tensor:
        """
        Encode a string into a tensor of integer token IDs.

        Args:
            text (str): The text to encode.

        Returns:
            torch.Tensor: The encoded tensor of token IDs.
        """
        # 1. Convert text to list of context-aware tokens
        tokens = self._tokenize_with_context(text)

        # 2. Map each token to its ID or <|UNK|>
        unk_id = self.token2idx["<|UNK|>"]
        token_ids = [self.token2idx.get(token, unk_id) for token in tokens]

        return torch.tensor(token_ids, dtype=torch.long)

    def decode(self, tokens: torch.Tensor) -> str:
        """
        Decode a tensor of integer token IDs back into text.

        Because we have context tags (B, M, E, S) appended to characters,
        we need to remove those tags and reconstruct the words.

        Args:
            tokens (torch.Tensor): The tensor of token IDs.

        Returns:
            str: The reconstructed text (best-effort).
        """
        decoded_text = []
        current_word = []

        for idx in tokens.tolist():
            if idx in self.idx2token:
                token = self.idx2token[idx]
            else:
                # If unknown index, skip or mark it
                decoded_text.append("<|UNK|>")
                continue

            # Check if it's one of the special tokens
            if token in self.SPECIAL_TOKENS:
                # Could handle these however you like
                # For simplicity, we’ll just skip them in final text reconstruction
                continue

            # token looks like "a_B", "p_M", etc.
            # Let's split by underscore
            if "_" in token:
                char, tag = token.rsplit("_", 1)
                if tag == "S":
                    # Single-character word
                    # If there's a word in progress, let's close it
                    if current_word:
                        # Join current word, add to text
                        decoded_text.append("".join(current_word))
                        current_word = []
                    # Now add this single-character word as its own piece
                    decoded_text.append(char)
                elif tag == "B":
                    # Beginning of a word
                    # If there's a word in progress, close it first
                    if current_word:
                        decoded_text.append("".join(current_word))
                        current_word = []
                    current_word.append(char)
                elif tag == "M":
                    # Middle of a word
                    current_word.append(char)
                elif tag == "E":
                    # End of a word
                    current_word.append(char)
                    # Now we finalize this word
                    decoded_text.append("".join(current_word))
                    current_word = []
                else:
                    # fallback if somehow the tag isn't recognized
                    current_word.append(token)
            else:
                # fallback if token doesn't have the underscore form
                current_word.append(token)

        # If something is left in current_word at the end, add it
        if current_word:
            decoded_text.append("".join(current_word))

        # Finally, join the decoded words with spaces
        # This is a best-effort. If you need exact spacing or punctuation,
        # you'd have to encode those tokens more explicitly.
        return " ".join(decoded_text)

    def save_vocab(self, file_path: str):
        """
        Save the vocabulary to a JSON file.

        Args:
            file_path (str): The path to save the vocabulary file.
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.token2idx, f, ensure_ascii=False, indent=4)
        print(f"Vocabulary saved to {file_path}.")

    def load_vocab(self, file_path: str):
        """
        Load the vocabulary from a JSON file.

        Args:
            file_path (str): The path to the vocabulary file.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            self.token2idx = json.load(f)
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}
        self.vocab_size = len(self.token2idx)
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
        choices=["char", "char-context", "bpe"],
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
    elif args.type == "char-context":
        tokenizer = ContextCharTokenizer()
    elif args.type == "bpe":
        tokenizer = BPETokenizer()
    else:
        raise ValueError("Invalid tokenizer type. Choose 'char' or 'bpe'.")


    # Build and save the vocabulary
    tokenizer.build_vocab(text=text)
    tokenizer.save_vocab(file_path=root_dir+f"{args.type}_vocab.json")


if __name__ == "__main__":
    main()
