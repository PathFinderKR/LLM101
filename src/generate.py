# src/generate.py

import os
import sys
import argparse
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import set_seed, configure_device, load_config
from train import init_tokenizer, init_model


def parse_args():
    """
    Parse command line arguments

    Returns:
        argparse.Namespace: Command-line arguments.
    """
    default_vocab_file = "char_vocab.json"

    parser = argparse.ArgumentParser(description='Generate text from a trained model.')
    parser.add_argument(
        "--model",
        type=str,
        choices=["bigram", "mlp", "gpt", "megabyte"],
        required=True,
        help="Choose the model architecture."
    )
    parser.add_argument(
        "--vocab_file",
        type=str,
        default=default_vocab_file,
        help=f"Path to the vocabulary JSON file. Default: {default_vocab_file}"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Root directory
    root_dir = os.path.dirname(os.path.abspath(__file__)) + "/../"

    # Load the configuration
    generate_config = load_config(root_dir + "configs/generate.yaml")
    model_config = load_config(root_dir + f"models/{args.model}/config.yaml")

    # Set the seed for reproducibility
    set_seed(generate_config["seed"])

    # Configure the device
    device = configure_device()

    # Initialize the tokenizer and the model
    tokenizer = init_tokenizer(root_dir + args.vocab_file)
    model = init_model(model_config, tokenizer, device)

    # Load the model weights
    model.load_state_dict(torch.load(root_dir + f"models/{model_config["name"]}/checkpoints/{model_config["name"]}.pt"))

    # Generate text
    print("=== Generated Text ===")
    model.generate(
        tokenizer,
        prompt=generate_config["prompt"],
        max_new_tokens=generate_config["max_new_tokens"],
        device=device,
        temperature=generate_config["temperature"]
    )
    print()


if __name__ == '__main__':
    main()
