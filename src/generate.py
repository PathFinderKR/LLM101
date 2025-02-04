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
    default_generate_config = "configs/generate.yaml"
    default_model_config = "models/megabyte/config.yaml"
    default_vocab_file = "char_vocab.json"

    parser = argparse.ArgumentParser(description='Generate text from a trained model.')
    parser.add_argument(
        "--generate_config",
        type=str,
        default=default_generate_config,
        help=f"Path to the generate config file. Default: {default_generate_config}"
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default=default_model_config,
        help=f"Path to the model configuration YAML file. Default: {default_model_config}"
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
    generate_config = load_config(root_dir + args.generate_config)
    model_config = load_config(root_dir + args.model_config)

    # Set the seed for reproducibility
    set_seed(generate_config["seed"])

    # Configure the device
    device = configure_device()

    # Initialize the tokenizer and the model
    tokenizer = init_tokenizer(root_dir + args.vocab_file)
    model = init_model(model_config, tokenizer, device)

    # Load the model weights
    model.load_state_dict(torch.load(root_dir + model_config["model_path"]))

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
