# src/generate.py

import os
import sys
import argparse
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import set_seed, configure_device, load_config
from train import initialize_tokenizer
from models.bigram.bigram import Bigram, BigramConfig
from models.mlp.mlp import MLP, MLPConfig
from models.gpt.gpt import GPT, GPTConfig
from models.megabyte.megabyte import MEGABYTE, MegabyteConfig


def parse_args():
    """
    Parse command line arguments

    Returns:
        argparse.Namespace: Command-line arguments.
    """
    default_vocab_path = "char_vocab.json"

    parser = argparse.ArgumentParser(description='Generate text from a trained model.')
    parser.add_argument(
        "--model",
        type=str,
        choices=["bigram", "mlp", "gpt", "megabyte"],
        required=True,
        help="Choose the model architecture."
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        default=default_vocab_path,
        help=f"Path to the vocabulary JSON file. Default: {default_vocab_path}"
    )
    return parser.parse_args()


def main():
    args = parse_args()


    # Root directory
    root_dir = os.path.dirname(os.path.abspath(__file__)) + "/../"


    # Load the configuration
    generate_config = load_config(file_path=root_dir+"configs/generate.yaml")
    model_config = load_config(file_path=root_dir+f"models/{args.model}/config.yaml")


    # Set the seed for reproducibility
    set_seed(seed=generate_config["seed"])


    # Configure the device
    device = configure_device()


    # Initialize the tokenizer
    tokenizer = initialize_tokenizer(file_path=root_dir+args.vocab_path)


    # Initialize the model
    if args.model == "bigram":
        model = Bigram(BigramConfig(
            vocab_size=tokenizer.vocab_size
        ))
    elif args.model == "mlp":
        model = MLP(MLPConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=model_config["context_size"],
            d_embed=model_config["d_embed"],
            d_ff=model_config["d_ff"]
        ))
    elif args.model == "gpt":
        model = GPT(GPTConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=model_config["context_size"],
            n_layer=model_config["n_layer"],
            n_head=model_config["n_head"],
            d_embed=model_config["d_embed"],
            d_ff=model_config["d_ff"],
            dropout=model_config["dropout"]
        ))
    elif args.model == "megabyte":
        context_size = model_config["context_size"]
        pad_id = model_config["pad_id"]
        patch_size = model_config["patch_size"]
        # Global transformer configuration
        global_config = GPTConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=context_size,
            n_layer=model_config["global_n_layer"],
            n_head=model_config["global_n_head"],
            d_embed=model_config["global_d_embed"],
            d_ff=model_config["global_d_ff"],
            dropout=model_config["global_dropout"]
        )
        # Local transformer configuration
        local_config = GPTConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=patch_size,
            n_layer=model_config["local_n_layer"],
            n_head=model_config["local_n_head"],
            d_embed=model_config["local_d_embed"],
            d_ff=model_config["local_d_ff"],
            dropout=model_config["local_dropout"]
        )
        model = MEGABYTE(MegabyteConfig(
            vocab_size=tokenizer.vocab_size,
            pad_id=pad_id,
            patch_size=patch_size,
            patch_num=context_size // patch_size,
            global_config=global_config,
            local_config=local_config
        ))
    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    model.to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model '{args.model}' initialized. Number of Parameters: {num_params}")
    model.load_state_dict(torch.load(root_dir+f"models/{model_config["name"]}/checkpoints/gpt.pt"))


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
