# src/scaling_laws.py

import os
import sys
import json
import argparse
import math
from typing import Tuple
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_
#from datasets import load_dataset
import wandb
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import set_seed, configure_device, load_text, load_config, save_checkpoint, load_checkpoint
from src.tokenizer import CharTokenizer, BPETokenizer
from src.train import initialize_tokenizer, setup_optimizer, setup_scheduler, split_text, TextDataset, init_dataloader, train_epoch, evaluate
from models.bigram.bigram import Bigram, BigramConfig
from models.mlp.mlp import MLP, MLPConfig
from models.gpt.gpt import GPT, GPTConfig
from models.megabyte.megabyte import MEGABYTE, MegabyteConfig


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Command-line arguments.
    """
    default_vocab_file = "char_vocab.json"

    parser = argparse.ArgumentParser(description="Train a language model.")
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
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to the checkpoint file to resume training."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode."
    )
    return parser.parse_args()







def compute_experiment(
        model_arch: str,
        context_size: int, d_embed: int, d_ff: int, n_layer: int, n_head: int, dropout: float,
        model_sizes: list[str], dataset_sizes: list[float],
        train_text: str, val_text: str, tokenizer: CharTokenizer | BPETokenizer,
        batch_size: int, optimizer: Optimizer, scheduler: LambdaLR, current_epoch: int, total_epochs: int, grad_clip: float, device: torch.device, wandb_run: wandb.sdk.wandb_run.Run):
    """
    Compute the experiment for the scaling laws.

    Args:
        model_arch (str): Model architecture.
        context_size (int): Context size of the model.
        d_embed (int): Embedding dimension.
        d_ff (int): Feed-forward dimension.
        n_layer (int): Number of layers.
        n_head (int): Number of attention heads.
        dropout (float): Dropout probability.
        model_sizes (list[str]): List of model sizes.
        dataset_sizes (list[float]): List of dataset sizes.
        train_text (str): Text data for training.
        val_text (str): Text data for validation.
        tokenizer (CharTokenizer | BPETokenizer): Tokenizer instance.
        batch_size (int): Batch size the DataLoaders.
        optimizer (Optimizer): Optimizer for training.
        scheduler (LambdaLR): Learning rate scheduler.
        current_epoch (int): Current epoch.
        total_epochs (int): Total number of epochs.
        grad_clip (float): Gradient clipping value.
        device (torch.device): Device for training.
        wandb_run (wandb.sdk.wandb_run.Run): Wandb run for logging.
    """
    for model_size in model_sizes:
        for dataset_size in dataset_sizes:
            # Subset the training data
            subset_train_text = train_text[:int(len(train_text) * dataset_size)]

            train_dataset = TextDataset(subset_train_text, tokenizer, context_size)
            val_dataset = TextDataset(val_text, tokenizer, context_size)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

            print(f"Training data: {len(train_dataset)} samples, Validation data: {len(val_dataset)} samples")

            # Initialize the model
            model = init_model(model_arch, context_size, d_embed, d_ff, n_layer, n_head, dropout, tokenizer, device)


def dataset_size_experiment():
    pass



def model_size_experiment():
    pass




def main():
    args = parse_args()


    # Root directory
    root_dir = os.path.dirname(os.path.abspath(__file__)) + "/../"


    # Load the configuration
    scaling_config = load_config(file_path=root_dir+"config/scaling_laws.yaml")


    # Set the seed for reproducibility
    set_seed(seed=scaling_config["seed"])


    # Configure the device
    device = configure_device()


    # Initialize wandb
    if not args.debug:
        wandb.login(key=os.environ.get("WANDB_API_KEY"))
        wandb_run = wandb.init(
            project=scaling_config["project"],
            config={
                "scaling_config": scaling_config,
                "resume": args.resume
            },
            resume=args.resume is not None,
            dir=root_dir
        )
        print(f"Wandb run initialized: {wandb_run.id}")
    else:
        wandb_run = None
        print("Debug mode enabled")


    # Initialize the tokenizer
    tokenizer = initialize_tokenizer(file_path=root_dir+args.vocab_path)


    # Load and the text data
    text = load_text(file_path=root_dir+"data/shakespeare.txt")
    train_text, val_text = split_text(text=text, val_size=scaling_config["val_size"])


    # Run the experiment
    try:
        compute_experiment()

    except KeyboardInterrupt:
        print("Training interrupted by the user")
    except Exception as e:
        print(f"Error during training: {e}")


    # Finish the wandb run
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
