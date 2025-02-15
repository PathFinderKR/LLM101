# src/train.py

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


def init_wandb(args: argparse.Namespace, train_config: dict, model_config: dict, wandb_path: str) -> wandb.sdk.wandb_run.Run:
    """
    Initialize wandb.

    Args:
        args (argparse.Namespace): Command-line arguments.
        train_config (dict): Training configuration.
        model_config (dict): Model configuration.
        wandb_path (str): Path to the wandb directory.

    Returns:
        wandb.sdk.wandb_run.Run: Wandb run for logging.
    """
    wandb.login(key=os.environ.get("WANDB_API_KEY"),)
    wandb_run = wandb.init(
        project=train_config["project"],
        config={
            "train_config": train_config,
            "model_config": model_config,
            "vocab_file": args.vocab_file,
            "resume": args.resume
        },
        resume=args.resume is not None,
        dir=wandb_path
    )
    print(f"Wandb run initialized: {wandb_run.id}")
    return wandb_run


def init_tokenizer(vocab_file: str) -> CharTokenizer | BPETokenizer:
    """
    Initialize the tokenizer based on the vocabulary file.

    Args:
        vocab_file (str): Path to the vocabulary JSON file.

    Returns:
        CharTokenizer | BPETokenizer: An instance of the tokenizer.
    """
    if not os.path.isfile(vocab_file):
        print(f"Vocabulary file not found: {vocab_file}")
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_file}")

    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    if "<|UNK|>" in vocab:
        tokenizer = CharTokenizer(vocab=vocab)
    else:
        tokenizer = BPETokenizer()

    print(f"Tokenizer '{tokenizer.tokenizer_type}' initialized with vocab size: {tokenizer.vocab_size}")
    return tokenizer


def init_model(model_config: dict, tokenizer: CharTokenizer | BPETokenizer, device: torch.device) -> nn.Module:
    """
    Initialize the model based on the model configuration.

    Args:
        model_config (dict): Model configuration.
        tokenizer(CharTokenizer | BPETokenizer): Tokenizer instance.
        device (torch.device): Device to run the model on.

    Returns:
        model: The initialized model.
    """
    model_arch = model_config["name"].lower()

    if model_arch == "bigram":
        model = Bigram(BigramConfig(
            vocab_size=tokenizer.vocab_size
        ))

    elif model_arch == "mlp":
        model = MLP(MLPConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=model_config["context_size"],
            d_embed=model_config["d_embed"],
            d_ff=model_config["d_ff"]
        ))

    elif model_arch == "gpt":
        required_keys = ["context_size", "n_layer", "n_head", "d_embed", "d_ff", "dropout"]
        for key in required_keys:
            if key not in model_config:
                raise ValueError(f"Missing configuration for the {model_arch} model: {key}")

        model = GPT(GPTConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=model_config["context_size"],
            n_layer=model_config["n_layer"],
            n_head=model_config["n_head"],
            d_embed=model_config["d_embed"],
            d_ff=model_config["d_ff"],
            dropout=model_config["dropout"]
        ))

    elif model_arch == "megabyte":
        required_keys = ["pad_id", "patch_size", "global", "local"]
        for key in required_keys:
            if key not in model_config:
                raise ValueError(f"Missing configuration for the {model_arch} model: {key}")

        model = MEGABYTE(MegabyteConfig(
            vocab_size=tokenizer.vocab_size,
            pad_id=model_config["pad_id"],
            patch_size=model_config["patch_size"],
            patch_num=model_config["patch_num"],
            global_config=GPTConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=model_config["patch_size"] * model_config["patch_num"],
                n_layer=model_config["global"]["n_layer"],
                n_head=model_config["global"]["n_head"],
                d_embed=model_config["global"]["d_embed"],
                d_ff=model_config["global"]["d_ff"],
                dropout=model_config["global"]["dropout"]
            ),
            local_config=GPTConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=model_config["patch_size"],
                n_layer=model_config["local"]["n_layer"],
                n_head=model_config["local"]["n_head"],
                d_embed=model_config["local"]["d_embed"],
                d_ff=model_config["local"]["d_ff"],
                dropout=model_config["local"]["dropout"]
            )
        ))
    else:
        raise ValueError(f"Unsupported model type: {model_arch}")

    # Move the model to the device
    model.to(device)

    print(f"Model '{model_arch}' initialized."
          f" Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    return model


def setup_optimizer(model: nn.Module, optimizer_name: str, lr: float, weight_decay: float) -> Optimizer:
    """
    Set up the optimizer based on the configuration.

    Args:
        model (nn.Module): The model to optimize.
        optimizer_name (str): Name of the optimizer.
        lr (float): Learning rate.
        weight_decay (float): Weight decay.

    Returns:
        Optimizer: The initialized optimizer.
    """

    optimizer_name = optimizer_name.lower()
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    print(f"Optimizer '{optimizer_name}' initialized with learning rate: {lr}")
    return optimizer


def setup_scheduler(optimizer: Optimizer, scheduler_type: str, warmup_ratio: float, total_steps: int) -> LambdaLR:
    """
    Set up the learning rate scheduler based on the configuration.

    Args:
        optimizer (Optimizer): The optimizer to use.
        scheduler_type (str): Type of the scheduler.
        warmup_ratio (float): Ratio of the warmup steps.
        total_steps (int): Total number of steps.

    Returns:
        LambdaLR: The initialized scheduler.
    """
    assert 0 <= warmup_ratio <= 1, "Warmup ratio must be between 0 and 1"
    warmup_steps = int(total_steps * warmup_ratio)
    decay_steps = total_steps - warmup_steps

    scheduler_type = scheduler_type.lower()
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            progress = float(current_step - warmup_steps) / float(max(1, decay_steps))
            if scheduler_type == "linear":
                # Linear decay
                return max(0.0, 1.0 - progress)
            elif scheduler_type == "cosine":
                # Cosine decay
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            elif scheduler_type == "none":
                # No decay
                return 1.0
            else:
                raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    print(f"Scheduler '{scheduler_type}' initialized with warmup steps: {warmup_steps}")
    return scheduler


def split_text(text: str, val_size: float) -> Tuple[str, str]:
    """
    Split text into training and validation sets.

    Args:
        text (str): The data to split.
        val_size (float): Size of the validation set.

    Returns:
        Tuple[str, str]: Training and validation data.
    """
    if val_size <= 0 or val_size >= 1:
        raise ValueError(f"Invalid validation size: {val_size}")

    split_idx = int(len(text) * (1 - val_size))
    train_text, val_text = text[:split_idx], text[split_idx:]
    return train_text, val_text


class TextDataset(Dataset):
    def __init__(self, text: str, tokenizer: CharTokenizer | BPETokenizer, context_size: int):
        self.text = text
        self.tokenizer = tokenizer
        self.context_size = context_size

    def __len__(self) -> int:
        return len(self.tokenizer.encode(self.text)) - self.context_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids = self.tokenizer.encode(self.text[idx:idx + self.context_size])
        target_ids = self.tokenizer.encode(self.text[idx + 1:idx + self.context_size + 1])
        return input_ids, target_ids


def init_dataloader(text: str, tokenizer: CharTokenizer | BPETokenizer, batch_size: int, context_size: int, val_size: float) \
        -> Tuple[DataLoader, DataLoader]:
    """
    Initialize the training and validation DataLoaders.

    Args:
        text (str): The text data.
        tokenizer (CharTokenizer | BPETokenizer): Tokenizer instance.
        batch_size (int): Batch size for the DataLoaders.
        context_size (int): Context size for the model.
        val_size (float): Size of the validation set.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and validation DataLoaders.
    """
    if batch_size != 2 ** int(math.log2(batch_size)):
        import warnings
        warnings.warn("Batch size is not the power of 2, which may cause performance issues.")

    # Train-Validation split
    train_text, val_text = split_text(text, val_size)

    train_dataset = TextDataset(train_text, tokenizer, context_size)
    val_dataset = TextDataset(val_text, tokenizer, context_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Training data: {len(train_dataset)} samples, Validation data: {len(val_dataset)} samples")
    return train_loader, val_loader


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: Optimizer, scheduler: LambdaLR, current_epoch: int, total_epochs: int, grad_clip: float, device: torch.device, wandb_run: wandb.sdk.wandb_run.Run):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): DataLoader for the training data.
        optimizer (Optimizer): Optimizer to use.
        scheduler (lr_scheduler): Learning rate scheduler.
        current_epoch (int): Current epoch number.
        total_epochs (int): Total number of epochs.
        grad_clip (float): Gradient clipping value.
        device (torch.device): Device to run the model on.
        wandb_run (wandb.sdk.wandb_run.Run): Wandb run for logging.
    """
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {current_epoch+1}/{total_epochs}")

    for batch_idx, (inputs, targets) in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = model.loss(logits, targets)
        loss.backward()
        clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()
        progress_bar.set_postfix(loss=f"{running_loss / (batch_idx + 1):.4f}")

        if wandb_run is not None:
            wandb_run.log({
                "Train Loss": loss.item(),
                "Learning Rate": optimizer.param_groups[0]['lr']
            })

    progress_bar.close()
    print(f"Epoch {current_epoch+1}/{total_epochs} Loss: {running_loss / len(dataloader):.4f}")


def train_steps(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, optimizer: Optimizer, scheduler: LambdaLR, max_steps: int, val_interval: int, grad_clip: float, device: torch.device, wandb_run: wandb.sdk.wandb_run.Run):
    """
    Train the model for a fixed number of steps.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training data.
        val_loader (DataLoader): DataLoader for the validation data.
        optimizer (Optimizer): Optimizer to use.
        scheduler (lr_scheduler): Learning rate scheduler.
        max_steps (int): Maximum number of steps to train.
        grad_clip (float): Gradient clipping value.
        val_interval (int): Interval for validation.
        device (torch.device): Device to run the model on.
        wandb_run (wandb.sdk.wandb_run.Run): Wandb run for logging.
    """
    running_loss = 0.0
    progress_bar = tqdm(enumerate(train_loader), total=max_steps, desc="Training")

    for step in range(1, max_steps + 1):
        model.train()
        for batch_idx, (inputs, targets) in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = model.loss(logits, targets)
            loss.backward()
            clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            progress_bar.set_postfix(loss=f"{running_loss / step:.4f}")

            if step % val_interval == 0:
                evaluate(model, val_loader, device, wandb_run)

            if wandb_run is not None:
                wandb_run.log({
                    "Train Loss": loss.item(),
                    "Learning Rate": optimizer.param_groups[0]['lr']
                })

            step += 1

    progress_bar.close()


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device, wandb_run: wandb.sdk.wandb_run.Run):
    """
    Evaluate the model on the validation set.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for the validation data.
        device (torch.device): Device to run the model on.
        wandb_run (wandb.sdk.wandb_run.Run): Wandb run for logging.
    """
    model.eval()
    running_loss = 0.0
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Validation")

    with torch.no_grad():
        for batch_idx, (inputs, targets) in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = model.loss(logits, targets)
            running_loss += loss.item()
            progress_bar.set_postfix(loss=f"{running_loss / (batch_idx + 1):.4f}")

    progress_bar.close()

    avg_loss = running_loss / len(dataloader)
    perplexity = math.exp(avg_loss)

    if wandb_run is not None:
        wandb_run.log({
            "Validation Loss": avg_loss,
            "Perplexity": perplexity
        })

    print(f"Validation Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")


def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, train_config: dict, device: torch.device, wandb_run: wandb.sdk.wandb_run.Run):
    """
    Train the model.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training data.
        val_loader (DataLoader): DataLoader for the validation data.
        train_config (dict): Training configuration.
        device (torch.device): Device to run the model on.
        wandb_run (wandb.sdk.wandb_run.Run): Wandb run for logging.
    """
    optimizer = setup_optimizer(model, train_config["optimizer"]["name"], train_config["optimizer"]["params"]["lr"], train_config["optimizer"]["params"]["weight_decay"])

    if train_config["train_strategy"] == "epochs":
        print(f"Training for {train_config['epochs']} epochs")
        scheduler = setup_scheduler(optimizer, train_config["scheduler"]["type"], train_config["scheduler"]["warmup_ratio"], len(train_loader) * train_config["epochs"])
        for epoch in range(train_config["epochs"]):
            train_epoch(model, train_loader, optimizer, scheduler, epoch, train_config["epochs"], train_config["grad_clip"], device, wandb_run)
            evaluate(model, val_loader, device, wandb_run)

    elif train_config["train_strategy"] == "steps":
        print(f"Training for {train_config['max_steps']} steps")
        scheduler = setup_scheduler(optimizer, train_config["scheduler"]["type"], train_config["scheduler"]["warmup_ratio"], train_config["max_steps"])
        train_steps(model, train_loader, val_loader, optimizer, scheduler, train_config["max_steps"], train_config["val_interval"], train_config["grad_clip"], device, wandb_run)

    else:
        raise ValueError(f"Unsupported training strategy: {train_config['train_strategy']}")

    print("Training completed successfully!!")


def main():
    args = parse_args()

    # Root directory
    root_dir = os.path.dirname(os.path.abspath(__file__)) + "/../"

    # Load the configuration
    train_config = load_config(root_dir + "configs/train.yaml")
    model_config = load_config(root_dir + f"models/{args.model}/config.yaml")

    # Set the seed for reproducibility
    set_seed(train_config["seed"])

    # Configure the device
    device = configure_device()

    # Initialize wandb
    if not args.debug:
        wandb_run = init_wandb(args, train_config, model_config, root_dir)
    else:
        wandb_run = None
        print("Debug mode enabled")

    # Initialize the tokenizer and the model
    tokenizer = init_tokenizer(root_dir + args.vocab_file)
    model = init_model(model_config, tokenizer, device)

    # Load and preprocess the text data
    # For Shakespeare dataset, use PyTorch DataLoader
    if train_config["dataset"] == "shakespeare":
        text = load_text(root_dir + "data/shakespeare.txt")
        # MEGABYTE model -> context size = patch_size * patch_num
        if model_config["name"].lower() == "megabyte":
            train_loader, val_loader = init_dataloader(text, tokenizer, train_config["batch_size"], model_config["patch_size"] * model_config["patch_num"], train_config["val_size"])
        else:
            train_loader, val_loader = init_dataloader(text, tokenizer, train_config["batch_size"], model_config["context_size"], train_config["val_size"])

    # For OpenWebText dataset, use Hugging Face Datasets
# TODO: Implement OpenWebText dataset
    elif train_config["dataset"] == "openweb":
        #text_dataset = load_dataset("openwebtext", num_proc=4, trust_remote_code=True)

        # MEGABYTE model -> context size = patch_size * patch_num
        if model_config["name"].lower() == "megabyte":
            train_loader, val_loader = None, None
        else:
            train_loader, val_loader = None, None
    else:
        raise ValueError(f"Unsupported dataset: {train_config['dataset']}")

# TODO: Resume training from a checkpoint

    # Train the model
    try:
        train(model, train_loader, val_loader, train_config, device, wandb_run)
    except KeyboardInterrupt:
        print("Training interrupted by the user")
    except Exception as e:
        print(f"Error during training: {e}")

    # Save the model
    if not args.debug:
        save_checkpoint(model, root_dir + f"models/{model_config["name"]}/checkpoints/{model_config["name"]}.pt")

    # Finish wandb run
    if wandb_run is not None:
        wandb_run.log({"Number of Parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})
        wandb_run.save(root_dir + f"models/{model_config["name"]}/checkpoints/{model_config["name"]}.pt")
        wandb_run.finish()


if __name__ == "__main__":
    main()
