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
from src.utils import set_seed, configure_device, load_text, split_text, load_config, save_checkpoint, load_checkpoint
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
    default_vocab_path = "char_vocab.json"

    parser = argparse.ArgumentParser(description="Train a language model.")
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


def initialize_tokenizer(file_path: str) -> CharTokenizer | BPETokenizer:
    """
    Initialize the tokenizer based on the vocabulary file.

    Args:
        file_path (str): Path to the vocabulary JSON file.

    Returns:
        CharTokenizer | BPETokenizer: An instance of the tokenizer.
    """
    if not os.path.isfile(file_path):
        print(f"Vocabulary file not found: {file_path}")
        raise FileNotFoundError(f"Vocabulary file not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    if "<|UNK|>" in vocab:
        tokenizer = CharTokenizer(vocab=vocab)
    else:
        tokenizer = BPETokenizer()

    print(f"Tokenizer '{tokenizer.tokenizer_type}' initialized with vocab size: {tokenizer.vocab_size}")
    return tokenizer


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
    progress_bar = tqdm(enumerate(train_loader, start=1), total=max_steps, desc="Training")

    for step, (inputs, targets) in progress_bar:
        if step > max_steps:
            break
        model.train()
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


def main():
    args = parse_args()


    # Root directory
    root_dir = os.path.dirname(os.path.abspath(__file__)) + "/../"


    # Load the configuration
    train_config = load_config(file_path=root_dir+"config/train.yaml")
    model_config = load_config(file_path=root_dir+f"models/{args.model}/config.yaml")


    # Set the seed for reproducibility
    set_seed(seed=train_config["seed"])


    # Configure the device
    device = configure_device()


    # Initialize wandb
    if not args.debug:
        wandb.login(key=os.environ.get("WANDB_API_KEY"))
        wandb_run = wandb.init(
            project=train_config["project"],
            config={
                "train_config": train_config,
                "model_config": model_config,
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
    wandb_run.log({"Number of Parameters": num_params})
    print(f"Model '{args.model}' initialized. Number of Parameters: {num_params}")


    # Load and preprocess the text data
    # For Shakespeare dataset, use PyTorch DataLoader
    if train_config["dataset"] == "shakespeare":
        text = load_text(file_path=root_dir+"data/shakespeare.txt")
        train_text, val_text = split_text(text=text, val_size=train_config["val_size"])
        if args.model == "megabyte":
            # MEGABYTE model -> context size = patch_size * patch_num
            train_dataset = TextDataset(text=train_text, tokenizer=tokenizer, context_size=model_config["patch_size"]*model_config["patch_num"])
            val_dataset = TextDataset(text=val_text, tokenizer=tokenizer, context_size=model_config["patch_size"]*model_config["patch_num"])
            train_loader = DataLoader(train_dataset, batch_size=train_config["batch_size"], shuffle=True, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=train_config["batch_size"], shuffle=False, num_workers=4)
        else:
            train_dataset = TextDataset(text=train_text, tokenizer=tokenizer, context_size=model_config["context_size"])
            val_dataset = TextDataset(text=val_text, tokenizer=tokenizer, context_size=model_config["context_size"])
            train_loader = DataLoader(train_dataset, batch_size=train_config["batch_size"], shuffle=True, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=train_config["batch_size"], shuffle=False, num_workers=4)

    # For OpenWebText dataset, use Hugging Face Datasets
    # TODO: Implement OpenWebText dataset
    elif train_config["dataset"] == "openweb":
        #text_dataset = load_dataset("openwebtext", num_proc=4, trust_remote_code=True)
        if args.model == "megabyte":
            # MEGABYTE model -> context size = patch_size * patch_num
            train_loader, val_loader = None, None
        else:
            train_loader, val_loader = None, None
    else:
        raise ValueError(f"Unsupported dataset: {train_config['dataset']}")


    # TODO: Resume training from a checkpoint


    # Initialize the optimizer
    optimizer = setup_optimizer(
        model=model,
        optimizer_name=train_config["optimizer"]["name"],
        lr=train_config["optimizer"]["params"]["lr"],
        weight_decay=train_config["optimizer"]["params"]["weight_decay"]
    )

    # Train the model
    try:
        if train_config["train_strategy"] == "epochs":
            print(f"Training for {train_config['epochs']} epochs")
            scheduler = setup_scheduler(
                optimizer=optimizer,
                scheduler_type=train_config["scheduler"]["type"],
                warmup_ratio=train_config["scheduler"]["warmup_ratio"],
                total_steps=len(train_loader) * train_config["epochs"]
            )
            for epoch in range(train_config["epochs"]):
                train_epoch(
                    model=model,
                    dataloader=train_loader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    current_epoch=epoch,
                    total_epochs=train_config["epochs"],
                    grad_clip=train_config["grad_clip"],
                    device=device,
                    wandb_run=wandb_run
                )
                evaluate(
                    model=model,
                    dataloader=val_loader,
                    device=device,
                    wandb_run=wandb_run
                )

        elif train_config["train_strategy"] == "steps":
            print(f"Training for {train_config['max_steps']} steps")
            scheduler = setup_scheduler(
                optimizer=optimizer,
                scheduler_type=train_config["scheduler"]["type"],
                warmup_ratio=train_config["scheduler"]["warmup_ratio"],
                total_steps=train_config["max_steps"]
            )
            train_steps(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                max_steps=train_config["max_steps"],
                val_interval=train_config["val_interval"],
                grad_clip=train_config["grad_clip"],
                device=device,
                wandb_run=wandb_run
            )

        else:
            raise ValueError(f"Unsupported training strategy: {train_config['train_strategy']}")

    except KeyboardInterrupt:
        print("Training interrupted by the user")
    except Exception as e:
        print(f"Error during training: {e}")

    # Save the model
    if not args.debug:
        save_checkpoint(model=model, file_path=root_dir+f"models/{model_config["name"]}/checkpoints/{model_config["name"]}.pt")

    # Finish wandb run
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
