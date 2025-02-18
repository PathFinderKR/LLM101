# src/scaling_laws.py

import os
import sys
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from typing import List
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_
import wandb
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import set_seed, configure_device, load_text, split_text, load_config
from src.tokenizer import CharTokenizer, BPETokenizer
from src.train import TextDataset, initialize_tokenizer, setup_optimizer, setup_scheduler
from models.mlp.mlp import MLP, MLPConfig
from models.gpt.gpt import GPT, GPTConfig


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Command-line arguments.
    """
    default_vocab_path = "char_vocab.json"

    parser = argparse.ArgumentParser(description="Scaling Laws for Neural Language Models.")
    parser.add_argument(
        "--vocab_path",
        type=str,
        default=default_vocab_path,
        help=f"Path to the vocabulary JSON file. Default: {default_vocab_path}"
    )
    return parser.parse_args()


def plot_scaling_laws(x: List[int], y: List[float], x_label: str, y_label: str, title: str, wandb_run: wandb.sdk.wandb_run.Run):
    """
    Plot the given data in log-log scale and log the figure to wandb.

    Args:
        x (List[int]): The x-axis data.
        y (List[float]): The y-axis data.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        title (str): Title of the plot.
        wandb_run (wandb.sdk.wandb_run.Run): The current wandb run, used for logging.
    """
    from scipy.stats import linregress

    # Convert to log-space
    x_log = np.log10(x)
    y_log = np.log10(y)
    slope, intercept, r_value, p_value, std_err = linregress(x_log, y_log)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(x, y, label='Data', alpha=0.7)

    # Best fit line in log space
    fit_line_x = np.linspace(x_log.min(), x_log.max(), 100)
    fit_line_y = slope * fit_line_x + intercept
    ax.plot(10 ** fit_line_x, 10 ** fit_line_y, 'r--',
            label=f'Fit: y = {10 ** intercept:.2f} * x^{slope:.2f}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()

    # Log the figure to wandb
    wandb_run.log({title: wandb.Image(fig)})
    plt.close(fig)


def plot_umap(model: nn.Module, tokenizer: CharTokenizer, title: str, wandb_run: wandb.sdk.wandb_run.Run):
    """
    Creates a UMAP projection of the character (or token) embeddings and logs the resulting 2D scatter plot.

    Args:
        model (nn.Module): The trained model.
        tokenizer (CharTokenizer): The character-level tokenizer with idx2char mapping.
        title (str): Title of the plot.
        wandb_run (wandb.sdk.wandb_run.Run): The current wandb run, used for logging.
    """
    import umap

    # Extract token embeddings from the model
    model.eval()
    with torch.no_grad():
        if hasattr(model, "token_embedding"):
            embeddings = model.token_embedding.weight.detach().cpu().numpy()
        elif hasattr(model, "embedding"):
            embeddings = model.embedding.weight.detach().cpu().numpy()
        else:
            raise ValueError("Unable to locate the embedding layer in the provided model.")

    # Reduce the dimensionality to 2D with UMAP
    reducer = umap.UMAP(
        n_neighbors=5,
        min_dist=0.3,
        metric="cosine"
    )
    embedding_2d = reducer.fit_transform(embeddings)  # (vocab_size, 2)

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(f"Token Embeddings UMAP {title}")

    # Plot each character
    for i in range(tokenizer.vocab_size):
        char = tokenizer.idx2char.get(i, "<UNK>")
        x, y = embedding_2d[i]
        ax.scatter(x, y, color="blue", alpha=0.7)
        ax.text(x, y, char, fontsize=9, alpha=0.9)

    # Log the figure to wandb
    wandb_run.log({f"Token Embeddings UMAP {title}": wandb.Image(fig)})
    plt.close(fig)


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: Optimizer, scheduler: LambdaLR, grad_clip: float, flop_per_step: int, device: torch.device, wandb_run: wandb.sdk.wandb_run.Run) -> (nn.Module, float):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): DataLoader for the training data.
        optimizer (Optimizer): Optimizer to use.
        scheduler (lr_scheduler): Learning rate scheduler.
        grad_clip (float): Gradient clipping value.
        flop_per_step (int): Floating point operations per step.
        device (torch.device): Device to run the model on.
        wandb_run (wandb.sdk.wandb_run.Run): Wandb run for logging.

    Returns:
        model (nn.Module): The trained model.
        train_loss (float): The average loss on the training set.
    """
    model.train()
    steps = 0
    running_loss = 0.0
    best_train_loss = float("inf")
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Training")
    # Charts
    # Train loss vs step
    # Learning rate vs step
    # Train loss vs compute
    wandb_run.define_metric("Step", hidden=True)
    wandb_run.define_metric("Compute", hidden=True)
    wandb_run.define_metric("Train Loss vs Step", step_metric="Step")
    wandb_run.define_metric("Learning Rate vs Step", step_metric="Step")
    wandb_run.define_metric("Train Loss vs Compute", step_metric="Compute")

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
        if loss.item() < best_train_loss:
            best_train_loss = loss.item()

        steps += 1
        progress_bar.set_postfix(loss=f"{running_loss / steps:.4f}")

        wandb_run.log({
            "Train Loss vs Step": loss.item(),
            "Learning Rate vs Step": optimizer.param_groups[0]["lr"],
            "Train Loss vs Compute": loss.item(),
            "Compute": flop_per_step * steps,
            "Step": steps
        })

    progress_bar.close()
    return model, best_train_loss, steps


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device, flop_per_step: int, wandb_run: wandb.sdk.wandb_run.Run) -> float:
    """
    Evaluate the model on the validation set.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for the validation data.
        flop_per_step (int): Floating point operations per step.
        device (torch.device): Device to run the model on.
        wandb_run (wandb.sdk.wandb_run.Run): Wandb run for logging.

    Returns:
        val_loss (float): The average loss on the validation set.
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

    val_loss = running_loss / len(dataloader)

    wandb_run.log(
        {"Validation Loss vs Compute": val_loss},
        step=flop_per_step * len(dataloader))

    return val_loss


def compute_experiment(
        model_architectures: dict, dataset_sizes: dict,
        train_text: str, val_text: str, tokenizer: CharTokenizer | BPETokenizer,
        optimizer_name: str, weight_decay: float, scheduler_type: str, warmup_ratio: float,
        grad_clip: float, device: torch.device, root_dir: str
):
    """
    Compute vs test loss scaling laws.

    Args:
        model_architectures (dict): Dictionary with the model architectures and configurations.
        dataset_sizes (dict): Dictionary with the dataset sizes.
        train_text (str): Text data for training.
        val_text (str): Text data for validation.
        tokenizer (CharTokenizer | BPETokenizer): Tokenizer instance.
        optimizer_name (str): Name of the optimizer.
        weight_decay (float): Weight decay.
        scheduler_type (str): Type of the scheduler.
        warmup_ratio (float): Ratio of the warmup steps.
        grad_clip (float): Gradient clipping value.
        device (torch.device): Device for training.
        root_dir (str): Root directory of the project.
    """
    project = "Compute vs Loss"
    for architecture, model_configs in model_architectures.items():
        compute_values = []
        train_losses = []
        test_losses = []

        for model_size, model_config in model_configs.items():
            for dataset_size in dataset_sizes:
                name = f"{architecture}({model_size}), Dataset({dataset_size})"
                wandb_run = wandb.init(
                    project=project,
                    name=name,
                    dir=root_dir
                )

                # Subset the training data
                subset_train_text = train_text[:int(len(train_text) * dataset_sizes[dataset_size])]
                train_dataset = TextDataset(text=subset_train_text, tokenizer=tokenizer, context_size=model_config["context_size"])
                val_dataset = TextDataset(text=val_text, tokenizer=tokenizer, context_size=model_config["context_size"])
                if model_size == "small":
                    batch_size = 512
                    lr = 0.001
                elif model_size == "medium":
                    batch_size = 128
                    lr = 0.0005
                elif model_size == "large":
                    batch_size = 64
                    lr = 0.0001
                elif model_size == "xl":
                    batch_size = 32
                    lr = 0.00005
                else:
                    batch_size = 128
                    lr = 0.0001
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
                print(f"Number of tokens: {len(train_dataset)}")

                # Initialize the model
                if architecture == "gpt":
                    model = GPT(GPTConfig(
                        vocab_size=tokenizer.vocab_size,
                        context_size=model_config["context_size"],
                        n_layer=model_config["n_layer"],
                        n_head=model_config["n_head"],
                        d_embed=model_config["d_embed"],
                        d_ff=model_config["d_ff"],
                        dropout=model_config["dropout"]
                    )).to(device)
                elif architecture == "mlp":
                    model = MLP(MLPConfig(
                        vocab_size=tokenizer.vocab_size,
                        context_size=model_config["context_size"],
                        d_embed=model_config["d_embed"],
                        d_ff=model_config["d_ff"],
                        dropout=model_config["dropout"]
                    )).to(device)
                else:
                    raise ValueError(f"Model architecture {architecture} is not supported")
                num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"Number of parameters: {num_params}")

                # FLOPs
                flop_per_step = 6 * num_params * batch_size

                # Initialize the optimizer and scheduler
                optimizer = setup_optimizer(
                    model=model,
                    optimizer_name=optimizer_name,
                    lr=lr,
                    weight_decay=weight_decay
                )
                scheduler = setup_scheduler(
                    optimizer=optimizer,
                    scheduler_type=scheduler_type,
                    warmup_ratio=warmup_ratio,
                    total_steps=len(train_loader) * 1
                )

                # Train the model for one epoch
                model, train_loss, steps = train_epoch(
                    model=model,
                    dataloader=train_loader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    grad_clip=grad_clip,
                    flop_per_step=flop_per_step,
                    device=device,
                    wandb_run=wandb_run
                )
                test_loss = evaluate(
                    model=model,
                    dataloader=val_loader,
                    flop_per_step=flop_per_step,
                    device=device,
                    wandb_run=wandb_run
                )

                plot_umap(model, tokenizer, name, wandb_run)
                compute_values.append(flop_per_step * steps)
                train_losses.append(train_loss)
                test_losses.append(test_loss)
                wandb_run.finish()

        print(f"architecture: {architecture}")
        wandb_run = wandb.init(
            project=project,
            name=project,
            dir=root_dir
        )
        plot_scaling_laws(
            x=compute_values,
            y=train_losses,
            x_label="Compute (FLOPs)",
            y_label="Train Loss",
            title=f"Compute vs Train Loss ({architecture})",
            wandb_run=wandb_run
        )
        plot_scaling_laws(
            x=compute_values,
            y=test_losses,
            x_label="Compute (FLOPs)",
            y_label="Test Loss",
            title=f"Compute vs Test Loss ({architecture})",
            wandb_run=wandb_run
        )
        wandb_run.finish()


def dataset_size_experiment(
        model_architectures: dict, dataset_sizes: dict,
        train_text: str, val_text: str, tokenizer: CharTokenizer | BPETokenizer,
        optimizer_name: str, weight_decay: float, scheduler_type: str, warmup_ratio: float,
        grad_clip: float, device: torch.device, root_dir: str
):
    """
    Dataset size vs test loss scaling laws.

    Args:
        model_architectures (dict): Dictionary with the model architectures and configurations.
        dataset_sizes (dict): Dictionary with the dataset sizes.
        train_text (str): Text data for training.
        val_text (str): Text data for validation.
        tokenizer (CharTokenizer | BPETokenizer): Tokenizer instance.
        optimizer_name (str): Name of the optimizer.
        weight_decay (float): Weight decay.
        scheduler_type (str): Type of the scheduler.
        warmup_ratio (float): Ratio of the warmup steps.
        grad_clip (float): Gradient clipping value.
        device (torch.device): Device for training.
        root_dir (str): Root directory of the project.
    """
    project = "Dataset size vs Loss"
    for architecture, model_configs in model_architectures.items():
        num_tokens = []
        train_losses = []
        test_losses = []

        for model_size, model_config in model_configs.items():
            if model_size != "large":
                continue
            for dataset_size in dataset_sizes:
                name = f"{architecture}({model_size}), Dataset({dataset_size})"
                wandb_run = wandb.init(
                    project=project,
                    name=name,
                    dir=root_dir
                )

                subset_train_text = train_text[:int(len(train_text) * dataset_sizes[dataset_size])]
                train_dataset = TextDataset(text=subset_train_text, tokenizer=tokenizer, context_size=model_config["context_size"])
                val_dataset = TextDataset(text=val_text, tokenizer=tokenizer, context_size=model_config["context_size"])
                if model_size == "small":
                    batch_size = 512
                    lr = 0.001
                elif model_size == "medium":
                    batch_size = 128
                    lr = 0.0005
                elif model_size == "large":
                    batch_size = 64
                    lr = 0.0001
                elif model_size == "xl":
                    batch_size = 32
                    lr = 0.00005
                else:
                    batch_size = 128
                    lr = 0.0001
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
                print(f"Number of tokens: {len(train_dataset)}")

                # Initialize the model sdf
                if architecture == "gpt":
                    model = GPT(GPTConfig(
                        vocab_size=tokenizer.vocab_size,
                        context_size=model_config["context_size"],
                        n_layer=model_config["n_layer"],
                        n_head=model_config["n_head"],
                        d_embed=model_config["d_embed"],
                        d_ff=model_config["d_ff"],
                        dropout=model_config["dropout"]
                    )).to(device)
                elif architecture == "mlp":
                    model = MLP(MLPConfig(
                        vocab_size=tokenizer.vocab_size,
                        context_size=model_config["context_size"],
                        d_embed=model_config["d_embed"],
                        d_ff=model_config["d_ff"],
                        dropout=model_config["dropout"]
                    )).to(device)
                else:
                    raise ValueError(f"Model architecture {architecture} is not supported")
                num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"Number of parameters: {num_params}")

                # FLOPs
                flop_per_step = 6 * num_params * batch_size

                # Initialize the optimizer and scheduler
                optimizer = setup_optimizer(
                    model=model,
                    optimizer_name=optimizer_name,
                    lr=lr,
                    weight_decay=weight_decay
                )
                scheduler = setup_scheduler(
                    optimizer=optimizer,
                    scheduler_type=scheduler_type,
                    warmup_ratio=warmup_ratio,
                    total_steps=len(train_loader) * 1
                )

                # Train the model for one epoch
                model, train_loss, steps = train_epoch(
                    model=model,
                    dataloader=train_loader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    grad_clip=grad_clip,
                    flop_per_step=flop_per_step,
                    device=device,
                    wandb_run=wandb_run
                )
                test_loss = evaluate(
                    model=model,
                    dataloader=val_loader,
                    flop_per_step=flop_per_step,
                    device=device,
                    wandb_run=wandb_run
                )

                plot_umap(model, tokenizer, name, wandb_run)
                num_tokens.append(len(train_dataset))
                train_losses.append(train_loss)
                test_losses.append(test_loss)
                wandb_run.finish()

        wandb_run = wandb.init(
            project=project,
            name=project,
            dir=root_dir
        )
        plot_scaling_laws(
            x=num_tokens,
            y=train_losses,
            x_label="Dataset Size",
            y_label="Train Loss",
            title=f"Dataset Size vs Train Loss ({architecture})",
            wandb_run=wandb_run
        )
        plot_scaling_laws(
            x=num_tokens,
            y=test_losses,
            x_label="Dataset Size",
            y_label="Test Loss",
            title=f"Dataset Size vs Test Loss ({architecture})",
            wandb_run=wandb_run
        )
        wandb_run.finish()


def model_size_experiment(
        model_architectures: dict, dataset_sizes: dict,
        train_text: str, val_text: str, tokenizer: CharTokenizer | BPETokenizer,
        optimizer_name: str, weight_decay: float, scheduler_type: str, warmup_ratio: float,
        grad_clip: float, device: torch.device, root_dir: str
):
    """
    Model size vs test loss scaling laws.

    Args:
        model_architectures (dict): Dictionary with the model architectures and configurations.
        dataset_sizes (dict): Dictionary with the dataset sizes.
        train_text (str): Text data for training.
        val_text (str): Text data for validation.
        tokenizer (CharTokenizer | BPETokenizer): Tokenizer instance.
        optimizer_name (str): Name of the optimizer.
        weight_decay (float): Weight decay.
        scheduler_type (str): Type of the scheduler.
        warmup_ratio (float): Ratio of the warmup steps.
        grad_clip (float): Gradient clipping value.
        device (torch.device): Device for training.
        root_dir (str): Root directory of the project.
    """
    project = f"Model size vs Loss"
    for architecture, model_configs in model_architectures.items():
        parameters = []
        train_losses = []
        test_losses = []

        for model_size, model_config in model_configs.items():
            for dataset_size in dataset_sizes:
                if dataset_size != "large":
                    continue
                name = f"{architecture}({model_size}), Dataset({dataset_size})"
                wandb_run = wandb.init(
                    project=project,
                    name=name,
                    dir=root_dir
                )

                subset_train_text = train_text[:int(len(train_text) * dataset_sizes[dataset_size])]
                train_dataset = TextDataset(text=subset_train_text, tokenizer=tokenizer, context_size=model_config["context_size"])
                val_dataset = TextDataset(text=val_text, tokenizer=tokenizer, context_size=model_config["context_size"])
                if model_size == "small":
                    batch_size = 512
                    lr = 0.001
                elif model_size == "medium":
                    batch_size = 128
                    lr = 0.0005
                elif model_size == "large":
                    batch_size = 64
                    lr = 0.0001
                elif model_size == "xl":
                    batch_size = 32
                    lr = 0.00005
                else:
                    batch_size = 128
                    lr = 0.0001
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
                print(f"Number of tokens: {len(train_dataset)}")

                # Initialize the model
                if architecture == "gpt":
                    model = GPT(GPTConfig(
                        vocab_size=tokenizer.vocab_size,
                        context_size=model_config["context_size"],
                        n_layer=model_config["n_layer"],
                        n_head=model_config["n_head"],
                        d_embed=model_config["d_embed"],
                        d_ff=model_config["d_ff"],
                        dropout=model_config["dropout"]
                    )).to(device)
                elif architecture == "mlp":
                    model = MLP(MLPConfig(
                        vocab_size=tokenizer.vocab_size,
                        context_size=model_config["context_size"],
                        d_embed=model_config["d_embed"],
                        d_ff=model_config["d_ff"],
                        dropout=model_config["dropout"]
                    )).to(device)
                else:
                    raise ValueError(f"Model architecture {architecture} is not supported")
                num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"Number of parameters: {num_params}")

                # FLOPs
                flop_per_step = 6 * num_params * batch_size

                # Initialize the optimizer and scheduler
                optimizer = setup_optimizer(
                    model=model,
                    optimizer_name=optimizer_name,
                    lr=lr,
                    weight_decay=weight_decay
                )
                scheduler = setup_scheduler(
                    optimizer=optimizer,
                    scheduler_type=scheduler_type,
                    warmup_ratio=warmup_ratio,
                    total_steps=len(train_loader) * 1
                )

                # Train the model for one epoch
                model, train_loss, steps = train_epoch(
                    model=model,
                    dataloader=train_loader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    grad_clip=grad_clip,
                    flop_per_step=flop_per_step,
                    device=device,
                    wandb_run=wandb_run
                )
                test_loss = evaluate(
                    model=model,
                    dataloader=val_loader,
                    flop_per_step=flop_per_step,
                    device=device,
                    wandb_run=wandb_run
                )

                plot_umap(model, tokenizer, name, wandb_run)
                parameters.append(num_params)
                train_losses.append(train_loss)
                test_losses.append(test_loss)
                wandb_run.finish()

        wandb_run = wandb.init(
            project=project,
            name=project,
            dir=root_dir
        )
        plot_scaling_laws(
            x=parameters,
            y=train_losses,
            x_label="Number of Parameters",
            y_label="Train Loss",
            title=f"Model Size vs Train Loss ({architecture})",
            wandb_run=wandb_run
        )
        plot_scaling_laws(
            x=parameters,
            y=test_losses,
            x_label="Number of Parameters",
            y_label="Test Loss",
            title=f"Model Size vs Test Loss ({architecture})",
            wandb_run=wandb_run
        )

        wandb_run.finish()


def main():
    args = parse_args()


    # Root directory
    root_dir = os.path.dirname(os.path.abspath(__file__)) + "/../"


    # Load the configuration
    scaling_config = load_config(file_path=root_dir+"configs/scaling_laws.yaml")
    model_architectures = {
        "gpt": scaling_config["gpt"],
        "mlp": scaling_config["mlp"]
    }
    dataset_sizes = scaling_config["dataset_size"]


    # Set the seed for reproducibility
    set_seed(seed=scaling_config["seed"])


    # Configure the device
    device = configure_device()


    # Initialize wandb
    wandb.login(key=os.environ.get("WANDB_API_KEY"))


    # Initialize the tokenizer
    tokenizer = initialize_tokenizer(file_path=root_dir+args.vocab_path)


    # Load and the text data
    text = load_text(file_path=root_dir+"data/shakespeare.txt")
    train_text, val_text = split_text(text=text, val_size=scaling_config["val_size"])


    # Run the experiment
    try:
        compute_experiment(
            model_architectures=model_architectures,
            dataset_sizes=dataset_sizes,
            train_text=train_text,
            val_text=val_text,
            tokenizer=tokenizer,
            optimizer_name=scaling_config["optimizer"]["name"],
            weight_decay=scaling_config["optimizer"]["params"]["weight_decay"],
            scheduler_type=scaling_config["scheduler"]["type"],
            warmup_ratio=scaling_config["scheduler"]["warmup_ratio"],
            grad_clip=scaling_config["grad_clip"],
            device=device,
            root_dir=root_dir
        )
        dataset_size_experiment(
            model_architectures=model_architectures,
            dataset_sizes=dataset_sizes,
            train_text=train_text,
            val_text=val_text,
            tokenizer=tokenizer,
            optimizer_name=scaling_config["optimizer"]["name"],
            weight_decay=scaling_config["optimizer"]["params"]["weight_decay"],
            scheduler_type=scaling_config["scheduler"]["type"],
            warmup_ratio=scaling_config["scheduler"]["warmup_ratio"],
            grad_clip=scaling_config["grad_clip"],
            device=device,
            root_dir=root_dir
        )
        model_size_experiment(
            model_architectures=model_architectures,
            dataset_sizes=dataset_sizes,
            train_text=train_text,
            val_text=val_text,
            tokenizer=tokenizer,
            optimizer_name=scaling_config["optimizer"]["name"],
            weight_decay=scaling_config["optimizer"]["params"]["weight_decay"],
            scheduler_type=scaling_config["scheduler"]["type"],
            warmup_ratio=scaling_config["scheduler"]["warmup_ratio"],
            grad_clip=scaling_config["grad_clip"],
            device=device,
            root_dir=root_dir
        )

    except KeyboardInterrupt:
        print("Training interrupted by the user")
    except Exception as e:
        print(f"Error during training: {e}")


if __name__ == "__main__":
    main()
