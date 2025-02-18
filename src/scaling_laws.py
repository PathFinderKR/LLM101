# src/scaling_laws.py

import os
import sys
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode."
    )
    return parser.parse_args()


def plot_scaling_laws(x, y, x_label, y_label, title, wandb_run):
    """
    Plot the given data in log-log scale and log the figure to wandb.
    """
    from scipy.stats import linregress

    # Convert to log-space
    x_log = np.log10(x)
    y_log = np.log10(y)
    slope, intercept, r_value, p_value, std_err = linregress(x_log, y_log)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(6, 5))
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
    if wandb_run is not None:
        wandb.log({title: wandb.Image(fig)})
    plt.close(fig)


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: Optimizer, scheduler: LambdaLR, grad_clip: float, flops: int, device: torch.device, wandb_run: wandb.sdk.wandb_run.Run) -> (nn.Module, float):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): DataLoader for the training data.
        optimizer (Optimizer): Optimizer to use.
        scheduler (lr_scheduler): Learning rate scheduler.
        grad_clip (float): Gradient clipping value.
        flops (int): Floating point operations per second for the model.
        device (torch.device): Device to run the model on.
        wandb_run (wandb.sdk.wandb_run.Run): Wandb run for logging.

    Returns:
        model (nn.Module): The trained model.
        train_loss (float): The average loss on the training set.
    """
    model.train()
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
        progress_bar.set_postfix(loss=f"{running_loss / (batch_idx + 1):.4f}")

        if wandb_run is not None:
            wandb_run.log({
                "Train Loss vs Step": loss.item(),
                "Learning Rate vs Step": optimizer.param_groups[0]["lr"],
                "Train Loss vs Compute": loss.item(),
                "Compute": flops * (batch_idx + 1),
                "Step": (batch_idx + 1)
            })

    progress_bar.close()
    print(f"Loss: {running_loss / len(dataloader):.4f}")

    return model, best_train_loss


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device, flops: int, wandb_run: wandb.sdk.wandb_run.Run) -> float:
    """
    Evaluate the model on the validation set.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for the validation data.
        flops (int): Floating point operations per second for the model.
        device (torch.device): Device to run the model on.
        wandb_run (wandb.sdk.wandb_run.Run): Wandb run for logging.

    Returns:
        val_loss (float): The average loss on the validation set.
    """
    model.eval()
    running_loss = 0.0
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Validation")
    # Charts
    # Validation loss vs compute
    wandb_run.define_metric("Compute", hidden=True)
    wandb_run.define_metric("Validation Loss vs Compute", step_metric="Compute")

    with torch.no_grad():
        for batch_idx, (inputs, targets) in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = model.loss(logits, targets)
            running_loss += loss.item()
            progress_bar.set_postfix(loss=f"{running_loss / (batch_idx + 1):.4f}")

    progress_bar.close()

    val_loss = running_loss / len(dataloader)
    compute = flops * len(dataloader)
    print(f"compute: {compute}")

    if wandb_run is not None:
        wandb_run.log({
            "Validation Loss vs Compute": val_loss,
            "Compute": compute
        })

    print(f"Validation Loss: {val_loss:.4f}")

    return val_loss


def compute_experiment(
        model_sizes: dict, dataset_sizes: dict,
        train_text: str, val_text: str, tokenizer: CharTokenizer | BPETokenizer,
        optimizer_name: str, lr: float, weight_decay: float, scheduler_type: str, warmup_ratio: float,
        grad_clip: float, device: torch.device, root_dir: str
):
    """
    Compute vs test loss scaling laws.

    Args:
        model_sizes (dict): Dictionary with the model sizes.
        dataset_sizes (dict): Dictionary with the dataset sizes.
        train_text (str): Text data for training.
        val_text (str): Text data for validation.
        tokenizer (CharTokenizer | BPETokenizer): Tokenizer instance.
        optimizer_name (str): Name of the optimizer.
        lr (float): Learning rate.
        weight_decay (float): Weight decay.
        scheduler_type (str): Type of the scheduler.
        warmup_ratio (float): Ratio of the warmup steps.
        grad_clip (float): Gradient clipping value.
        device (torch.device): Device for training.
        root_dir (str): Root directory of the project.
    """
    compute_values = []
    train_losses = []
    test_losses = []
    project = "Compute vs Loss"

    for model_size in model_sizes:
        for dataset_size in dataset_sizes:
            wandb_run = wandb.init(
                project=project,
                name=f"Model size: {model_size} - Dataset size: {dataset_size}",
                dir=root_dir
            )
            print(f"Wandb run initialized: {wandb_run.id}")

            # Subset the training data
            subset_train_text = train_text[:int(len(train_text) * dataset_sizes[dataset_size])]
            train_dataset = TextDataset(text=subset_train_text, tokenizer=tokenizer, context_size=model_sizes[model_size]["context_size"])
            val_dataset = TextDataset(text=val_text, tokenizer=tokenizer, context_size=model_sizes[model_size]["context_size"])
            if model_size == "small":
                batch_size = 512
            elif model_size == "medium":
                batch_size = 128
            elif model_size == "large":
                batch_size = 64
            else:
                batch_size = 128
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            print(f"Number of tokens: {len(train_dataset)}")

            # Initialize the model
            model = GPT(GPTConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=model_sizes[model_size]["context_size"],
                n_layer=model_sizes[model_size]["n_layer"],
                n_head=model_sizes[model_size]["n_head"],
                d_embed=model_sizes[model_size]["d_embed"],
                d_ff=model_sizes[model_size]["d_ff"],
                dropout=model_sizes[model_size]["dropout"]
            )).to(device)
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            # FLOPs
            flops = 6 * num_params * len(train_dataset)
            print(f"FLOPs: {flops}")

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
            model, train_loss = train_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                grad_clip=grad_clip,
                flops=flops,
                device=device,
                wandb_run=wandb_run
            )
            test_loss = evaluate(
                model=model,
                dataloader=val_loader,
                flops=flops,
                device=device,
                wandb_run=wandb_run
            )

            compute_values.append(flops)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            wandb_run.finish()

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
        title="Compute vs Train Loss",
        wandb_run=wandb_run
    )
    plot_scaling_laws(
        x=compute_values,
        y=test_losses,
        x_label="Compute (FLOPs)",
        y_label="Test Loss",
        title="Compute vs Test Loss",
        wandb_run=wandb_run
    )
    wandb_run.finish()


def dataset_size_experiment(
        dataset_sizes: dict, model_size: dict,
        train_text: str, val_text: str, tokenizer: CharTokenizer | BPETokenizer, batch_size: int,
        optimizer_name: str, lr: float, weight_decay: float, scheduler_type: str, warmup_ratio: float,
        grad_clip: float, device: torch.device, root_dir: str
):
    """
    Dataset size vs test loss scaling laws.

    Args:
        dataset_sizes (dict): Dictionary with the dataset sizes.
        model_size (dict): Dictionary with the model size.
        train_text (str): Text data for training.
        val_text (str): Text data for validation.
        tokenizer (CharTokenizer | BPETokenizer): Tokenizer instance.
        batch_size (int): Batch size the DataLoaders.
        optimizer_name (str): Name of the optimizer.
        lr (float): Learning rate.
        weight_decay (float): Weight decay.
        scheduler_type (str): Type of the scheduler.
        warmup_ratio (float): Ratio of the warmup steps.
        grad_clip (float): Gradient clipping value.
        device (torch.device): Device for training.
        root_dir (str): Root directory of the project.
    """
    num_tokens = []
    train_losses = []
    test_losses = []
    project = "Dataset size vs Loss"

    for dataset_size in dataset_sizes:
        wandb_run = wandb.init(
            project=project,
            name=f"Dataset size: {dataset_size}",
            dir=root_dir
        )
        print(f"Wandb run initialized: {wandb_run.id}")

        subset_train_text = train_text[:int(len(train_text) * dataset_sizes[dataset_size])]
        train_dataset = TextDataset(text=subset_train_text, tokenizer=tokenizer, context_size=model_size["context_size"])
        val_dataset = TextDataset(text=val_text, tokenizer=tokenizer, context_size=model_size["context_size"])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        print(f"Number of tokens: {len(train_dataset)}")

        # Initialize the model
        model = GPT(GPTConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=model_size["context_size"],
            n_layer=model_size["n_layer"],
            n_head=model_size["n_head"],
            d_embed=model_size["d_embed"],
            d_ff=model_size["d_ff"],
            dropout=model_size["dropout"]
        )).to(device)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # FLOPs
        flops = 6 * num_params * len(train_dataset)

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
        model, train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            grad_clip=grad_clip,
            flops=flops,
            device=device,
            wandb_run=wandb_run
        )
        test_loss = evaluate(
            model=model,
            dataloader=val_loader,
            flops=flops,
            device=device,
            wandb_run=wandb_run
        )

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
        title="Dataset Size vs Train Loss",
        wandb_run=wandb_run
    )
    plot_scaling_laws(
        x=num_tokens,
        y=test_losses,
        x_label="Dataset Size",
        y_label="Test Loss",
        title="Dataset Size vs Test Loss",
        wandb_run=wandb_run
    )
    wandb_run.finish()


def model_size_experiment(
        model_sizes: dict, dataset_size: float,
        train_text: str, val_text: str, tokenizer: CharTokenizer | BPETokenizer,
        optimizer_name: str, lr: float, weight_decay: float, scheduler_type: str, warmup_ratio: float,
        grad_clip: float, device: torch.device, root_dir: str
):
    """
    Model size vs test loss scaling laws.

    Args:
        model_sizes (dict): Dictionary with the model sizes.
        dataset_size (float): Dictionary with the dataset sizes.
        train_text (str): Text data for training.
        val_text (str): Text data for validation.
        tokenizer (CharTokenizer | BPETokenizer): Tokenizer instance.
        optimizer_name (str): Name of the optimizer.
        lr (float): Learning rate.
        weight_decay (float): Weight decay.
        scheduler_type (str): Type of the scheduler.
        warmup_ratio (float): Ratio of the warmup steps.
        grad_clip (float): Gradient clipping value.
        device (torch.device): Device for training.
        root_dir (str): Root directory of the project.
    """
    parameters = []
    train_losses = []
    test_losses = []
    project = "Parameters vs Loss"

    for model_size in model_sizes:
        wandb_run = wandb.init(
            project=project,
            name=f"Model size: {model_size}",
            dir=root_dir
        )
        print(f"Wandb run initialized: {wandb_run.id}")

        subset_train_text = train_text[:int(len(train_text) * dataset_size)]
        train_dataset = TextDataset(text=subset_train_text, tokenizer=tokenizer, context_size=model_sizes[model_size]["context_size"])
        val_dataset = TextDataset(text=val_text, tokenizer=tokenizer, context_size=model_sizes[model_size]["context_size"])
        if model_size == "small":
            batch_size = 512
        elif model_size == "medium":
            batch_size = 128
        elif model_size == "large":
            batch_size = 64
        elif model_size == "xl":
            batch_size = 32
        else:
            batch_size = 128
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # Initialize the model
        model = GPT(GPTConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=model_sizes[model_size]["context_size"],
            n_layer=model_sizes[model_size]["n_layer"],
            n_head=model_sizes[model_size]["n_head"],
            d_embed=model_sizes[model_size]["d_embed"],
            d_ff=model_sizes[model_size]["d_ff"],
            dropout=model_sizes[model_size]["dropout"]
        )).to(device)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of parameters: {num_params}")

        # FLOPs
        flops = 6 * num_params * len(train_dataset)

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
        model, train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            grad_clip=grad_clip,
            flops=flops,
            device=device,
            wandb_run=wandb_run
        )
        test_loss = evaluate(
            model=model,
            dataloader=val_loader,
            flops=flops,
            device=device,
            wandb_run=wandb_run
        )

        parameters.append(num_params)
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
        title="Model Size vs Train Loss",
        wandb_run=wandb_run
    )
    plot_scaling_laws(
        x=parameters,
        y=test_losses,
        x_label="Number of Layers",
        y_label="Test Loss",
        title="Model Size vs Test Loss",
        wandb_run=wandb_run
    )
    wandb_run.finish()


def main():
    args = parse_args()


    # Root directory
    root_dir = os.path.dirname(os.path.abspath(__file__)) + "/../"


    # Load the configuration
    scaling_config = load_config(file_path=root_dir+"configs/scaling_laws.yaml")
    model_sizes = scaling_config["model_size"]
    dataset_sizes = scaling_config["dataset_size"]


    # Set the seed for reproducibility
    set_seed(seed=scaling_config["seed"])


    # Configure the device
    device = configure_device()


    # Initialize wandb
    if not args.debug:
        wandb.login(key=os.environ.get("WANDB_API_KEY"))
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
        compute_experiment(
            model_sizes=model_sizes,
            dataset_sizes=dataset_sizes,
            train_text=train_text,
            val_text=val_text,
            tokenizer=tokenizer,
            optimizer_name=scaling_config["optimizer"]["name"],
            lr=scaling_config["optimizer"]["params"]["lr"],
            weight_decay=scaling_config["optimizer"]["params"]["weight_decay"],
            scheduler_type=scaling_config["scheduler"]["type"],
            warmup_ratio=scaling_config["scheduler"]["warmup_ratio"],
            grad_clip=scaling_config["grad_clip"],
            device=device,
            root_dir=root_dir
        )
        dataset_size_experiment(
            dataset_sizes=dataset_sizes,
            model_size=model_sizes["medium"],
            train_text=train_text,
            val_text=val_text,
            tokenizer=tokenizer,
            batch_size=scaling_config["batch_size"],
            optimizer_name=scaling_config["optimizer"]["name"],
            lr=scaling_config["optimizer"]["params"]["lr"],
            weight_decay=scaling_config["optimizer"]["params"]["weight_decay"],
            scheduler_type=scaling_config["scheduler"]["type"],
            warmup_ratio=scaling_config["scheduler"]["warmup_ratio"],
            grad_clip=scaling_config["grad_clip"],
            device=device,
            root_dir=root_dir
        )
        model_size_experiment(
            model_sizes=model_sizes,
            dataset_size=dataset_sizes["medium"],
            train_text=train_text,
            val_text=val_text,
            tokenizer=tokenizer,
            optimizer_name=scaling_config["optimizer"]["name"],
            lr=scaling_config["optimizer"]["params"]["lr"],
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
