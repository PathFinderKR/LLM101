# src/utils.py

import os
import random
import numpy as np
import yaml
from typing import Tuple
import torch
import torch.nn as nn


def set_seed(seed: int):
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def configure_device() -> torch.device:
    """
    Configure the device for training.

    Returns:
        torch.device: The device to use for training.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        num_gpu = torch.cuda.device_count()
        print(f"Running on {num_gpu} {torch.cuda.get_device_name()} GPU(s)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Running on {device}")
    else:
        device = torch.device("cpu")
        print(f"Running on {device}")
    return device


def load_text(file_path: str, encoding: str = 'utf-8') -> str:
    """
    Load and read text data from a file.

    Args:
        file_path (str): Path to the text file.
        encoding (str, optional): File encoding. Defaults to 'utf-8'.

    Returns:
        str: The content of the text file.
    """
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r', encoding=encoding) as f:
        text = f.read()

    print(f"Loaded text data from {file_path} (length: {len(text)} characters).")
    return text


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


def load_config(file_path: str) -> dict:
    """
    Load YAML configuration file.

    Args:
        file_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Loaded configuration from {file_path}.")
    return config


def save_checkpoint(model: nn.Module, file_path: str):
    """
    Save the model checkpoint.

    Args:
        model (nn.Module): The model to save.
        file_path (str): Path to save the model.
    """
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")


def load_checkpoint():
    pass
