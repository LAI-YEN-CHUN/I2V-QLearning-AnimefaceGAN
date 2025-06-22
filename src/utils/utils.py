import os
import shutil
from typing import List, Optional

import torch

from configs import BaseConfig


def set_random_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.

    This function sets the random seed for various libraries to ensure that results are consistent across runs.

    Args:
        seed (int): The seed value to set for random number generation.
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_allowed_file(filename: str, allowed_extensions: Optional[List[str]] = None) -> bool:
    """
    Check if a file has an allowed extension.

    This function verifies if a given filename has an extension that is included in the list of allowed extensions.

    Args:
        filename (str): The name of the file to check.
        allowed_extensions (Optional[List[str]], optional): A list of allowed file extensions.
            Defaults to ['jpg', 'jpeg', 'png'] if None is provided.

    Returns:
        bool: True if the file extension is allowed, False otherwise.

    Example:
        >>> is_allowed_file("image.jpg")
        True
        >>> is_allowed_file("document.txt", ["txt", "pdf"])
        True
        >>> is_allowed_file("script.py")
        False
    """
    if allowed_extensions is None:
        allowed_extensions = ['jpg', 'jpeg', 'png']
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


def initialize_directories(cfg: BaseConfig) -> None:
    """
    Create all necessary directories if they do not exist.
    """
    # Remove existing results directory if it exists
    if os.path.exists(cfg.RESULTS_DIR):
        shutil.rmtree(cfg.RESULTS_DIR)

    # Create the all necessary directories
    directories = [
        cfg.RAW_DATA_DIR,
        cfg.RAW_IMAGES_DIR,
        cfg.PROCESSED_DATA_DIR,
        cfg.PROCESSED_IMAGES_DIR,
        cfg.GENERATOR_CHECKPOINTS_DIR,
        cfg.DISCRIMINATOR_CHECKPOINTS_DIR,
        cfg.SAMPLES_DIR,
        cfg.LOGS_DIR,
        cfg.FIGURES_DIR,
        cfg.FID_REAL_IMAGES_DIR,
        cfg.FID_FAKE_IMAGES_DIR
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
