from typing import List, Optional

import torch


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
