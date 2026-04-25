"""
core/metrics.py
---------------
Image quality metrics: RMSE and SSIM.
No Qt dependencies.
"""

import numpy as np
from skimage.metrics import structural_similarity as _ssim


def compute_rmse(reference: np.ndarray, image: np.ndarray) -> float:
    """
    Root Mean Square Error between a reference and a reconstructed image.

    Parameters
    ----------
    reference : np.ndarray
        Ground-truth image (e.g. original phantom).
    image : np.ndarray
        Reconstructed or processed image, same shape as reference.

    Returns
    -------
    rmse : float
    """
    return float(np.sqrt(np.mean((reference - image) ** 2)))


def compute_ssim(reference: np.ndarray, image: np.ndarray) -> float:
    """
    Structural Similarity Index Measure (SSIM).

    Parameters
    ----------
    reference : np.ndarray
        Ground-truth image, values in [0, 1].
    image : np.ndarray
        Reconstructed image, same shape and range as reference.

    Returns
    -------
    ssim_val : float  in [-1, 1], higher is better.
    """
    return float(_ssim(reference, image, data_range=1.0))
