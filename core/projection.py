"""
core/projection.py
------------------
Forward projection (Radon transform) — sinogram computation.
No Qt dependencies.
"""

import numpy as np
from skimage.transform import radon


def compute_sinogram(phantom: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Compute the Radon-transform sinogram of a 2-D phantom.

    Parameters
    ----------
    phantom : np.ndarray, shape (H, W)
        Normalized 2-D image.
    theta : np.ndarray
        Projection angles in degrees (e.g. np.linspace(0, 180, n_angles)).

    Returns
    -------
    sinogram : np.ndarray, shape (n_detectors, n_angles)
        Clean sinogram.
    """
    return radon(phantom, theta=theta)
