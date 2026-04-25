"""
core/phantom.py
---------------
Shepp-Logan phantom generation and normalization.
No Qt dependencies.
"""

import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import resize


def generate_phantom(size: int) -> np.ndarray:
    """
    Generate a normalized Shepp-Logan phantom of shape (size, size).

    Parameters
    ----------
    size : int
        Output image dimension (e.g. 128, 256, 512).

    Returns
    -------
    phantom : np.ndarray, float64 in [0, 1], shape (size, size)
    """
    phantom = shepp_logan_phantom()
    phantom = resize(phantom, (size, size), anti_aliasing=True)
    phantom = (phantom - phantom.min()) / (phantom.max() - phantom.min())
    return phantom
