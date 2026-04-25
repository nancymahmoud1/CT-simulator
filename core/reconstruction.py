"""
core/reconstruction.py
----------------------
CT image reconstruction algorithms: FBP and SART.
No Qt dependencies.
"""

import numpy as np
from skimage.transform import radon, iradon
from typing import Callable, Optional


def reconstruct_fbp(sinogram: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Filtered Back-Projection (FBP) reconstruction using a ramp filter.

    Parameters
    ----------
    sinogram : np.ndarray
        (Possibly noisy) sinogram.
    theta : np.ndarray
        Projection angles in degrees.

    Returns
    -------
    recon : np.ndarray
        Reconstructed image, values normalized to [0, 1].
    """
    recon = iradon(sinogram, theta=theta, filter_name='ramp', circle=True)
    recon = (recon - recon.min()) / (recon.max() - recon.min() + 1e-10)
    return recon


def reconstruct_sart(
    sinogram: np.ndarray,
    theta: np.ndarray,
    size: int,
    n_iters: int,
    progress_cb: Optional[Callable[[int, str], None]] = None,
) -> np.ndarray:
    """
    Simultaneous Algebraic Reconstruction Technique (SART).

    Parameters
    ----------
    sinogram : np.ndarray
        (Possibly noisy) sinogram.
    theta : np.ndarray
        Projection angles in degrees.
    size : int
        Output image side length (pixels).
    n_iters : int
        Number of SART iterations.
    progress_cb : callable(pct: int, msg: str) | None
        Optional callback to report per-iteration progress.

    Returns
    -------
    recon : np.ndarray
        Reconstructed image, values clipped to [0, 1].
    """
    n_angles = len(theta)
    recon = np.zeros((size, size))
    step = 0.2

    for i in range(n_iters):
        if progress_cb is not None:
            pct = 70 + int(20 * i / n_iters)
            progress_cb(pct, f"SART iteration {i + 1}/{n_iters}…")
        sino_est = radon(recon, theta=theta)
        residual = sinogram - sino_est
        update = iradon(residual, theta=theta, filter_name=None, circle=True)
        recon += step * update / (n_angles * 0.5)

    recon = np.clip(recon, 0, None)
    recon = (recon - recon.min()) / (recon.max() - recon.min() + 1e-10)
    return recon
