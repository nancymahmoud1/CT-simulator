"""
core/noise.py
-------------
Poisson quantum noise modeling for CT dose simulation.
No Qt dependencies.
"""

import numpy as np


def add_poisson_noise(sinogram_clean: np.ndarray, mas: float) -> np.ndarray:
    """
    Apply Poisson photon-count noise to a clean sinogram.

    The model:
        I0      = scale * ones            (incident photon count)
        I_trans = I0 * exp(-sinogram)     (Beer-Lambert attenuation)
        I_noisy ~ Poisson(I_trans)        (quantum noise)
        sinogram_noisy = -log(I_noisy / I0)

    Parameters
    ----------
    sinogram_clean : np.ndarray
        Noiseless sinogram (line integrals of attenuation coefficients).
    mas : float
        mAs value (1–100). Higher = more photons = less noise.

    Returns
    -------
    sinogram_noisy : np.ndarray
        Same shape as input, with Poisson noise applied.
    """
    scale = 1e5 * (mas / 100.0)
    I0 = scale * np.ones_like(sinogram_clean)
    I_transmitted = I0 * np.exp(-sinogram_clean)
    I_noisy = np.random.poisson(np.maximum(I_transmitted, 1e-6).astype(float))
    I_noisy = np.maximum(I_noisy, 1).astype(float)
    sinogram_noisy = -np.log(I_noisy / I0)
    sinogram_noisy = np.clip(sinogram_noisy, 0, None)
    return sinogram_noisy
