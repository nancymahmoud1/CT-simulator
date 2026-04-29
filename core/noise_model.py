"""
core/noise_model.py
--------------------
Poisson noise model for CT dose simulation.

WHY WE NORMALISE AND STAY NORMALISED
-------------------------------------
Raw sinograms from column-sum projection have values 0-134+ 
(number of pixels * attenuation per pixel). 

Beer-Lambert:  I = I0 * exp(-sinogram)
exp(-134) = basically zero -> no photons at any dose -> no visible noise difference.

Fix: scale sinogram so max = 4.0 (real CT physics: mu * path_length in cm).
Then noise at 5 mAs is ~0.6 on a 0-4 scale = 15% of range = clearly visible.
At 500 mAs noise is ~0.06 = barely visible.

IMPORTANT: we return the noisy sinogram in NORMALISED units (0-4 range).
The clean sinogram passed in is also normalised before processing.
This means the display will show the correct visual difference.

Author: [Your Name] - Task: Noise & Dose Modeling + mAs Sweep
"""

import numpy as np

_TARGET_MAX = 4.0   # realistic CT max: mu(soft tissue)=0.2/cm * 20cm body = 4.0


def normalise_sinogram(sinogram: np.ndarray):
    """
    Scale sinogram so its max = _TARGET_MAX (4.0).
    Returns (sinogram_normalised, scale_factor).
    """
    raw_max = sinogram.max()
    if raw_max < 1e-12:
        return sinogram.copy(), 1.0
    scale = _TARGET_MAX / raw_max
    return sinogram * scale, scale


def add_poisson_noise(sinogram: np.ndarray,
                      mAs: float,
                      I0_per_mAs: float = 1e4,
                      seed: int = None) -> np.ndarray:
    """
    Apply Poisson noise to a sinogram at a given dose (mAs).

    INPUT:  sinogram can be raw (any scale) - we normalise internally.
    OUTPUT: noisy sinogram in NORMALISED units (max ~4.0).
            This is intentional - the noise is visible on this scale.

    Parameters
    ----------
    sinogram   : clean sinogram (n_angles, n_detectors)
    mAs        : dose in milli-Ampere-seconds. Higher = less noise.
    I0_per_mAs : photons per mAs. Default 1e4.
    seed       : random seed for reproducibility.
    """
    # Step 1: normalise to physical units (max = 4.0)
    sino_norm, _ = normalise_sinogram(sinogram)

    # Step 2: Beer-Lambert -> expected photon counts
    I0 = I0_per_mAs * mAs
    I  = I0 * np.exp(-sino_norm)

    # Step 3: Poisson noise
    rng     = np.random.default_rng(seed)
    I_noisy = rng.poisson(np.clip(I, 0, None)).astype(np.float64)

    # Step 4: back to sinogram in normalised units
    I_safe = np.clip(I_noisy, 1.0, None)
    sino_noisy_norm = -np.log(I_safe / I0)

    # Return in normalised units - DO NOT scale back to raw units
    # because raw units (0-134) make the noise invisible in the display
    return sino_noisy_norm


def get_clean_normalised(sinogram: np.ndarray) -> np.ndarray:
    """
    Return the clean sinogram in normalised units (same scale as noisy output).
    Call this to get the clean sinogram for display/comparison alongside noisy ones.
    """
    sino_norm, _ = normalise_sinogram(sinogram)
    return sino_norm


def mas_sweep(sinogram: np.ndarray,
              mas_levels: list,
              I0_per_mAs: float = 1e4,
              base_seed: int = 42) -> dict:
    """
    Run mAs sweep experiment.
    Returns dict {mAs: {sinogram_noisy (normalised), sinogram_clean_norm, mAs, I0}}
    """
    sino_norm, _ = normalise_sinogram(sinogram)
    results = {}
    for i, mAs in enumerate(mas_levels):
        results[mAs] = {
            'sinogram_noisy'     : add_poisson_noise(sinogram, mAs=mAs,
                                                      I0_per_mAs=I0_per_mAs,
                                                      seed=base_seed + i),
            'sinogram_clean_norm': sino_norm,   # for fair comparison
            'mAs': mAs,
            'I0' : I0_per_mAs * mAs,
        }
    return results


def noise_std(sinogram_noisy: np.ndarray, sinogram_clean: np.ndarray) -> float:
    """Std of noise residual. Both inputs must be in the same units."""
    return float(np.std(sinogram_noisy - sinogram_clean))


def signal_to_noise_ratio(sinogram_noisy: np.ndarray,
                           sinogram_clean: np.ndarray) -> float:
    """SNR in dB = 20*log10(RMS_signal / RMS_noise)."""
    rms_signal = np.sqrt(np.mean(sinogram_clean ** 2))
    rms_noise  = np.sqrt(np.mean((sinogram_noisy - sinogram_clean) ** 2))
    if rms_noise < 1e-12:
        return float('inf')
    return float(20 * np.log10(rms_signal / rms_noise))