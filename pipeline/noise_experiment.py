"""
pipeline/noise_experiment.py
-----------------------------
Orchestrates the mAs sweep experiment.

IMPORTANT: all sinograms in results are in NORMALISED units (max ~4.0).
This means the clean sinogram used for comparison is also normalised,
so the noise residual and metrics are computed fairly.

Author: [Your Name] - Task: Noise & Dose Modeling + mAs Sweep
"""

import numpy as np
from core.noise_model import mas_sweep, noise_std, signal_to_noise_ratio, get_clean_normalised

DEFAULT_MAS_LEVELS = [5, 10, 25, 50, 100, 200, 500]


def compute_rmse(img, ref):
    return float(np.sqrt(np.mean((img - ref) ** 2)))


def compute_ssim(img, ref, data_range=None):
    try:
        from skimage.metrics import structural_similarity as _ssim
        if data_range is None:
            data_range = float(ref.max() - ref.min())
        return float(_ssim(img, ref, data_range=data_range))
    except ImportError:
        mu1, mu2 = img.mean(), ref.mean()
        s1, s2   = img.std(), ref.std()
        s12 = float(np.mean((img - mu1) * (ref - mu2)))
        C1, C2   = 0.01**2, 0.03**2
        num = (2*mu1*mu2 + C1) * (2*s12 + C2)
        den = (mu1**2 + mu2**2 + C1) * (s1**2 + s2**2 + C2)
        return float(num / den)


def _default_masks(shape):
    H, W = shape
    roi  = np.zeros(shape, bool)
    bg   = np.zeros(shape, bool)
    cy, cx = H//2, W//2
    r = min(H, W) // 6
    y, x = np.ogrid[:H, :W]
    roi[(y-cy)**2 + (x-cx)**2 <= r**2] = True
    bg[:H//8, :W//8] = bg[:H//8, -W//8:] = True
    bg[-H//8:, :W//8] = bg[-H//8:, -W//8:] = True
    return roi, bg


def run_mas_sweep_experiment(sinogram_clean: np.ndarray,
                              phantom: np.ndarray = None,
                              reconstruct_fn=None,
                              mas_levels: list = None,
                              I0_per_mAs: float = 1e4,
                              roi_mask=None,
                              bg_mask=None,
                              base_seed: int = 42) -> dict:
    """
    Full mAs sweep.

    sinogram_clean : raw sinogram (any scale) - normalised internally
    phantom        : ground truth image for RMSE/SSIM (optional)
    reconstruct_fn : function(sinogram_norm) -> image (optional)
    mas_levels     : list of dose values to test
    """
    if mas_levels is None:
        mas_levels = DEFAULT_MAS_LEVELS

    # Run sweep - all noisy sinograms come back in normalised units
    sweep = mas_sweep(sinogram_clean, mas_levels,
                      I0_per_mAs=I0_per_mAs, base_seed=base_seed)

    # Clean sinogram in normalised units (for fair comparison)
    sino_clean_norm = get_clean_normalised(sinogram_clean)

    sinograms_noisy = {mAs: sweep[mAs]['sinogram_noisy'] for mAs in mas_levels}

    # Reconstruct if function provided
    images_recon = {}
    if reconstruct_fn is not None:
        for mAs in mas_levels:
            images_recon[mAs] = reconstruct_fn(sinograms_noisy[mAs])

    # Compute metrics
    metrics = {}
    for mAs in mas_levels:
        m = {}
        sino_noisy = sinograms_noisy[mAs]

        # Compare noisy vs clean - both in normalised units
        m['noise_std'] = noise_std(sino_noisy, sino_clean_norm)
        m['snr_dB']    = signal_to_noise_ratio(sino_noisy, sino_clean_norm)

        if mAs in images_recon and phantom is not None:
            img = images_recon[mAs]
            ref = phantom
            img_n = (img - img.min()) / (img.max() - img.min() + 1e-12)
            ref_n = (ref - ref.min()) / (ref.max() - ref.min() + 1e-12)
            m['rmse'] = compute_rmse(img_n, ref_n)
            m['ssim'] = compute_ssim(img_n, ref_n, data_range=1.0)
            roi, bg   = (roi_mask, bg_mask) if (roi_mask is not None) else _default_masks(img.shape)
            std_bg    = float(np.std(img_n[bg]))
            m['cnr']  = float(abs(img_n[roi].mean() - img_n[bg].mean()) / (std_bg + 1e-12))
        else:
            m['rmse'] = m['ssim'] = m['cnr'] = None

        metrics[mAs] = m

    return {
        'mas_levels'       : mas_levels,
        'sinograms'        : sinograms_noisy,         # normalised units
        'sinogram_clean'   : sino_clean_norm,          # normalised, for display
        'images'           : images_recon,
        'metrics'          : metrics,
        'summary'          : {
            'n_levels'         : len(mas_levels),
            'I0_per_mAs'       : I0_per_mAs,
            'has_reconstruction': reconstruct_fn is not None,
            'has_ground_truth'  : phantom is not None,
        }
    }