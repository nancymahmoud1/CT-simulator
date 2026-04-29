"""
pipeline/ct_pipeline.py
-----------------------
Orchestrates the full CT simulation pipeline:
    Phantom → Sinogram → Noise → Reconstruction → Metrics → mAs Sweep

This module has no Qt dependencies and can be used standalone or called
from the SimulationWorker thread in the UI.
"""

import numpy as np
from typing import Callable, Optional

from core.phantom import generate_phantom
from core.projection import compute_sinogram
from core.noise_model import add_poisson_noise
from core.reconstruction import reconstruct_fbp, reconstruct_sart
from core.metrics import compute_rmse, compute_ssim


def run_pipeline(
    params: dict,
    progress_cb: Optional[Callable[[int, str], None]] = None,
) -> dict:
    """
    Execute the complete CT simulation and return all results.

    Parameters
    ----------
    params : dict
        Keys:
            image_size      (int)   – phantom / reconstruction resolution
            n_angles        (int)   – number of projection angles
            n_detectors     (int)   – detector elements (informational; radon auto-sizes)
            mas             (float) – mAs dose value (1–100)
            add_noise       (bool)  – whether to apply Poisson noise
            method          (str)   – "FBP" or "SART"
            sart_iterations (int)   – iterations used when method == "SART"
    progress_cb : callable(pct: int, msg: str) | None
        Optional progress reporter (forwarded to SART inner loop).

    Returns
    -------
    results : dict
        Contains all arrays and scalars needed by the UI to render every tab
        and chart.  Keys match what CTSimApp._update_displays() expects.
    """

    def _progress(pct, msg):
        if progress_cb:
            progress_cb(pct, msg)

    size      = params["image_size"]
    n_angles  = params["n_angles"]
    mas       = params["mas"]
    add_noise = params["add_noise"]
    method    = params["method"]
    sart_iters = params["sart_iterations"]

    # ── 1. Phantom ──────────────────────────────────────────────────────────
    _progress(10, "Generating phantom…")
    phantom = generate_phantom(size)

    # ── 2. Sinogram ─────────────────────────────────────────────────────────
    _progress(30, "Computing sinogram…")
    theta = np.linspace(0, 180, n_angles, endpoint=False)
    sinogram_clean = compute_sinogram(phantom, theta)

    # ── 3. Noise ────────────────────────────────────────────────────────────
    _progress(50, "Adding noise…")
    sinogram_noisy = add_poisson_noise(sinogram_clean, mas) if add_noise else sinogram_clean.copy()

    # ── 4. Reconstruction ───────────────────────────────────────────────────
    _progress(70, f"Reconstructing ({method})…")
    if method == "FBP":
        recon = reconstruct_fbp(sinogram_noisy, theta)
    else:
        recon = reconstruct_sart(sinogram_noisy, theta, size, sart_iters, progress_cb)

    # ── 5. FBP baseline (for comparison in the bar chart) ───────────────────
    recon_fbp = reconstruct_fbp(sinogram_noisy, theta)

    # ── 6. Metrics ──────────────────────────────────────────────────────────
    _progress(90, "Computing metrics…")
    diff      = phantom - recon
    rmse      = compute_rmse(phantom, recon)
    ssim_val  = compute_ssim(phantom, recon)
    rmse_fbp  = compute_rmse(phantom, recon_fbp)
    ssim_fbp  = compute_ssim(phantom, recon_fbp)

    # ── 7. mAs sweep ────────────────────────────────────────────────────────
    mas_vals = [1, 2, 5, 10, 20, 50, 100]
    rmse_sweep, ssim_sweep = [], []
    rmse_sart_sweep, ssim_sart_sweep = [], []

    for m in mas_vals:
        sg = add_poisson_noise(sinogram_clean, m)

        r_fbp = reconstruct_fbp(sg, theta)
        rmse_sweep.append(compute_rmse(phantom, r_fbp))
        ssim_sweep.append(compute_ssim(phantom, r_fbp))

        # Quick SART (5 iters) for the sweep — full quality not required
        r_s = reconstruct_sart(sg, theta, size, n_iters=5)
        rmse_sart_sweep.append(compute_rmse(phantom, r_s))
        ssim_sart_sweep.append(compute_ssim(phantom, r_s))

    _progress(100, "Done!")

    return {
        # Images
        "phantom":          phantom,
        "sinogram_clean":   sinogram_clean,
        "sinogram_noisy":   sinogram_noisy,
        "theta":            theta,
        "recon":            recon,
        "recon_fbp":        recon_fbp,
        "diff":             diff,
        # Scalar metrics
        "rmse":             rmse,
        "ssim":             ssim_val,
        "rmse_fbp":         rmse_fbp,
        "ssim_fbp":         ssim_fbp,
        # Sweep data
        "mas_vals":         mas_vals,
        "rmse_sweep":       rmse_sweep,
        "ssim_sweep":       ssim_sweep,
        "rmse_sart_sweep":  rmse_sart_sweep,
        "ssim_sart_sweep":  ssim_sart_sweep,
        # Passthrough params (used by display layer)
        "method":           method,
        "mas":              mas,
        "n_angles":         n_angles,
        "n_detectors":      params["n_detectors"],
    }
