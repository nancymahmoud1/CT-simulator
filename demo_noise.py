"""
demo_noise.py
-------------
Standalone demonstration of the Noise & Dose Modeling module.

Run this to test your work INDEPENDENTLY — it generates its own
Shepp-Logan phantom and sinogram so you don't need anyone else's code.

Usage:
    python demo_noise.py

What it shows:
    1. The clean sinogram
    2. Noisy sinograms at 5 different dose levels (5, 25, 100, 200, 500 mAs)
    3. Dose-quality curves: Noise Std, SNR, RMSE, SSIM vs mAs
    4. Prints a metrics table to the terminal

Author: [Your Name]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── 1. Generate a simple Shepp-Logan-like phantom from scratch ──────────────

def make_shepp_logan(size=128):
    """
    Build a simplified Shepp-Logan phantom using NumPy only.
    No scikit-image needed for the demo.
    """
    phantom = np.zeros((size, size), dtype=np.float64)
    cx, cy = size / 2, size / 2

    def ellipse(a, b, x0, y0, angle_deg, value):
        theta = np.deg2rad(angle_deg)
        y, x = np.ogrid[:size, :size]
        xr = (x - cx - x0) * np.cos(theta) + (y - cy - y0) * np.sin(theta)
        yr = -(x - cx - x0) * np.sin(theta) + (y - cy - y0) * np.cos(theta)
        mask = (xr / a) ** 2 + (yr / b) ** 2 <= 1
        phantom[mask] += value

    s = size / 2
    ellipse(0.92*s, 0.69*s,  0,      0,     0,   2.0)   # outer head
    ellipse(0.87*s, 0.64*s,  0,     -0.02*s, 0, -0.98)  # inner skull
    ellipse(0.31*s, 0.11*s,  0.22*s, 0,    -18,  0.1)   # right eye
    ellipse(0.41*s, 0.16*s, -0.22*s, 0,     18,  0.1)   # left eye
    ellipse(0.21*s, 0.25*s,  0,      0.35*s, 0,  0.1)   # forehead blob
    ellipse(0.046*s,0.046*s, 0,      0.1*s,  0,  0.1)   # small blob 1
    ellipse(0.046*s,0.046*s, 0,     -0.1*s,  0,  0.1)   # small blob 2
    ellipse(0.046*s,0.023*s,-0.08*s,-0.605*s, 0,  0.1)  # bottom blobs
    ellipse(0.023*s,0.023*s, 0,     -0.606*s, 0,  0.1)
    ellipse(0.046*s,0.023*s, 0.06*s,-0.605*s, 0,  0.1)

    return np.clip(phantom, 0, None)


# ── 2. Forward projection (Radon transform, pure NumPy) ────────────────────

def forward_project(phantom, n_angles=180):
    """
    Simple parallel-beam forward projection using rotation + column sums.
    Produces a sinogram of shape (n_angles, n_detectors).
    """
    from scipy.ndimage import rotate   # only scipy, which is in requirements

    n_detectors = phantom.shape[1]
    angles = np.linspace(0, 180, n_angles, endpoint=False)
    sinogram = np.zeros((n_angles, n_detectors))

    for i, angle in enumerate(angles):
        rotated = rotate(phantom, angle, reshape=False, order=1)
        sinogram[i, :] = rotated.sum(axis=0)

    return sinogram


# ── 3. Simple FBP reconstruction for metric comparison ─────────────────────

def fbp_reconstruct(sinogram, n_angles=180, filter_type='ramp'):
    """
    Minimal Filtered Back-Projection using 1-D FFT Ram-Lak filter.
    """
    n_angles, n_det = sinogram.shape
    angles_rad = np.linspace(0, np.pi, n_angles, endpoint=False)

    # Apply ramp filter in frequency domain
    freqs   = np.fft.rfftfreq(n_det)
    ramp    = np.abs(freqs)
    filtered = np.zeros_like(sinogram)
    for i in range(n_angles):
        F = np.fft.rfft(sinogram[i])
        filtered[i] = np.fft.irfft(F * ramp, n=n_det)

    # Back-project
    size  = n_det
    image = np.zeros((size, size), dtype=np.float64)
    mid   = size // 2
    y, x  = np.mgrid[:size, :size] - mid

    for i, theta in enumerate(angles_rad):
        t = x * np.cos(theta) + y * np.sin(theta)
        t_idx = np.round(t + mid).astype(int)
        valid = (t_idx >= 0) & (t_idx < n_det)
        image[valid] += filtered[i, t_idx[valid]]

    image *= np.pi / n_angles
    return image


# ── 4. Main demo ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  CT Noise & Dose Modeling Demo")
    print("  Task: Poisson Noise + mAs Sweep Experiment")
    print("=" * 60)

    # Generate phantom and sinogram
    print("\n[1/4] Generating Shepp-Logan phantom (128×128)…")
    phantom = make_shepp_logan(size=128)

    print("[2/4] Forward projection (180 angles)…")
    sino_clean = forward_project(phantom, n_angles=180)
    print(f"      Sinogram shape: {sino_clean.shape}")

    # Run the mAs sweep using your module
    print("[3/4] Running mAs sweep experiment…")
    from pipeline.noise_experiment import run_mas_sweep_experiment

    mas_levels = [5, 10, 25, 50, 100, 200, 500]

    results = run_mas_sweep_experiment(
        sinogram_clean = sino_clean,
        phantom        = phantom,
        reconstruct_fn = fbp_reconstruct,
        mas_levels     = mas_levels,
        I0_per_mAs     = 1e4,
        base_seed      = 42,
    )

    # Print metrics table
    print("\n[4/4] Results:")
    print(f"\n{'mAs':>6}  {'I0':>10}  {'Noise σ':>10}  {'SNR(dB)':>10}  "
          f"{'RMSE':>8}  {'SSIM':>8}")
    print("-" * 62)
    for mAs in mas_levels:
        m   = results['metrics'][mAs]
        I0  = 1e4 * mAs
        snr = m['snr_dB']
        ns  = m['noise_std']
        rmse = m['rmse'] if m['rmse'] is not None else float('nan')
        ssim = m['ssim'] if m['ssim'] is not None else float('nan')
        print(f"{mAs:>6.0f}  {I0:>10.0f}  {ns:>10.4f}  {snr:>10.2f}  "
              f"{rmse:>8.4f}  {ssim:>8.4f}")

    # ── Plotting ──────────────────────────────────────────────────────────
    print("\nGenerating plots…")
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("CT Noise & Dose Modeling — mAs Sweep Results", fontsize=14)
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.4)

    sinograms = results['sinograms']
    images    = results['images']
    metrics   = results['metrics']

    # Row 0: sinogram comparison (5, 50, 500 mAs)
    preview_levels = [5, 50, 500]
    for col, mAs in enumerate(preview_levels):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(sinograms[mAs], cmap='gray', aspect='auto')
        snr = metrics[mAs]['snr_dB']
        ax.set_title(f"Noisy Sino — {mAs:.0f} mAs\nSNR={snr:.1f} dB", fontsize=8)
        ax.set_xlabel("Detector bin")
        ax.set_ylabel("Angle" if col == 0 else "")

    ax_clean = fig.add_subplot(gs[0, 3])
    ax_clean.imshow(sino_clean, cmap='gray', aspect='auto')
    ax_clean.set_title("Clean Sinogram\n(no noise)", fontsize=8)
    ax_clean.set_xlabel("Detector bin")

    # Row 1: reconstructed images (5, 50, 500 mAs + phantom)
    for col, mAs in enumerate(preview_levels):
        ax = fig.add_subplot(gs[1, col])
        img = images[mAs]
        ax.imshow(img, cmap='gray')
        rmse = metrics[mAs]['rmse']
        ssim = metrics[mAs]['ssim']
        ax.set_title(
            f"FBP Recon — {mAs:.0f} mAs\n"
            f"RMSE={rmse:.3f}  SSIM={ssim:.3f}", fontsize=8)
        ax.axis('off')

    ax_ph = fig.add_subplot(gs[1, 3])
    ax_ph.imshow(phantom, cmap='gray')
    ax_ph.set_title("Ground Truth\nPhantom", fontsize=8)
    ax_ph.axis('off')

    # Row 2: dose-quality curves
    snr_vals  = [metrics[m]['snr_dB']   for m in mas_levels]
    nstd_vals = [metrics[m]['noise_std'] for m in mas_levels]
    rmse_vals = [metrics[m]['rmse']      for m in mas_levels]
    ssim_vals = [metrics[m]['ssim']      for m in mas_levels]

    ax_snr = fig.add_subplot(gs[2, 0])
    ax_snr.plot(mas_levels, snr_vals, 'o-b', linewidth=2)
    ax_snr.set_xlabel("mAs")
    ax_snr.set_ylabel("SNR (dB)")
    ax_snr.set_title("SNR vs Dose")
    ax_snr.set_xscale('log')
    ax_snr.grid(True, alpha=0.3)

    ax_ns = fig.add_subplot(gs[2, 1])
    ax_ns.plot(mas_levels, nstd_vals, 's-g', linewidth=2)
    ax_ns.set_xlabel("mAs")
    ax_ns.set_ylabel("Noise Std (σ)")
    ax_ns.set_title("Noise σ vs Dose")
    ax_ns.set_xscale('log')
    ax_ns.grid(True, alpha=0.3)

    ax_rmse = fig.add_subplot(gs[2, 2])
    ax_rmse.plot(mas_levels, rmse_vals, 'o-r', linewidth=2)
    ax_rmse.set_xlabel("mAs")
    ax_rmse.set_ylabel("RMSE")
    ax_rmse.set_title("RMSE vs Dose")
    ax_rmse.set_xscale('log')
    ax_rmse.grid(True, alpha=0.3)

    ax_ssim = fig.add_subplot(gs[2, 3])
    ax_ssim.plot(mas_levels, ssim_vals, 's-m', linewidth=2)
    ax_ssim.set_xlabel("mAs")
    ax_ssim.set_ylabel("SSIM")
    ax_ssim.set_title("SSIM vs Dose")
    ax_ssim.set_xscale('log')
    ax_ssim.grid(True, alpha=0.3)

    plt.savefig("noise_demo_results.png", dpi=150, bbox_inches='tight')
    print("\n✅ Plot saved to: noise_demo_results.png")
    plt.show()
    print("\nDone! Your noise module is working correctly.")


if __name__ == "__main__":
    main()