# Functional Sandbox for CT Physics


## Overview

This project replicates the core stages of CT acquisition and reconstruction in a library-driven Python environment. It serves as an educational tool for studying CT physics, particularly the trade-offs between radiation dose (mAs) and image quality.

### Pipeline Stages

```
Phantom → Forward Projection → Noise (Poisson) → Reconstruction → Metrics
```

1. **2D Phantom Generation** — Shepp-Logan head phantom (mathematically defined ground truth)
2. **Forward Projection** — Parallel-beam sinogram via Radon transform
3. **Dose-Noise Simulation** — Beer-Lambert + Poisson noise model at configurable mAs levels
4. **Image Reconstruction** — FBP (analytic) and SART (iterative algebraic)
5. **Evaluation** — RMSE, SSIM, SNR, Noise σ, CNR metrics

---

## Screenshots

### Phantom & Sinogram View
The left panel shows the Shepp-Logan phantom; the right shows its clean sinogram. The sidebar panel displays live RMSE/SSIM metrics and performance-vs-dose charts.

<img width="3840" height="2005" alt="screenshot_phantom_sinogram" src="https://github.com/user-attachments/assets/84639a6e-6237-4832-960a-62cd93118af1" />

---

### FBP vs SART Reconstruction
Side-by-side comparison of Filtered Back-Projection (FBP) and SART at 20 mAs / 180 angles. SART (RMSE: 0.1329, SSIM: 0.5910) achieves smoother results with fewer streak artifacts than FBP at this dose level.

<img width="3825" height="2009" alt="screenshot_reconstruction" src="https://github.com/user-attachments/assets/2f701866-0611-4333-821a-14f904716d52" />

---

### Noise & Dose — mAs Sweep Experiment
The Noise/Dose tab runs a full mAs sweep (5–500 mAs). The metrics table shows how Noise σ falls and SNR rises with increasing dose. The view shows the clean sinogram, a noisy sinogram at 5 mAs (SNR = 23.6 dB, σ = 0.1400), and the noise residual map.

<img width="3838" height="2004" alt="screenshot_noise_dose" src="https://github.com/user-attachments/assets/c3868bbc-d9e1-4b2f-97d9-71368aebb166" />

---

## Project Structure

```
ct-physics-sandbox/
├── main.py                        # Application entry point
│
├── core/
│   ├── phantom.py                 # Shepp-Logan phantom generation
│   ├── projection.py              # Forward projection (Radon / custom rotation)
│   ├── noise_model.py             # Poisson noise model + mAs sweep
│   ├── reconstruction.py          # FBP and SART reconstruction
│   └── metrics.py                 # RMSE and SSIM computation
│
├── pipeline/
│   ├── ct_pipeline.py             # Full end-to-end pipeline orchestrator
│   └── noise_experiment.py        # mAs sweep experiment runner
│
├── ui/
│   ├── ct_simulation_ui.py        # Main PyQt application window
│   └── noise_dose_tab.py          # Noise & Dose tab widget
│
└── demo_noise.py                  # Standalone noise module demo (no UI required)
```

---

## Installation

### Requirements

- Python 3.7+
- NumPy
- SciPy
- scikit-image
- PyQt5 (for the GUI)
- Matplotlib (for `demo_noise.py`)

### Install dependencies

```bash
pip install numpy scipy scikit-image PyQt5 matplotlib
```

---

## Usage

### Launch the full GUI

```bash
python main.py
```

### `core/reconstruction.py`
Two reconstruction algorithms:

| Method | Type | Speed | Low-dose performance |
|--------|------|-------|----------------------|
| **FBP** | Analytic (Ram-Lak ramp filter) | Very fast | Sensitive to noise |
| **SART** | Iterative algebraic | Slow | Robust at low mAs |


---

## Noise Physics

The noise model follows CT quantum noise physics:

```
I  = I0 × exp(−sinogram)          # Beer-Lambert (ideal photon count)
I* ~ Poisson(I)                    # quantum noise
sinogram_noisy = −log(I* / I0)    # back to projection domain
```

where `I0 = I0_per_mAs × mAs` (default: `I0_per_mAs = 1e4`).

Increasing mAs raises photon count → lower relative noise → better image quality.

---

## mAs Sweep Results

| mAs | I₀ photons | Noise σ | SNR (dB) |
|----:|----------:|--------:|---------:|
| 5   | 500       | 0.1400  | 23.60    |
| 10  | 1,000     | 0.0969  | 26.81    |
| 25  | 2,500     | 0.0618  | 30.71    |
| 50  | 5,000     | 0.0431  | 33.85    |
| 100 | 10,000    | 0.0307  | 36.80    |
| 200 | 20,000    | 0.0217  | 39.82    |
| 500 | 50,000    | 0.0137  | 43.79    |

*Values taken from the GUI at 128×128 resolution, seed = 42.*

---

## GUI Features

The PyQt5 application provides five tabs:

- **Phantom** — original Shepp-Logan phantom + clean sinogram
- **Sinogram** — clean projection data
- **Reconstruction** — side-by-side FBP vs SART output
- **Difference Map** — pixel-wise error visualization
- **Noise/Dose** — mAs sweep experiment with interactive metrics table and sinogram comparison

Configurable parameters:
- Image size (128 / 256 / 512)
- Number of projection angles (30–360)
- Detector elements
- mAs dose (1–100)
- Reconstruction method (FBP / SART) and SART iterations (1–100)

---

## Key Observations

- **FBP** is fast but noise-sensitive; streak artifacts emerge quickly at low mAs.
- **SART** preserves structure at low doses at the cost of computation time; quality improves with more iterations.
- More projection angles → smoother sinogram → fewer reconstruction artifacts.
- Higher mAs → more photons → reduced Poisson variance → lower RMSE and higher SSIM.


## Contributors

<div align="center">
  <table>
    <tr>
      <td align="center">
        <a href="https://github.com/nancymahmoud1">
          <img src="https://github.com/nancymahmoud1.png" width="80" style="border-radius:50%"/><br/>
          <sub><b>Nancy Mahmoud</b></sub>
        </a>
      </td>
      <td align="center">
        <a href="https://github.com/madonna-mosaad">
          <img src="https://github.com/madonna-mosaad.png" width="80" style="border-radius:50%"/><br/>
          <sub><b>Madonna Mosaad</b></sub>
        </a>
      </td>
      <td align="center">
        <a href="https://github.com/Emaaanabdelazeemm">
          <img src="https://github.com/Emaaanabdelazeemm.png" width="80" style="border-radius:50%"/><br/>
          <sub><b>Eman Abd Elazem</b></sub>
        </a>
      </td>
      <td align="center">
        <a href="https://github.com/shahdragab89">
          <img src="https://github.com/shahdragab89.png" width="80" style="border-radius:50%"/><br/>
          <sub><b>Shahd Ahmed Ragab</b></sub>
        </a>
      </td>
      <td align="center">
        <a href="https://github.com/hassnaa11">
          <img src="https://github.com/hassnaa11.png" width="80" style="border-radius:50%"/><br/>
          <sub><b>Hasnaa Hossam</b></sub>
        </a>
      </td>
    </tr>
  </table>
</div>
