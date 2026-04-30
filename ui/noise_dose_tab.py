"""
ui/noise_dose_tab.py
---------------------
Noise & Dose Modeling Tab — works in TWO ways:

  WAY 1 (automatic): Run the main simulation → your tab gets the sinogram
                     automatically. The green status message appears.

  WAY 2 (standalone): Click "Generate Test Sinogram" inside this tab.
                      It builds its own Shepp-Logan phantom + sinogram
                      from scratch using only NumPy/SciPy — no need to
                      run the main simulation at all.

Author: [Your Name] — Task: Poisson Noise Modeling + mAs Sweep
"""

import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFrame, QSizePolicy, QTableWidget, QTableWidgetItem,
    QComboBox, QHeaderView, QSplitter
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec

from pipeline.noise_experiment import run_mas_sweep_experiment


# ─────────────────────────────────────────────────────────────────────────────
#  BUILT-IN PHANTOM + SINOGRAM GENERATOR (pure NumPy/SciPy, from scratch)
#  This is what runs when you click "Generate Test Sinogram"
# ─────────────────────────────────────────────────────────────────────────────

class _PhantomWorker(QThread):
    """Generates phantom + sinogram on a background thread so UI doesn't freeze."""
    finished = pyqtSignal(object, object)   # (sinogram, phantom)
    error    = pyqtSignal(str)

    def __init__(self, size=128, n_angles=180):
        super().__init__()
        self._size     = size
        self._n_angles = n_angles

    def run(self):
        try:
            phantom  = _make_phantom(self._size)
            sinogram = _forward_project(phantom, self._n_angles)
            self.finished.emit(sinogram, phantom)
        except Exception as e:
            import traceback
            self.error.emit(traceback.format_exc())


def _make_phantom(size=128):
    """
    Build a Shepp-Logan phantom entirely in NumPy — no scikit-image needed.
    Uses overlapping ellipses with standard Shepp-Logan parameters.
    """
    phantom = np.zeros((size, size), dtype=np.float64)
    cx, cy  = size / 2.0, size / 2.0
    s       = size / 2.0   # half-size scale

    # (a, b, x0, y0, angle_deg, value)
    ellipses = [
        (0.92*s, 0.69*s,  0.00*s,  0.00*s,   0,  2.00),
        (0.87*s, 0.64*s,  0.00*s, -0.02*s,   0, -0.98),
        (0.31*s, 0.11*s,  0.22*s,  0.00*s, -18,  0.10),
        (0.41*s, 0.16*s, -0.22*s,  0.00*s,  18,  0.10),
        (0.21*s, 0.25*s,  0.00*s,  0.35*s,   0,  0.10),
        (0.046*s,0.046*s, 0.00*s,  0.10*s,   0,  0.10),
        (0.046*s,0.046*s, 0.00*s, -0.10*s,   0,  0.10),
        (0.046*s,0.023*s,-0.08*s, -0.61*s,   0,  0.10),
        (0.023*s,0.023*s, 0.00*s, -0.61*s,   0,  0.10),
        (0.046*s,0.023*s, 0.06*s, -0.61*s,   0,  0.10),
    ]

    y_idx, x_idx = np.ogrid[:size, :size]
    for (a, b, x0, y0, angle_deg, value) in ellipses:
        theta = np.deg2rad(angle_deg)
        dx    = (x_idx - cx - x0)
        dy    = (y_idx - cy - y0)
        xr    =  dx * np.cos(theta) + dy * np.sin(theta)
        yr    = -dx * np.sin(theta) + dy * np.cos(theta)
        mask  = (xr / a)**2 + (yr / b)**2 <= 1.0
        phantom[mask] += value

    return np.clip(phantom, 0, None)


def _forward_project(phantom, n_angles=180):
    """
    Parallel-beam forward projection (discrete Radon transform).
    Rotates the phantom for each angle and sums columns → sinogram.
    Pure SciPy — built from scratch as required by the proposal.
    """
    from scipy.ndimage import rotate as _rotate
    n_det    = phantom.shape[1]
    angles   = np.linspace(0, 180, n_angles, endpoint=False)
    sinogram = np.zeros((n_angles, n_det), dtype=np.float64)
    for i, angle in enumerate(angles):
        rotated       = _rotate(phantom, angle, reshape=False, order=1)
        sinogram[i]   = rotated.sum(axis=0)
    return sinogram


# ─────────────────────────────────────────────────────────────────────────────
#  Plot / display helpers — dark theme matching ct_simulation_ui.py
# ─────────────────────────────────────────────────────────────────────────────

def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor('#0d1117')
    ax.set_title(title, color='#c9d1d9', fontsize=10, pad=5)
    ax.tick_params(colors='#8b949e', labelsize=7)
    for s in ax.spines.values():
        s.set_edgecolor('#30363d')
    if xlabel:
        ax.set_xlabel(xlabel, color='#8b949e', fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color='#8b949e', fontsize=8)


def _imshow(ax, data, title="", cmap="gray", xlabel="", ylabel=""):
    ax.set_facecolor('#0d1117')
    im = ax.imshow(data, cmap=cmap, aspect='auto',
                   interpolation='nearest',
                   vmin=data.min(), vmax=data.max())
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.ax.yaxis.set_tick_params(color='#8b949e', labelcolor='#8b949e', labelsize=7)
    cb.outline.set_edgecolor('#30363d')
    _style_ax(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    return im


def _line(ax, x, y, color, marker, label,
          title="", xlabel="", ylabel="", highlight_x=None):
    ax.semilogx(x, y, marker + '-', color=color, lw=2, ms=5, label=label)
    if highlight_x is not None:
        ax.axvline(highlight_x, color='#f85149', lw=1.2,
                   linestyle='--', alpha=0.7)
    ax.set_facecolor('#161b22')
    _style_ax(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(True, color='#30363d', linewidth=0.6, alpha=0.7)
    ax.legend(fontsize=7, facecolor='#161b22',
              edgecolor='#30363d', labelcolor='#c9d1d9')


# ─────────────────────────────────────────────────────────────────────────────
#  Matplotlib canvas
# ─────────────────────────────────────────────────────────────────────────────

class _Canvas(FigureCanvas):
    def __init__(self, w=9, h=5, dpi=90):
        self.fig = Figure(figsize=(w, h), dpi=dpi, facecolor='#0d1117')
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color:#0d1117;")


# ─────────────────────────────────────────────────────────────────────────────
#  Background sweep worker
# ─────────────────────────────────────────────────────────────────────────────

class _SweepWorker(QThread):
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, sinogram, phantom, mas_levels, I0_per_mAs, seed):
        super().__init__()
        self._sino   = sinogram
        self._ph     = phantom
        self._levels = mas_levels
        self._i0     = I0_per_mAs
        self._seed   = seed

    def run(self):
        try:
            res = run_mas_sweep_experiment(
                sinogram_clean = self._sino,
                phantom        = self._ph,
                reconstruct_fn = None,   # sinogram-domain metrics only
                mas_levels     = self._levels,
                I0_per_mAs     = self._i0,
                base_seed      = self._seed,
            )
            self.finished.emit(res)
        except Exception as e:
            import traceback
            self.error.emit(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
#  Input field style (shared)
# ─────────────────────────────────────────────────────────────────────────────

_INPUT = ("background-color:#161b22;border:1px solid #30363d;"
          "border-radius:4px;padding:4px 8px;"
          "color:#e6edf3;font-size:11px;")

_LABEL = "color:#8b949e;font-size:11px;"


# ─────────────────────────────────────────────────────────────────────────────
#  THE MAIN TAB WIDGET
# ─────────────────────────────────────────────────────────────────────────────

class NoiseDoseTab(QWidget):
    """
    Noise & Dose Modeling tab.

    Works in two modes:
      - Standalone : click "Generate Test Sinogram" — no main simulation needed
      - Integrated : ct_simulation_ui calls set_sinogram() after pipeline runs
    """

    def __init__(self, parent=None, reconstruct_fn=None):
        super().__init__(parent)
        self._sinogram        = None
        self._phantom         = None
        self._results         = None
        self._reconstruct_fn  = reconstruct_fn
        self._sweep_worker    = None
        self._phantom_worker  = None
        self._preview_idx     = 0
        self._build_ui()

    # ── PUBLIC API ──────────────────────────────────────────────────────────

    def set_sinogram(self, sinogram: np.ndarray, phantom: np.ndarray = None):
        """Called automatically by ct_simulation_ui after the pipeline runs."""
        self._sinogram = sinogram
        self._phantom  = phantom
        self._run_btn.setEnabled(True)
        h, w = sinogram.shape
        self._set_status(
            f"✅  Sinogram ready from main simulation — "
            f"{h} angles × {w} detectors. "
            f"Now click 'Run mAs Sweep'.",
            color='#3fb950')

    def set_reconstruct_fn(self, fn):
        self._reconstruct_fn = fn

    # ── BUILD UI ────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # ── Title ──────────────────────────────────────────────────────────
        title_row = QHBoxLayout()
        icon  = QLabel("📉")
        icon.setStyleSheet("font-size:16px;")
        title = QLabel("Noise & Dose Modeling  ·  mAs Sweep Experiment")
        title.setStyleSheet(
            "color:#58a6ff;font-size:13px;font-weight:700;letter-spacing:0.5px;")
        title_row.addWidget(icon)
        title_row.addWidget(title)
        title_row.addStretch()
        root.addLayout(title_row)

        # ── Status bar ─────────────────────────────────────────────────────
        self._status = QLabel(
            "⚠️  No sinogram loaded. Either run the main simulation OR "
            "click 'Generate Test Sinogram' below to start immediately.")
        self._status.setStyleSheet("color:#8b949e;font-size:11px;")
        self._status.setWordWrap(True)
        root.addWidget(self._status)

        div = QFrame(); div.setObjectName("divider")
        root.addWidget(div)

        # ── STANDALONE SECTION ─────────────────────────────────────────────
        # This is the key part — lets you use the tab without the main sim
        standalone_row = QHBoxLayout()
        standalone_row.setSpacing(10)

        standalone_lbl = QLabel("Don't have a sinogram yet?")
        standalone_lbl.setStyleSheet("color:#8b949e;font-size:11px;")
        standalone_row.addWidget(standalone_lbl)

        # Phantom size picker
        self._size_combo = QComboBox()
        self._size_combo.addItems(["64 × 64  (fast)", "128 × 128", "256 × 256"])
        self._size_combo.setCurrentIndex(1)
        self._size_combo.setFixedWidth(130)
        self._size_combo.setStyleSheet(
            "background:#161b22;border:1px solid #30363d;border-radius:4px;"
            "color:#e6edf3;font-size:11px;padding:3px 8px;")
        standalone_row.addWidget(self._size_combo)

        self._gen_btn = QPushButton("🔬  Generate Test Sinogram")
        self._gen_btn.setObjectName("secondary_btn")
        self._gen_btn.setMinimumHeight(32)
        self._gen_btn.setStyleSheet(
            "QPushButton{background:#21262d;color:#58a6ff;"
            "border:1px solid #30363d;border-radius:6px;"
            "font-size:11px;padding:0 14px;}"
            "QPushButton:hover{background:#30363d;color:#79c0ff;"
            "border-color:#58a6ff;}"
            "QPushButton:disabled{color:#484f58;border-color:#21262d;}")
        self._gen_btn.clicked.connect(self._on_generate)
        standalone_row.addWidget(self._gen_btn)

        standalone_row.addStretch()
        root.addLayout(standalone_row)

        div2 = QFrame(); div2.setObjectName("divider")
        root.addWidget(div2)

        # ── SWEEP PARAMETERS ───────────────────────────────────────────────
        params_row = QHBoxLayout()
        params_row.setSpacing(14)

        params_row.addWidget(self._lbl("I₀ per mAs:"))
        self._i0_edit = QLineEdit("100")
        self._i0_edit.setFixedWidth(80)
        self._i0_edit.setStyleSheet(_INPUT)
        self._i0_edit.setToolTip(
            "Photons per mAs.\n"
            "Higher = brighter detector = less noise at same dose.\n"
            "Default 10000 is realistic for most CT scanners.")
        params_row.addWidget(self._i0_edit)

        params_row.addSpacing(8)
        params_row.addWidget(self._lbl("mAs levels:"))
        self._mas_edit = QLineEdit("5, 10, 25, 50, 100, 200, 500")
        self._mas_edit.setMinimumWidth(220)
        self._mas_edit.setStyleSheet(_INPUT)
        self._mas_edit.setToolTip(
            "Comma-separated dose levels to test.\n"
            "5 mAs = very noisy,   500 mAs = near noise-free.")
        params_row.addWidget(self._mas_edit)

        params_row.addSpacing(8)
        params_row.addWidget(self._lbl("Seed:"))
        self._seed_edit = QLineEdit("42")
        self._seed_edit.setFixedWidth(45)
        self._seed_edit.setStyleSheet(_INPUT)
        self._seed_edit.setToolTip("Random seed (any integer). Same seed = same noise.")
        params_row.addWidget(self._seed_edit)

        params_row.addStretch()

        self._run_btn = QPushButton("▶  Run mAs Sweep")
        self._run_btn.setObjectName("run_btn")
        self._run_btn.setEnabled(False)
        self._run_btn.setMinimumHeight(36)
        self._run_btn.setMinimumWidth(160)
        self._run_btn.clicked.connect(self._on_run_sweep)
        params_row.addWidget(self._run_btn)

        root.addLayout(params_row)

        # ── MAIN CONTENT: plots + table ────────────────────────────────────
        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet(
            "QSplitter::handle{background:#30363d;width:2px;}")

        # Left — canvas + view selector + dose preview
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        view_row = QHBoxLayout()
        view_row.addWidget(self._lbl("View:"))
        self._view_combo = QComboBox()
        self._view_combo.addItems([
            "Sinogram Comparison",
            "Dose-Quality Curves",
        ])
        self._view_combo.setFixedWidth(200)
        self._view_combo.setStyleSheet(
            "background:#161b22;border:1px solid #30363d;border-radius:4px;"
            "color:#e6edf3;font-size:11px;padding:3px 8px;")
        self._view_combo.currentIndexChanged.connect(self._refresh_plots)
        view_row.addWidget(self._view_combo)
        view_row.addStretch()
        left_layout.addLayout(view_row)

        self._canvas = _Canvas(w=9, h=5, dpi=90)
        left_layout.addWidget(self._canvas)

        dose_row = QHBoxLayout()
        dose_row.addWidget(self._lbl("Preview dose:"))
        self._dose_combo = QComboBox()
        self._dose_combo.setFixedWidth(110)
        self._dose_combo.setStyleSheet(
            "background:#161b22;border:1px solid #30363d;border-radius:4px;"
            "color:#e6edf3;font-size:11px;padding:3px 8px;")
        self._dose_combo.currentIndexChanged.connect(self._on_dose_changed)
        dose_row.addWidget(self._dose_combo)
        dose_row.addStretch()
        left_layout.addLayout(dose_row)

        splitter.addWidget(left)

        # Right — metrics table + tip
        right = QWidget()
        right.setFixedWidth(340)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 0, 0, 0)
        right_layout.setSpacing(6)

        hdr = QLabel("📊  Metrics per Dose Level")
        hdr.setStyleSheet(
            "color:#58a6ff;font-size:11px;font-weight:600;letter-spacing:0.5px;")
        right_layout.addWidget(hdr)

        self._table = QTableWidget()
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels(
            ["mAs", "Noise σ", "SNR (dB)", "I₀ photons"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setStyleSheet("""
            QTableWidget {
                background-color:#0d1117;
                color:#c9d1d9;
                font-size:11px;
                border:1px solid #30363d;
                gridline-color:#21262d;
            }
            QTableWidget::item:selected {
                background-color:#1f6feb;
                color:#e6edf3;
            }
            QHeaderView::section {
                background-color:#161b22;
                color:#58a6ff;
                border:none;
                border-bottom:1px solid #30363d;
                padding:5px;
                font-size:10px;
                font-weight:bold;
                letter-spacing:0.5px;
            }
        """)
        self._table.cellClicked.connect(
            lambda row, _: self._dose_combo.setCurrentIndex(row))
        right_layout.addWidget(self._table)

        note = QLabel(
            "💡  Lower mAs = fewer photons = more Poisson noise.\n"
            "   Noise σ and SNR measure sinogram-domain noise.\n"
            "   Click any row to preview that dose level.")
        note.setStyleSheet("color:#484f58;font-size:10px;line-height:1.6;")
        note.setWordWrap(True)
        right_layout.addWidget(note)

        splitter.addWidget(right)
        splitter.setSizes([850, 340])
        root.addWidget(splitter)

    # ── HELPERS ─────────────────────────────────────────────────────────────

    @staticmethod
    def _lbl(text):
        l = QLabel(text)
        l.setStyleSheet(_LABEL)
        return l

    def _set_status(self, text, color='#8b949e'):
        self._status.setText(text)
        self._status.setStyleSheet(f"color:{color};font-size:11px;")

    def _parse_sweep_inputs(self):
        """Returns (I0, levels, seed) or None if inputs are invalid."""
        try:
            I0 = float(self._i0_edit.text().replace(',', ''))
            assert I0 > 0
        except Exception:
            self._set_status(
                "❌  Invalid I₀. Enter a positive number (e.g. 10000).",
                '#f85149')
            return None
        try:
            levels = sorted(set(
                float(x.strip())
                for x in self._mas_edit.text().split(',')
                if x.strip()))
            assert levels
        except Exception:
            self._set_status(
                "❌  Invalid mAs levels. Use commas, e.g.: 5, 10, 25, 100",
                '#f85149')
            return None
        try:
            seed = int(self._seed_edit.text())
        except Exception:
            seed = 42
        return I0, levels, seed

    def _get_size(self):
        txt = self._size_combo.currentText()   # "128 × 128" etc.
        return int(txt.split('×')[0].strip().split()[0])

    # ── GENERATE TEST SINOGRAM ──────────────────────────────────────────────

    def _on_generate(self):
        """Build a Shepp-Logan phantom + sinogram from scratch (standalone mode)."""
        size = self._get_size()
        self._gen_btn.setEnabled(False)
        self._gen_btn.setText("⏳  Generating…")
        self._run_btn.setEnabled(False)
        self._set_status(
            f"Generating {size}×{size} Shepp-Logan phantom and sinogram… "
            "This may take 10–30 seconds.", '#8b949e')

        self._phantom_worker = _PhantomWorker(size=size, n_angles=180)
        self._phantom_worker.finished.connect(self._on_phantom_done)
        self._phantom_worker.error.connect(self._on_phantom_error)
        self._phantom_worker.start()

    def _on_phantom_done(self, sinogram, phantom):
        self._sinogram = sinogram
        self._phantom  = phantom
        self._gen_btn.setEnabled(True)
        self._gen_btn.setText("🔬  Generate Test Sinogram")
        self._run_btn.setEnabled(True)
        h, w = sinogram.shape
        self._set_status(
            f"✅  Test sinogram generated ({h} angles × {w} detectors) — "
            f"Shepp-Logan phantom, 180 projection angles. "
            f"Now click 'Run mAs Sweep'.",
            '#3fb950')

    def _on_phantom_error(self, msg):
        self._gen_btn.setEnabled(True)
        self._gen_btn.setText("🔬  Generate Test Sinogram")
        self._set_status(f"❌  Generation failed: {msg[:100]}", '#f85149')

    # ── RUN SWEEP ───────────────────────────────────────────────────────────

    def _on_run_sweep(self):
        if self._sinogram is None:
            self._set_status(
                "❌  No sinogram. Click 'Generate Test Sinogram' first.",
                '#f85149')
            return

        parsed = self._parse_sweep_inputs()
        if parsed is None:
            return
        I0, levels, seed = parsed

        self._run_btn.setEnabled(False)
        self._run_btn.setText("⏳  Running…")
        self._set_status(
            f"Running Poisson noise sweep across {len(levels)} dose levels…",
            '#8b949e')

        self._sweep_worker = _SweepWorker(
            sinogram   = self._sinogram,
            phantom    = self._phantom,
            mas_levels = levels,
            I0_per_mAs = I0,
            seed       = seed,
        )
        self._sweep_worker.finished.connect(self._on_sweep_done)
        self._sweep_worker.error.connect(self._on_sweep_error)
        self._sweep_worker.start()

    def _on_sweep_done(self, results: dict):
        self._results = results
        self._run_btn.setEnabled(True)
        self._run_btn.setText("▶  Run mAs Sweep")
        n = results['summary']['n_levels']
        self._set_status(
            f"✅  Sweep complete — {n} dose levels. "
            f"Use the dose combo box to preview each one.",
            '#3fb950')
        self._fill_table(results)
        self._fill_dose_combo(results)
        self._preview_idx = 0
        self._refresh_plots()

    def _on_sweep_error(self, msg):
        self._run_btn.setEnabled(True)
        self._run_btn.setText("▶  Run mAs Sweep")
        self._set_status(f"❌  Sweep error: {msg[:120]}", '#f85149')

    # ── TABLE ───────────────────────────────────────────────────────────────

    def _fill_table(self, results: dict):
        levels  = results['mas_levels']
        metrics = results['metrics']
        I0_per  = float(self._i0_edit.text().replace(',', ''))
        self._table.setRowCount(len(levels))

        for row, mAs in enumerate(levels):
            m  = metrics[mAs]
            I0 = I0_per * mAs

            def cell(v, fmt="{:.4f}"):
                txt  = "—" if v is None else fmt.format(v)
                item = QTableWidgetItem(txt)
                item.setTextAlignment(Qt.AlignCenter)
                return item

            self._table.setItem(row, 0, cell(mAs,           "{:.0f}"))
            self._table.setItem(row, 1, cell(m['noise_std'], "{:.4f}"))
            self._table.setItem(row, 2, cell(m['snr_dB'],    "{:.2f}"))
            self._table.setItem(row, 3, cell(I0,             "{:.0f}"))

    def _fill_dose_combo(self, results: dict):
        self._dose_combo.blockSignals(True)
        self._dose_combo.clear()
        for mAs in results['mas_levels']:
            self._dose_combo.addItem(f"{mAs:.0f} mAs")
        self._dose_combo.blockSignals(False)
        self._dose_combo.setCurrentIndex(0)

    # ── DOSE COMBO ──────────────────────────────────────────────────────────

    def _on_dose_changed(self, idx):
        if self._results is None or idx < 0:
            return
        self._preview_idx = idx
        if self._view_combo.currentIndex() == 0:
            self._draw_sinogram_view()

    # ── PLOTS ───────────────────────────────────────────────────────────────

    def _refresh_plots(self):
        if self._results is None:
            return
        if self._view_combo.currentIndex() == 0:
            self._draw_sinogram_view()
        else:
            self._draw_curves_view()

    def _draw_sinogram_view(self):
        results     = self._results
        levels      = results['mas_levels']
        sinograms   = results['sinograms']
        preview_mAs = levels[self._preview_idx]
        sino_noisy  = sinograms[preview_mAs]
        # Use the normalised clean sinogram (same scale as noisy) for fair comparison
        sino_clean  = results['sinogram_clean']

        self._canvas.fig.clear()
        self._canvas.fig.patch.set_facecolor('#0d1117')
        gs = gridspec.GridSpec(1, 3, figure=self._canvas.fig,
                               wspace=0.35,
                               left=0.06, right=0.97,
                               top=0.88, bottom=0.12)

        ax1 = self._canvas.fig.add_subplot(gs[0, 0])
        ax2 = self._canvas.fig.add_subplot(gs[0, 1])
        ax3 = self._canvas.fig.add_subplot(gs[0, 2])

        _imshow(ax1, sino_clean.T,
                "Clean Sinogram",
                xlabel="Detector", ylabel="Angle")
        _imshow(ax2, sino_noisy.T,
                f"Noisy — {preview_mAs:.0f} mAs",
                xlabel="Detector")
        diff = sino_noisy - sino_clean
        _imshow(ax3, diff.T,
                "Noise Residual (noisy − clean)",
                cmap="seismic", xlabel="Detector")

        snr  = results['metrics'][preview_mAs]['snr_dB']
        nstd = results['metrics'][preview_mAs]['noise_std']
        self._canvas.fig.suptitle(
            f"{preview_mAs:.0f} mAs  |  SNR = {snr:.1f} dB  |  "
            f"Noise σ = {nstd:.4f}",
            color='#c9d1d9', fontsize=10, y=0.97)
        self._canvas.draw()

    def _draw_curves_view(self):
        results     = self._results
        levels      = results['mas_levels']
        metrics     = results['metrics']
        preview_mAs = levels[self._preview_idx]

        snr_vals  = [metrics[m]['snr_dB']   for m in levels]
        nstd_vals = [metrics[m]['noise_std'] for m in levels]

        self._canvas.fig.clear()
        self._canvas.fig.patch.set_facecolor('#0d1117')
        gs = gridspec.GridSpec(1, 2, figure=self._canvas.fig,
                               wspace=0.40,
                               left=0.08, right=0.97,
                               top=0.88, bottom=0.15)

        ax_snr  = self._canvas.fig.add_subplot(gs[0, 0])
        ax_nstd = self._canvas.fig.add_subplot(gs[0, 1])

        _line(ax_snr, levels, snr_vals, '#58a6ff', 'o',
              'SNR (dB)',
              title='SNR vs Dose Level',
              xlabel='mAs  (log scale)',
              ylabel='SNR (dB)',
              highlight_x=preview_mAs)

        _line(ax_nstd, levels, nstd_vals, '#3fb950', 's',
              'Noise σ',
              title='Noise Std vs Dose Level',
              xlabel='mAs  (log scale)',
              ylabel='Noise σ',
              highlight_x=preview_mAs)

        self._canvas.fig.suptitle(
            "Dose-Quality Curves  (red dashed = selected preview dose)",
            color='#c9d1d9', fontsize=10, y=0.97)
        self._canvas.draw()