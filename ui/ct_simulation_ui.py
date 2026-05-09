"""
ui/ct_simulation_ui.py
----------------------
CT Simulation & Reconstruction Desktop App — ALL UI code lives here.

Non-UI science is delegated to:
    pipeline.ct_pipeline.run_pipeline()   ← orchestrates core/* modules

Requires: PyQt5, numpy, matplotlib, scikit-image, scipy
Install:  pip install PyQt5 numpy matplotlib scikit-image scipy
"""

import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QSlider, QPushButton, QCheckBox, QTabWidget,
    QGroupBox, QGridLayout, QFrame, QSpinBox, QSizePolicy, QSplitter,
    QToolButton, QScrollArea, QProgressBar, QMessageBox, QFileDialog,
    QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon, QPixmap, QPainter, QLinearGradient

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from ui.noise_dose_tab import NoiseDoseTab

# ─────────────────────────────────────────────
#  DARK THEME STYLESHEET
# ─────────────────────────────────────────────
DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #0d1117;
    color: #e6edf3;
    font-family: 'Segoe UI', 'Ubuntu', 'Helvetica Neue', sans-serif;
}

QGroupBox {
    border: 1px solid #30363d;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 8px;
    font-size: 11px;
    font-weight: bold;
    color: #58a6ff;
    letter-spacing: 0.8px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    color: #58a6ff;
}

QLabel {
    color: #8b949e;
    font-size: 12px;
}

QLabel#metric_value {
    color: #3fb950;
    font-size: 28px;
    font-weight: bold;
    font-family: 'Segoe UI', 'Ubuntu', sans-serif;
}

QLabel#metric_label {
    color: #58a6ff;
    font-size: 12px;
    font-weight: bold;
    letter-spacing: 1px;
}

QLabel#section_num {
    background-color: #58a6ff;
    color: #0d1117;
    border-radius: 9px;
    font-size: 10px;
    font-weight: bold;
    min-width: 18px;
    max-width: 18px;
    min-height: 18px;
    max-height: 18px;
    qproperty-alignment: AlignCenter;
}

QLabel#section_title {
    color: #e6edf3;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.8px;
}

QComboBox {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 4px;
    padding: 5px 10px;
    color: #e6edf3;
    font-size: 12px;
    min-height: 28px;
}

QComboBox:hover { border-color: #58a6ff; }
QComboBox::drop-down { border: none; width: 20px; }
QComboBox::down-arrow { image: none; border-left: 4px solid transparent; border-right: 4px solid transparent; border-top: 5px solid #8b949e; margin-right: 5px; }
QComboBox QAbstractItemView { background-color: #161b22; border: 1px solid #30363d; color: #e6edf3; selection-background-color: #1f6feb; }

QSlider::groove:horizontal {
    height: 4px;
    background: #30363d;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
    background: #58a6ff;
}

QSlider::sub-page:horizontal { background: #1f6feb; border-radius: 2px; }
QSlider::handle:horizontal:hover { background: #79c0ff; }

QSpinBox, QDoubleSpinBox {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 4px;
    padding: 4px 8px;
    color: #e6edf3;
    font-size: 12px;
    min-height: 26px;
}

QSpinBox:hover, QDoubleSpinBox:hover { border-color: #58a6ff; }
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    background-color: #30363d; border: none; width: 16px;
}

QCheckBox {
    color: #8b949e;
    font-size: 12px;
    spacing: 8px;
}

QCheckBox::indicator {
    width: 14px; height: 14px;
    border: 1px solid #30363d;
    border-radius: 3px;
    background: #161b22;
}

QCheckBox::indicator:checked {
    background: #1f6feb;
    border-color: #58a6ff;
    image: none;
}

QCheckBox::indicator:checked::after {
    content: "✓";
}

QPushButton#run_btn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #1f6feb, stop:1 #388bfd);
    color: white;
    border: none;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 1px;
    min-height: 42px;
    padding: 0 20px;
}

QPushButton#run_btn:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #388bfd, stop:1 #58a6ff);
}

QPushButton#run_btn:pressed { background: #1158c7; }

QPushButton#run_btn:disabled {
    background: #21262d;
    color: #484f58;
}

QPushButton#secondary_btn {
    background-color: #21262d;
    color: #8b949e;
    border: 1px solid #30363d;
    border-radius: 6px;
    font-size: 12px;
    min-height: 34px;
    padding: 0 14px;
}

QPushButton#secondary_btn:hover {
    background-color: #30363d;
    color: #e6edf3;
    border-color: #58a6ff;
}

QTabWidget::pane {
    border: 1px solid #30363d;
    border-radius: 6px;
    background-color: #0d1117;
}

QTabBar::tab {
    background: #161b22;
    color: #8b949e;
    border: 1px solid #30363d;
    border-bottom: none;
    border-radius: 4px 4px 0 0;
    padding: 7px 16px;
    margin-right: 2px;
    font-size: 12px;
}

QTabBar::tab:selected {
    background: #0d1117;
    color: #58a6ff;
    border-color: #30363d;
    border-bottom-color: #0d1117;
}

QTabBar::tab:hover:!selected { background: #21262d; color: #e6edf3; }

QScrollBar:vertical {
    background: #0d1117; width: 8px; margin: 0;
}

QScrollBar::handle:vertical {
    background: #30363d; border-radius: 4px; min-height: 20px;
}

QScrollBar::handle:vertical:hover { background: #484f58; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

QProgressBar {
    background: #21262d;
    border: none;
    border-radius: 3px;
    height: 4px;
    text-align: center;
    color: transparent;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #1f6feb, stop:1 #3fb950);
    border-radius: 3px;
}

QFrame#divider {
    background-color: #21262d;
    max-height: 1px;
    min-height: 1px;
}

QFrame#metric_card {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
}
"""


# ─────────────────────────────────────────────
#  WORKER THREAD
# ─────────────────────────────────────────────
class SimulationWorker(QThread):
    """Runs the CT pipeline on a background thread to keep the UI responsive."""

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            # All CT science lives in pipeline/ct_pipeline.py
            from pipeline.ct_pipeline import run_pipeline
            results = run_pipeline(self.params, progress_cb=self.progress.emit)
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


# ─────────────────────────────────────────────
#  MATPLOTLIB CANVAS
# ─────────────────────────────────────────────
class MplCanvas(FigureCanvas):
    def __init__(self, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#0d1117')
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #0d1117;")

    def clear(self):
        self.fig.clear()
        self.draw()


def imshow_ax(ax, data, title="", cmap="gray", colorbar=True, xlabel="", ylabel=""):
    """Styled matplotlib imshow helper used exclusively for display."""
    ax.set_facecolor('#0d1117')
    im = ax.imshow(data, cmap=cmap, aspect='auto',
                   interpolation='nearest',
                   vmin=data.min(), vmax=data.max())
    if colorbar:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cb.ax.yaxis.set_tick_params(color='#8b949e', labelcolor='#8b949e', labelsize=8)
        cb.outline.set_edgecolor('#30363d')
    ax.set_title(title, color='#c9d1d9', fontsize=11, pad=6, fontfamily='DejaVu Sans')
    ax.tick_params(colors='#8b949e', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    if xlabel:
        ax.set_xlabel(xlabel, color='#8b949e', fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color='#8b949e', fontsize=9)
    return im


# ─────────────────────────────────────────────
#  SECTION HEADER WIDGET
# ─────────────────────────────────────────────
class SectionHeader(QWidget):
    def __init__(self, num, title, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(8)

        num_lbl = QLabel(str(num))
        num_lbl.setObjectName("section_num")
        num_lbl.setFixedSize(20, 20)
        num_lbl.setAlignment(Qt.AlignCenter)

        title_lbl = QLabel(title.upper())
        title_lbl.setObjectName("section_title")

        layout.addWidget(num_lbl)
        layout.addWidget(title_lbl)
        layout.addStretch()


# ─────────────────────────────────────────────
#  METRIC CARD
# ─────────────────────────────────────────────
class MetricCard(QFrame):
    def __init__(self, label, value="—", parent=None):
        super().__init__(parent)
        self.setObjectName("metric_card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(2)

        lbl = QLabel(label)
        lbl.setObjectName("metric_label")
        lbl.setAlignment(Qt.AlignCenter)

        self.val_lbl = QLabel(value)
        self.val_lbl.setObjectName("metric_value")
        self.val_lbl.setAlignment(Qt.AlignCenter)

        layout.addWidget(lbl)
        layout.addWidget(self.val_lbl)

    def set_value(self, v):
        self.val_lbl.setText(v)


# ─────────────────────────────────────────────
#  MAIN WINDOW
# ─────────────────────────────────────────────
class CTSimApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CT Simulation & Reconstruction")
        self.setMinimumSize(1280, 800)
        self.resize(1440, 900)
        self.setStyleSheet(DARK_STYLESHEET)
        self._results = None
        self._worker  = None
        self._build_ui()

    # ── BUILD UI ──────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        root.addWidget(self._build_left_panel(),   stretch=0)
        root.addWidget(self._build_center_panel(), stretch=3)
        root.addWidget(self._build_right_panel(),  stretch=0)

    # ── LEFT PANEL ────────────────────────────
    def _build_left_panel(self):
        panel = QFrame()
        panel.setFixedWidth(270)
        panel.setObjectName("metric_card")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        title = QLabel("CT SIMULATION\n& RECONSTRUCTION")
        title.setStyleSheet("color:#58a6ff;font-size:14px;font-weight:700;letter-spacing:0.5px;font-family:'Segoe UI','Ubuntu',sans-serif;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        sub = QLabel("Numerical Simulation of CT Acquisition\nand Image Reconstruction")
        sub.setStyleSheet("color:#484f58;font-size:10px;font-family:'Segoe UI','Ubuntu',sans-serif;")
        sub.setAlignment(Qt.AlignCenter)
        layout.addWidget(sub)

        div = QFrame(); div.setObjectName("divider")
        layout.addWidget(div)
        layout.addSpacing(4)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(0, 0, 4, 0)
        inner_layout.setSpacing(12)

        # ── 1. PHANTOM ──
        inner_layout.addWidget(SectionHeader(1, "Phantom Settings"))

        inner_layout.addWidget(QLabel("Phantom Type"))
        self.phantom_combo = QComboBox()
        self.phantom_combo.addItems(["Shepp–Logan", "Modified Shepp–Logan"])
        inner_layout.addWidget(self.phantom_combo)

        inner_layout.addWidget(QLabel("Image Size"))
        self.size_combo = QComboBox()
        self.size_combo.addItems(["128 × 128", "256 × 256", "512 × 512"])
        self.size_combo.setCurrentIndex(1)
        inner_layout.addWidget(self.size_combo)

        # ── 2. PROJECTION ──
        inner_layout.addWidget(SectionHeader(2, "Projection Settings"))

        inner_layout.addWidget(QLabel("Number of Angles"))
        angles_row = QHBoxLayout()
        self.angles_spin = QSpinBox()
        self.angles_spin.setRange(10, 360)
        self.angles_spin.setValue(180)
        self.angles_slider = QSlider(Qt.Horizontal)
        self.angles_slider.setRange(10, 360)
        self.angles_slider.setValue(180)
        self.angles_slider.valueChanged.connect(self.angles_spin.setValue)
        self.angles_spin.valueChanged.connect(self.angles_slider.setValue)
        angles_row.addWidget(self.angles_spin)
        inner_layout.addLayout(angles_row)
        inner_layout.addWidget(self.angles_slider)
        self._add_range_labels(inner_layout, "30", "360")

        # ── 3. NOISE ──
        inner_layout.addWidget(SectionHeader(3, "Noise / Dose Settings"))

        inner_layout.addWidget(QLabel("mAs (Dose)"))
        mas_row = QHBoxLayout()
        self.mas_spin = QSpinBox()
        self.mas_spin.setRange(1, 100)
        self.mas_spin.setValue(20)
        mas_row.addWidget(self.mas_spin)
        inner_layout.addLayout(mas_row)
        self.mas_slider = QSlider(Qt.Horizontal)
        self.mas_slider.setRange(1, 100)
        self.mas_slider.setValue(20)
        self.mas_slider.valueChanged.connect(self.mas_spin.setValue)
        self.mas_spin.valueChanged.connect(self.mas_slider.setValue)
        inner_layout.addWidget(self.mas_slider)
        self._add_range_labels(inner_layout, "1 (Low)", "100 (High)")

        self.noise_check = QCheckBox("Add Poisson Noise")
        self.noise_check.setChecked(True)
        self.noise_check.setStyleSheet("color:#e6edf3;font-size:12px;")
        inner_layout.addWidget(self.noise_check)

        # ── 4. RECONSTRUCTION ──
        inner_layout.addWidget(SectionHeader(4, "Reconstruction Settings"))

        inner_layout.addWidget(QLabel("Reconstruction Method"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(["FBP (Filtered Back-Projection)", "SART (Algebraic)"])
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        inner_layout.addWidget(self.method_combo)

        self.iter_label = QLabel("SART Iterations")
        inner_layout.addWidget(self.iter_label)
        self.iter_spin = QSpinBox()
        self.iter_spin.setRange(1, 100)
        self.iter_spin.setValue(20)
        self.iter_slider = QSlider(Qt.Horizontal)
        self.iter_slider.setRange(1, 100)
        self.iter_slider.setValue(20)
        self.iter_slider.valueChanged.connect(self.iter_spin.setValue)
        self.iter_spin.valueChanged.connect(self.iter_slider.setValue)
        inner_layout.addWidget(self.iter_spin)
        inner_layout.addWidget(self.iter_slider)
        self._add_range_labels(inner_layout, "1", "100")
        self._on_method_changed(0)

        inner_layout.addStretch()
        scroll.setWidget(inner)
        layout.addWidget(scroll, stretch=1)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(4)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color:#484f58;font-size:11px;")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        # Buttons
        self.run_btn = QPushButton("▶  RUN SIMULATION")
        self.run_btn.setObjectName("run_btn")
        self.run_btn.clicked.connect(self._run_simulation)
        layout.addWidget(self.run_btn)

        btn_row = QHBoxLayout()
        reset_btn = QPushButton("↺  RESET")
        reset_btn.setObjectName("secondary_btn")
        reset_btn.clicked.connect(self._reset)
        export_btn = QPushButton("⬇  EXPORT")
        export_btn.setObjectName("secondary_btn")
        export_btn.clicked.connect(self._export)
        btn_row.addWidget(reset_btn)
        btn_row.addWidget(export_btn)
        layout.addLayout(btn_row)

        return panel

    def _add_range_labels(self, layout, low, high):
        row = QHBoxLayout()
        l = QLabel(low); l.setStyleSheet("color:#484f58;font-size:10px;")
        h = QLabel(high); h.setStyleSheet("color:#484f58;font-size:10px;")
        h.setAlignment(Qt.AlignRight)
        row.addWidget(l); row.addWidget(h)
        layout.addLayout(row)

    def _on_method_changed(self, idx):
        is_sart = idx == 1
        self.iter_label.setEnabled(is_sart)
        self.iter_spin.setEnabled(is_sart)
        self.iter_slider.setEnabled(is_sart)

    # ── CENTER PANEL ──────────────────────────
    def _build_center_panel(self):
        panel = QFrame()
        panel.setObjectName("metric_card")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)

        self.tab_phantom = QWidget()
        t1 = QVBoxLayout(self.tab_phantom)
        t1.setContentsMargins(4, 4, 4, 4)
        self.canvas_phantom = MplCanvas(6, 4, 90)
        t1.addWidget(self.canvas_phantom)
        self.tabs.addTab(self.tab_phantom, "⬛ Phantom")

        self.tab_sino = QWidget()
        t2 = QVBoxLayout(self.tab_sino)
        t2.setContentsMargins(4, 4, 4, 4)
        self.canvas_sino = MplCanvas(6, 4, 90)
        t2.addWidget(self.canvas_sino)
        self.tabs.addTab(self.tab_sino, "⚟ Sinogram")

        self.tab_recon = QWidget()
        t4 = QVBoxLayout(self.tab_recon)
        t4.setContentsMargins(4, 4, 4, 4)
        self.canvas_recon = MplCanvas(6, 4, 90)
        t4.addWidget(self.canvas_recon)
        self.tabs.addTab(self.tab_recon, "↺ Reconstruction")

        self.tab_diff = QWidget()
        t5 = QVBoxLayout(self.tab_diff)
        t5.setContentsMargins(4, 4, 4, 4)
        self.canvas_diff = MplCanvas(6, 4, 90)
        t5.addWidget(self.canvas_diff)
        self.tabs.addTab(self.tab_diff, "◎ Difference Map")

        # ── Noise & Dose tab (your task) ──────
        self.noise_tab = NoiseDoseTab(parent=self)
        self.tabs.addTab(self.noise_tab, "📉 Noise & Dose")

        layout.addWidget(self.tabs)

        opts_frame = QFrame()
        opts_frame.setStyleSheet("background:#161b22;border-radius:4px;padding:4px;")
        opts_layout = QHBoxLayout(opts_frame)
        opts_layout.setContentsMargins(10, 4, 10, 4)

        opts_layout.addWidget(QLabel("Display Options:"))
        self.grid_check = QCheckBox("Show Grid")
        self.norm_check = QCheckBox("Normalize")
        self.norm_check.setChecked(True)
        self.log_check  = QCheckBox("Log Scale (Sinogram)")
        opts_layout.addWidget(self.grid_check)
        opts_layout.addWidget(self.norm_check)
        opts_layout.addWidget(self.log_check)
        opts_layout.addStretch()

        opts_layout.addWidget(QLabel("Colormap:"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(["gray", "hot", "viridis", "plasma", "bone", "inferno"])
        self.cmap_combo.setFixedWidth(100)
        opts_layout.addWidget(self.cmap_combo)
        layout.addWidget(opts_frame)

        return panel

    # ── RIGHT PANEL ───────────────────────────
    def _build_right_panel(self):
        panel = QFrame()
        panel.setFixedWidth(310)
        panel.setObjectName("metric_card")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        hdr = QLabel("QUANTITATIVE METRICS")
        hdr.setStyleSheet("color:#58a6ff;font-size:12px;font-weight:700;letter-spacing:1px;")
        layout.addWidget(hdr)

        cards_row = QHBoxLayout()
        self.rmse_card = MetricCard("RMSE")
        self.ssim_card = MetricCard("SSIM")
        self.ssim_card.val_lbl.setStyleSheet("color:#3fb950;font-size:28px;font-weight:bold;font-family:'Courier New';")
        cards_row.addWidget(self.rmse_card)
        cards_row.addWidget(self.ssim_card)
        layout.addLayout(cards_row)

        info_frame = QFrame()
        info_frame.setStyleSheet("background:#161b22;border-radius:6px;")
        info_layout = QGridLayout(info_frame)
        info_layout.setContentsMargins(10, 8, 10, 8)
        info_layout.setSpacing(4)

        def info_row(r, k, v):
            kl = QLabel(k); kl.setStyleSheet("color:#484f58;font-size:11px;")
            vl = QLabel(v); vl.setStyleSheet("color:#8b949e;font-size:11px;")
            info_layout.addWidget(kl, r, 0)
            info_layout.addWidget(vl, r, 1)
            return vl

        self.info_gt      = info_row(0, "Ground Truth",       "Shepp–Logan Phantom")
        self.info_method  = info_row(1, "Reconstruction",     "FBP (Filtered Back-Projection)")
        self.info_mas     = info_row(2, "mAs (Dose)",         "20")
        self.info_angles  = info_row(3, "Angles",             "180")
        layout.addWidget(info_frame)

        div = QFrame(); div.setObjectName("divider")
        layout.addWidget(div)

        hdr2 = QLabel("PERFORMANCE vs DOSE (mAs)")
        hdr2.setStyleSheet("color:#58a6ff;font-size:11px;font-weight:600;letter-spacing:0.5px;")
        layout.addWidget(hdr2)
        self.canvas_sweep = MplCanvas(3, 2.5, 90)
        layout.addWidget(self.canvas_sweep)

        div2 = QFrame(); div2.setObjectName("divider")
        layout.addWidget(div2)

        hdr3 = QLabel("RECONSTRUCTION COMPARISON")
        hdr3.setStyleSheet("color:#58a6ff;font-size:11px;font-weight:600;letter-spacing:0.5px;")
        layout.addWidget(hdr3)
        self.canvas_compare = MplCanvas(3, 2, 90)
        layout.addWidget(self.canvas_compare)

        div3 = QFrame(); div3.setObjectName("divider")
        layout.addWidget(div3)

        self.summary_label = QLabel("SUMMARY")
        self.summary_label.setStyleSheet("color:#3fb950;font-size:11px;font-weight:600;letter-spacing:0.5px;")
        layout.addWidget(self.summary_label)
        self.summary_text = QLabel("Run a simulation to see results.")
        self.summary_text.setWordWrap(True)
        self.summary_text.setStyleSheet("color:#8b949e;font-size:11px;line-height:1.4;")
        layout.addWidget(self.summary_text)

        layout.addStretch()
        return panel

    # ─────────────────────────────────────────
    #  SIMULATION LOGIC
    # ─────────────────────────────────────────
    def _get_image_size(self):
        txt = self.size_combo.currentText()
        return int(txt.split("×")[0].strip())

    def _get_method(self):
        return "FBP" if self.method_combo.currentIndex() == 0 else "SART"

    def _run_simulation(self):
        self.run_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting simulation…")

        params = {
            "image_size":      self._get_image_size(),
            "n_angles":        self.angles_spin.value(),
            "n_detectors":     256,
            "mas":             self.mas_spin.value(),
            "add_noise":       self.noise_check.isChecked(),
            "method":          self._get_method(),
            "sart_iterations": self.iter_spin.value(),
        }

        self._worker = SimulationWorker(params)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, pct, msg):
        self.progress_bar.setValue(pct)
        self.status_label.setText(msg)

    def _on_finished(self, results):
        self._results = results
        self.run_btn.setEnabled(True)
        self.status_label.setText("Simulation complete ✓")
        self._update_displays(results)

        # ── Feed the clean sinogram + phantom into the Noise & Dose tab ──
        self.noise_tab.set_sinogram(
            results["sinogram_clean"],
            phantom=results["phantom"],
        )

    def _on_error(self, msg):
        self.run_btn.setEnabled(True)
        self.status_label.setText("Error!")
        QMessageBox.critical(self, "Simulation Error",
                             f"An error occurred:\n\n{msg}\n\n"
                             "Make sure scikit-image and scipy are installed.")

    # ─────────────────────────────────────────
    #  DISPLAY UPDATES
    # ─────────────────────────────────────────
    def _update_displays(self, r):
        cmap      = self.cmap_combo.currentText()
        show_grid = self.grid_check.isChecked()

        # ── Phantom tab ──
        self.canvas_phantom.fig.clear()
        ax1 = self.canvas_phantom.fig.add_subplot(121)
        ax2 = self.canvas_phantom.fig.add_subplot(122)
        self.canvas_phantom.fig.patch.set_facecolor('#0d1117')
        imshow_ax(ax1, r["phantom"], "Original Phantom", cmap=cmap)
        imshow_ax(ax2, r["sinogram_clean"].T, "Sinogram (Clean)", cmap=cmap,
                  xlabel="Detector Position", ylabel="Projection Angle (°)")
        if show_grid:
            ax1.grid(True, alpha=0.2, color='#58a6ff')
            ax2.grid(True, alpha=0.2, color='#58a6ff')
        self.canvas_phantom.fig.tight_layout(pad=1.5)
        self.canvas_phantom.draw()

        # ── Sinogram tab ──
        self.canvas_sino.fig.clear()
        ax = self.canvas_sino.fig.add_subplot(111)
        self.canvas_sino.fig.patch.set_facecolor('#0d1117')
        data = np.log1p(r["sinogram_clean"].T) if self.log_check.isChecked() else r["sinogram_clean"].T
        imshow_ax(ax, data, "Sinogram (Clean)", cmap=cmap,
                  xlabel="Detector Position", ylabel="Projection Angle (°)")
        self.canvas_sino.fig.tight_layout(pad=1.5)
        self.canvas_sino.draw()

        # ── Reconstruction tab ──
        self.canvas_recon.fig.clear()
        ax1 = self.canvas_recon.fig.add_subplot(121)
        ax2 = self.canvas_recon.fig.add_subplot(122)
        self.canvas_recon.fig.patch.set_facecolor('#0d1117')
        imshow_ax(ax1, r["recon_fbp"], "Reconstruction (FBP)", cmap=cmap)
        imshow_ax(ax2, r["recon"],     f"Reconstruction ({r['method']})", cmap=cmap)
        self.canvas_recon.fig.tight_layout(pad=1.5)
        self.canvas_recon.draw()

        # ── Difference Map tab ──
        self.canvas_diff.fig.clear()
        ax1 = self.canvas_diff.fig.add_subplot(131)
        ax2 = self.canvas_diff.fig.add_subplot(132)
        ax3 = self.canvas_diff.fig.add_subplot(133)
        self.canvas_diff.fig.patch.set_facecolor('#0d1117')
        imshow_ax(ax1, r["phantom"], "Ground Truth", cmap=cmap)
        imshow_ax(ax2, r["recon"],   f"{r['method']} Recon", cmap=cmap)
        imshow_ax(ax3, np.abs(r["diff"]), "Error Map", cmap="inferno")
        self.canvas_diff.fig.tight_layout(pad=0.5)
        self.canvas_diff.fig.subplots_adjust(wspace=0.08)
        self.canvas_diff.draw()

        # ── Metrics ──
        rmse_val = r["rmse"]
        ssim_val = r["ssim"]
        self.rmse_card.set_value(f"{rmse_val:.4f}")
        self.ssim_card.set_value(f"{ssim_val:.4f}")
        self.info_method.setText(r["method"] + (" (Filtered Back-Projection)" if r["method"] == "FBP" else " (Algebraic)"))
        self.info_mas.setText(str(r["mas"]))
        self.info_angles.setText(str(r["n_angles"]))

        # ── Sweep chart ──
        self.canvas_sweep.fig.clear()
        ax = self.canvas_sweep.fig.add_subplot(111)
        ax.set_facecolor('#0d1117')
        self.canvas_sweep.fig.patch.set_facecolor('#0d1117')
        ax.semilogx(r["mas_vals"], r["rmse_sweep"], 'o-', color='#58a6ff', lw=1.5, ms=4, label='RMSE (↓)')
        ax.semilogx(r["mas_vals"], r["ssim_sweep"], 's-', color='#3fb950', lw=1.5, ms=4, label='SSIM (↑)')
        ax.set_xlabel("mAs (Dose) [log scale]", color='#8b949e', fontsize=9)
        ax.tick_params(colors='#8b949e', labelsize=8)
        for s in ax.spines.values(): s.set_edgecolor('#30363d')
        ax.legend(fontsize=8, facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9', loc='center right')
        ax.set_ylim(0, 1.05)
        self.canvas_sweep.fig.tight_layout(pad=1.0)
        self.canvas_sweep.draw()

        # ── Comparison bar chart ──
        self.canvas_compare.fig.clear()
        ax = self.canvas_compare.fig.add_subplot(111)
        ax.set_facecolor('#0d1117')
        self.canvas_compare.fig.patch.set_facecolor('#0d1117')
        methods   = ["FBP", r["method"] if r["method"] != "FBP" else "SART"]
        rmse_vals = [r["rmse_fbp"], r["rmse"]]
        ssim_vals = [r["ssim_fbp"], r["ssim"]]
        x     = np.arange(len(methods))
        width = 0.35
        ax.bar(x - width/2, rmse_vals, width, color='#1f6feb', label='RMSE (↓)')
        ax.bar(x + width/2, ssim_vals, width, color='#3fb950', label='SSIM (↑)')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=9, color='#8b949e')
        ax.tick_params(colors='#8b949e', labelsize=8)
        for s in ax.spines.values(): s.set_edgecolor('#30363d')
        ax.legend(fontsize=8, facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
        ax.set_ylim(0, 1.0)
        self.canvas_compare.fig.tight_layout(pad=0.8)
        self.canvas_compare.draw()

        # ── Summary ──
        mas = r["mas"]
        if rmse_val < r["rmse_fbp"] and r["method"] != "FBP":
            self.summary_text.setText(
                f"At low dose ({mas} mAs), <b style='color:#3fb950'>{r['method']}</b> provides better "
                f"quality (lower RMSE, higher SSIM) than FBP."
            )
        elif r["method"] == "FBP":
            self.summary_text.setText(
                f"FBP reconstruction at {mas} mAs. "
                f"RMSE: <b style='color:#58a6ff'>{rmse_val:.4f}</b>, SSIM: <b style='color:#3fb950'>{ssim_val:.4f}</b>."
            )
        else:
            self.summary_text.setText(
                f"At {mas} mAs, FBP achieves comparable quality. "
                f"Try lower mAs to see SART advantages."
            )

    # ─────────────────────────────────────────
    #  ACTIONS
    # ─────────────────────────────────────────
    def _reset(self):
        self.phantom_combo.setCurrentIndex(0)
        self.size_combo.setCurrentIndex(1)
        self.angles_slider.setValue(180)
        self.mas_slider.setValue(20)
        self.noise_check.setChecked(True)
        self.method_combo.setCurrentIndex(0)
        self.iter_slider.setValue(20)
        self.progress_bar.setValue(0)
        self.status_label.setText("Ready")
        for c in [self.canvas_phantom, self.canvas_sino,
                  self.canvas_recon,   self.canvas_diff,
                  self.canvas_sweep,   self.canvas_compare]:
            c.clear()
        self.rmse_card.set_value("—")
        self.ssim_card.set_value("—")
        self.summary_text.setText("Run a simulation to see results.")

    def _export(self):
        if self._results is None:
            QMessageBox.information(self, "Export", "Please run a simulation first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Results", "ct_simulation", "PNG Files (*.png)")
        if not path:
            return
        r = self._results
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), facecolor='#0d1117')
        titles = ["Original Phantom", "Sinogram (Clean)",
                  f"Reconstruction ({r['method']})", "Error Map", "Metrics"]
        imgs   = [r["phantom"], r["sinogram_clean"].T,
                  r["recon"],   np.abs(r["diff"]),     None]
        for ax, img, title in zip(axes.flat, imgs, titles):
            ax.set_facecolor('#0d1117')
            if img is not None:
                cm = "inferno" if "Error" in title else "gray"
                im = ax.imshow(img, cmap=cm, aspect='auto')
                fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
            else:
                ax.axis('off')
                ax.text(0.5, 0.6, f"RMSE: {r['rmse']:.4f}", ha='center', va='center',
                        color='#58a6ff', fontsize=16, transform=ax.transAxes)
                ax.text(0.5, 0.4, f"SSIM: {r['ssim']:.4f}", ha='center', va='center',
                        color='#3fb950', fontsize=16, transform=ax.transAxes)
            ax.set_title(title, color='#c9d1d9', fontsize=11)
            ax.tick_params(colors='#8b949e', labelsize=8)
        fig.tight_layout(pad=2)
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
        plt.close(fig)
        QMessageBox.information(self, "Export", f"Saved to:\n{path}")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("CT Simulation & Reconstruction")

    missing = []
    for pkg, name in [("skimage", "scikit-image"), ("scipy", "scipy")]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(name)

    if missing:
        msg = QMessageBox()
        msg.setWindowTitle("Missing Dependencies")
        msg.setText(
            f"The following packages are required but not installed:\n\n"
            f"  {', '.join(missing)}\n\n"
            f"Install them with:\n  pip install {' '.join(missing)}"
        )
        msg.setIcon(QMessageBox.Warning)
        msg.exec_()

    window = CTSimApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()