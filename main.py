import sys
import json
import math
import pathlib
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QComboBox, QDoubleSpinBox, QSpinBox,
                             QTableWidget, QTableWidgetItem, QHeaderView, QPushButton,
                             QGroupBox, QCheckBox, QTabWidget, QFileDialog, QMessageBox)
from PyQt6.QtGui import QPixmap, QPainter, QColor, QFont, QQuaternion
from PyQt6.QtCore import Qt, QTimer, QRect
import pyqtgraph.opengl as gl

from physics import calculate_fields, calculate_poynting, calculate_h_field, C, MU0

# Propagation frames: (k_hat, e1_hat, e2_hat) where e2 = k × e1 (right-hand rule for B)
_BASE_FRAMES = {
    '+Z': (np.array([0.,0.,1.]),  np.array([1.,0.,0.]), np.array([0., 1.,0.])),
    '-Z': (np.array([0.,0.,-1.]), np.array([1.,0.,0.]), np.array([0.,-1.,0.])),
    '+X': (np.array([1.,0.,0.]),  np.array([0.,1.,0.]), np.array([0., 0.,1.])),
    '-X': (np.array([-1.,0.,0.]), np.array([0.,1.,0.]), np.array([0., 0.,-1.])),
    '+Y': (np.array([0.,1.,0.]),  np.array([1.,0.,0.]), np.array([0., 0.,-1.])),
    '-Y': (np.array([0.,-1.,0.]), np.array([1.,0.,0.]), np.array([0., 0.,1.])),
}

_BG_COLORS = {
    "Чорний": {"gl": "k", "grid": (0.3, 0.3, 0.3, 1.0)},
    "Білий":  {"gl": "w", "grid": (0.5, 0.5, 0.5, 1.0)},
}

# Wave colors — kept here so both __init__ and update_plot use the same values
_W1_E = (0.10, 0.40, 1.00, 1.0)   # blue   — wave 1 electric field
_W1_B = (1.00, 0.20, 0.20, 1.0)   # red    — wave 1 magnetic field
_W1_S = (0.10, 0.90, 0.10, 1.0)   # green  — wave 1 Poynting
_W1_K = (0.20, 1.00, 0.40, 1.0)   # lime   — wave 1 propagation
_W2_E = (0.10, 0.85, 0.85, 1.0)   # cyan   — wave 2 electric field
_W2_B = (1.00, 0.55, 0.10, 1.0)   # orange — wave 2 magnetic field
_W2_S = (0.90, 0.90, 0.10, 1.0)   # yellow — wave 2 Poynting
_W2_K = (0.60, 1.00, 0.30, 1.0)   # light green — wave 2 propagation

# Coordinate-axis colors — distinct from all wave colors above to avoid confusion
_AXIS_X = (0.65, 0.20, 0.85, 1.0)  # violet
_AXIS_Y = (0.95, 0.25, 0.65, 1.0)  # magenta/pink
_AXIS_Z = (0.65, 0.45, 0.20, 1.0)  # brown/tan

def _c2hex(r, g, b, *_): return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'

def _txt_rgba(c):
    """GLTextItem routes its color through pg.mkColor(), which treats a 4-tuple
    as 0-255 ints — unlike GLLinePlotItem/GLMeshItem, which take 0-1 floats for
    OpenGL vertex colors. Our palette constants are all 0-1 floats, so any color
    handed to a GLTextItem must be rescaled here or it renders as near-invisible
    (truncated to 0 and alpha ~1/255)."""
    return tuple(int(round(v * 255)) for v in c)

# Camera orientation for Tab 1 ("Хвиля 1"): X up, Z left-to-right, Y toward the viewer.
# A plain elevation/azimuth pair can't reach this exact axis mapping (it's the "wrong-handed"
# 3-cycle of the two reachable via X-then-Z Euler rotation), so it's set as a fixed quaternion.
_WAVE1_CAMERA_ROTATION = QQuaternion(0.5, 0.5, 0.5, 0.5)

# Camera orientation for Tab 2 ("Хвиля 2"): X down, Y away from the viewer, Z (visually)
# right-to-left. A single rotation can only flip two of Tab 1's three axes and stay a proper
# rotation (flipping all three would require mirroring the scene). This quaternion gives the
# proper-rotation part (X down, Y away, Z left-to-right); the Z arrow/ticks/label are then
# drawn toward -Z (see s_z in update_plot) so they read right-to-left on screen — which also
# matches wave 2's own propagation direction (-Z by default), so its curve and indicator
# arrows need no special-casing and stay physically correct.
_WAVE2_CAMERA_ROTATION = QQuaternion(-0.5, 0.5, -0.5, 0.5)

# Precomputed λ/4 tick labels: index i → text for position i*λ/4
_TICK_LABELS = [
    "0", "λ/4", "λ/2", "3λ/4", "λ",
    "5λ/4", "3λ/2", "7λ/4", "2λ",
    "9λ/4", "5λ/2", "11λ/4", "3λ",
    "13λ/4", "7λ/2", "15λ/4", "4λ",
    "17λ/4", "9λ/2", "19λ/4", "5λ",
    "21λ/4", "11λ/2", "23λ/4", "6λ",
    "25λ/4", "13λ/2", "27λ/4", "7λ", "29λ/4",
]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Візуалізація вектора Пойнтінга")
        self.resize(1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left: control panel
        control_panel = QWidget()
        control_panel.setFixedWidth(300)
        control_layout = QVBoxLayout(control_panel)
        main_layout.addWidget(control_panel)

        # Right: 3D GL view
        self.view = gl.GLViewWidget()
        self.view.opts['distance'] = 20
        self.grid = gl.GLGridItem(glOptions='translucent')
        self.grid.scale(2, 2, 1)
        self.view.addItem(self.grid)
        # Single colored axis-arrow item (X=red, Y=green, Z=blue) — replaces GLAxisItem
        self.axis_arrows = self._make_axis_arrows()
        self.view.addItem(self.axis_arrows)

        # λ/4 tick marks on all three world axes
        self.tick_marks = gl.GLLinePlotItem(
            width=1, color=(0.5, 0.5, 0.5, 0.6), mode='lines', glOptions='translucent')
        self.view.addItem(self.tick_marks)

        # Label pool — 30 per axis × 3 axes = 90 GLTextItem objects
        self.tick_label_pool: list[gl.GLTextItem] = []
        for _ in range(90):
            lbl = gl.GLTextItem(pos=(0, 0, 0), text='', color=_txt_rgba((0.7, 0.7, 0.7, 0.9)))
            lbl.setVisible(False)
            self.view.addItem(lbl)
            self.tick_label_pool.append(lbl)

        # Current text color for tick labels (updated by _apply_background)
        self._txt_color = _txt_rgba((1.0, 1.0, 1.0, 1.0))

        # Scenario tabs — each stores independent wave/modulation + source parameters
        self.scenario_params = {
            0: {'pol': 'linear', 'phase': 0.0, 'amplitude': 1.0, 'mod_type': 'none', 'mod_freq': 0.1, 'mod_index': 0.5,
                'n_periods': 2, 'prop_dir': '+Z', 'dir_mode': 'Базова вісь', 'dir_vec': [0., 0., 1.]},
            1: {'pol': 'linear', 'phase': 0.0, 'amplitude': 1.0, 'mod_type': 'none', 'mod_freq': 0.1, 'mod_index': 0.5,
                'n_periods': 2, 'prop_dir': '-Z', 'dir_mode': 'Базова вісь', 'dir_vec': [0., 0., -1.],
                'x0': 0.0, 'y0': 0.0, 'z0': 3.0},
        }
        self.scenario_tabs = QTabWidget()
        self.scenario_tabs.addTab(QWidget(), "Хвиля 1 — (0,0,0)")
        self.scenario_tabs.addTab(QWidget(), "Хвиля 2 — Довільне джерело")
        self.scenario_tabs.addTab(QWidget(), "Суперпозиція (1+2)")
        self.scenario_tabs.currentChanged.connect(self._on_scenario_changed)
        # Show only the tab bar — tab content is the shared graph, not the QWidget placeholders
        self.scenario_tabs.setMaximumHeight(self.scenario_tabs.tabBar().sizeHint().height() + 4)

        # Always-visible legend strip between tab bar and graph
        self.legend_strip = QLabel("")
        self.legend_strip.setStyleSheet(
            "QLabel { background: rgba(20,20,20,200); color: white; "
            "padding: 3px 10px; font-size: 9pt; border-bottom: 1px solid #444; }")
        self.legend_strip.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.legend_strip.setFixedHeight(26)

        layout_graph = QVBoxLayout()
        layout_graph.setSpacing(0)
        layout_graph.addWidget(self.scenario_tabs)
        layout_graph.addWidget(self.legend_strip)
        layout_graph.addWidget(self.view)
        container_graph = QWidget()
        container_graph.setLayout(layout_graph)
        main_layout.addWidget(container_graph)

        # Controls
        self.init_controls(control_layout)
        self._on_visibility_changed()
        self._apply_background("Чорний")

        # Animation
        self.t = 0.0
        self._paused = True
        self._is_updating = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        # self.timer.start(50) # Start timer AFTER initialization is complete

        # 3D line items — wave 1
        _w1e8 = _W1_E[:3] + (0.8,); _w1b8 = _W1_B[:3] + (0.8,)
        self.e_line  = gl.GLLinePlotItem(width=2, color=_w1e8, mode='lines',      glOptions='translucent')
        self.b_line  = gl.GLLinePlotItem(width=2, color=_w1b8, mode='lines',      glOptions='translucent')
        self.s_line  = gl.GLLinePlotItem(width=4, color=_W1_S, mode='lines',      glOptions='translucent')
        self.e_curve = gl.GLLinePlotItem(width=3, color=_W1_E, mode='line_strip', glOptions='translucent')
        self.b_curve = gl.GLLinePlotItem(width=3, color=_W1_B, mode='line_strip', glOptions='translucent')
        for item in (self.e_line, self.b_line, self.s_line, self.e_curve, self.b_curve):
            self.view.addItem(item)

        # 3D surface items — wave 1
        self.e_surf = gl.GLMeshItem(smooth=False, color=_W1_E[:3]+(0.5,), shader='shaded', glOptions='translucent')
        self.b_surf = gl.GLMeshItem(smooth=False, color=_W1_B[:3]+(0.5,), shader='shaded', glOptions='translucent')
        self.s_surf = gl.GLMeshItem(smooth=False, color=_W1_S[:3]+(0.5,), shader='shaded', glOptions='translucent')
        for surf in (self.e_surf, self.b_surf, self.s_surf):
            surf.setVisible(False)
            self.view.addItem(surf)

        # 3D line items — wave 2 (cyan/orange palette to distinguish from wave 1)
        _w2e8 = _W2_E[:3] + (0.8,); _w2b8 = _W2_B[:3] + (0.8,)
        self.e_line2  = gl.GLLinePlotItem(width=2, color=_w2e8, mode='lines',      glOptions='translucent')
        self.b_line2  = gl.GLLinePlotItem(width=2, color=_w2b8, mode='lines',      glOptions='translucent')
        self.s_line2  = gl.GLLinePlotItem(width=4, color=_W2_S, mode='lines',      glOptions='translucent')
        self.e_curve2 = gl.GLLinePlotItem(width=3, color=_W2_E, mode='line_strip', glOptions='translucent')
        self.b_curve2 = gl.GLLinePlotItem(width=3, color=_W2_B, mode='line_strip', glOptions='translucent')
        for item in (self.e_line2, self.b_line2, self.s_line2, self.e_curve2, self.b_curve2):
            item.setVisible(False)
            self.view.addItem(item)

        # Source position marker for wave 2
        self.source2_marker = gl.GLScatterPlotItem(
            pos=np.array([[0.0, 0.0, 0.0]]),
            color=(1.0, 0.85, 0.0, 1.0),
            size=14,
            pxMode=True,
        )
        self.source2_marker.setVisible(False)
        self.view.addItem(self.source2_marker)
        self.text_source2 = gl.GLTextItem(pos=(0, 0, 0), text='Дж.2', color=_txt_rgba((1.0, 0.85, 0.0, 1.0)))
        self.text_source2.setVisible(False)
        self.view.addItem(self.text_source2)

        # Direction indicator arrows: k (propagation), E-axis, B-axis
        self.k_indicator = gl.GLLinePlotItem(width=6, color=_W1_K, mode='lines', glOptions='translucent')
        self.e_indicator = gl.GLLinePlotItem(width=4, color=_W1_E, mode='lines', glOptions='translucent')
        self.b_indicator = gl.GLLinePlotItem(width=4, color=_W1_B, mode='lines', glOptions='translucent')
        self.text_k  = gl.GLTextItem(pos=(0, 0, 0), text='k',  color=_txt_rgba(_W1_K))
        for item in (self.k_indicator, self.e_indicator, self.b_indicator, self.text_k):
            self.view.addItem(item)

        self.z_steps = 200
        self.z = np.linspace(-10, 10, self.z_steps)
        self.surface_faces = self._build_surface_faces(self.z_steps, closed=False)
        self.volume_faces = self._build_surface_faces(self.z_steps, closed=True)

        # Text items for coordinate axes (colored to match axis arrows)
        self.text_x = gl.GLTextItem(pos=(12, 0, 0), text='X', color=_txt_rgba(_AXIS_X))
        self.text_y = gl.GLTextItem(pos=(0, 12, 0), text='Y', color=_txt_rgba(_AXIS_Y))
        self.text_z = gl.GLTextItem(pos=(0, 0, 12), text='Z', color=_txt_rgba(_AXIS_Z))
        # E/B direction labels — per-wave colors
        self.text_e  = gl.GLTextItem(pos=(5, 0, 0), text='E₁', color=_txt_rgba(_W1_E))
        self.text_b  = gl.GLTextItem(pos=(0, 5, 0), text='B₁', color=_txt_rgba(_W1_B))
        self.text_e2 = gl.GLTextItem(pos=(5, 0, 0), text='E₂', color=_txt_rgba(_W2_E))
        self.text_b2 = gl.GLTextItem(pos=(0, 5, 0), text='B₂', color=_txt_rgba(_W2_B))
        for txt in (self.text_x, self.text_y, self.text_z,
                    self.text_e, self.text_b, self.text_e2, self.text_b2):
            self.view.addItem(txt)
        for txt in (self.text_e, self.text_b, self.text_e2, self.text_b2):
            txt.setVisible(False)

        # Start timer after everything is initialized
        self._update_legend(0)
        self.timer.start(50)

    # ------------------------------------------------------------------
    # Static builders
    # ------------------------------------------------------------------

    @staticmethod
    def _make_axis_arrows():
        """Colored axis arrows: X/Y/Z use fixed colors distinct from wave colors. No background dependency."""
        s, hs, hw = 12, 0.8, 0.3
        # 6 points for X, 6 for Y, 10 for Z
        pts = np.array([
            [0,0,0],[s,0,0], [s,0,0],[s-hs,hw,0], [s,0,0],[s-hs,-hw,0],
            [0,0,0],[0,s,0], [0,s,0],[hw,s-hs,0], [0,s,0],[-hw,s-hs,0],
            [0,0,0],[0,0,s], [0,0,s],[hw,0,s-hs], [0,0,s],[-hw,0,s-hs],
            [0,0,s],[0,hw,s-hs], [0,0,s],[0,-hw,s-hs],
        ], dtype=float)
        colors = np.array(
            [_AXIS_X] * 6 +
            [_AXIS_Y] * 6 +
            [_AXIS_Z] * 10,
        )
        item = gl.GLLinePlotItem(pos=pts, color=colors, width=3, mode='lines', glOptions='translucent')
        return item

    @staticmethod
    def _build_surface_faces(z_steps, closed=False):
        faces = []
        n = 2 if not closed else 4 # 4 for cylinder-like
        if not closed:
            for i in range(z_steps - 1):
                idx = i * 2
                faces.append([idx,   idx+1, idx+2])
                faces.append([idx+1, idx+3, idx+2])
        else:
            # For volume representation (e.g. 4 points per Z step)
            for i in range(z_steps - 1):
                for j in range(4):
                    p1 = i * 4 + j
                    p2 = i * 4 + (j + 1) % 4
                    p3 = (i + 1) * 4 + j
                    p4 = (i + 1) * 4 + (j + 1) % 4
                    faces.append([p1, p2, p3])
                    faces.append([p2, p4, p3])
        return np.array(faces)

    @staticmethod
    def _make_wave_frame(params):
        """Return (k_hat, e1_hat, e2_hat) for the wave's propagation frame.
        e2 = k × e1, so B = (1/c)*Ex*e2 is correct under direction=+1 in calculate_fields."""
        if params.get('dir_mode', 'Базова вісь') == 'Довільний вектор':
            v = np.asarray(params.get('dir_vec', [0., 0., 1.]), dtype=float)
            nrm = np.linalg.norm(v)
            k = v / nrm if nrm > 1e-9 else np.array([0., 0., 1.])
            ref = np.array([1., 0., 0.]) if abs(k[0]) < 0.9 else np.array([0., 1., 0.])
            e1 = np.cross(ref, k); e1 /= np.linalg.norm(e1)
            e2 = np.cross(k, e1)
            return k, e1, e2
        else:
            return _BASE_FRAMES[params.get('prop_dir', '+Z')]

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------

    def init_controls(self, layout):
        # Visibility controls
        vis_ctrl_group = QGroupBox("Відображення панелей")
        vis_ctrl_layout = QVBoxLayout()
        
        self.chk_wave = QCheckBox("Параметри хвилі")
        self.chk_viz = QCheckBox("Візуалізація")
        self.chk_mod = QCheckBox("Модуляція")
        self.chk_data = QCheckBox("Дані та легенда")
        
        # Initial states (mimic "Basic parameters" set)
        self.chk_wave.setChecked(True)
        self.chk_viz.setChecked(False)
        self.chk_mod.setChecked(False)
        self.chk_data.setChecked(False)
        
        for chk in [self.chk_wave, self.chk_viz, self.chk_mod, self.chk_data]:
            chk.stateChanged.connect(self._on_visibility_changed)
            vis_ctrl_layout.addWidget(chk)
            
        vis_ctrl_group.setLayout(vis_ctrl_layout)
        layout.addWidget(vis_ctrl_group)

        # Wave parameters
        self.wave_group = QGroupBox("Параметри хвилі")
        wave_layout = QVBoxLayout()

        wave_layout.addWidget(QLabel("Поляризація:"))
        self.pol_combo = QComboBox()
        self.pol_combo.addItems(["linear", "circular_right", "circular_left", "elliptical"])
        self.pol_combo.currentTextChanged.connect(self._on_param_changed)
        wave_layout.addWidget(self.pol_combo)

        wave_layout.addWidget(QLabel("Амплітуда E = A (В/м):"))
        self.amp_spin = QDoubleSpinBox()
        self.amp_spin.setRange(0.1, 100.0)
        self.amp_spin.setValue(1.0)
        self.amp_spin.setSingleStep(0.5)
        self.amp_spin.valueChanged.connect(self._on_param_changed)
        wave_layout.addWidget(self.amp_spin)

        wave_layout.addWidget(QLabel("Зсув по фазі (°):"))
        phase_row = QHBoxLayout()
        self.phase_spin = QDoubleSpinBox()
        self.phase_spin.setRange(0, 360)
        self.phase_spin.setValue(0)
        self.phase_spin.valueChanged.connect(self._on_param_changed)
        phase_row.addWidget(self.phase_spin)
        self.phase_preset_combo = QComboBox()
        self.phase_preset_combo.addItems(["Custom", "0", "90", "180", "270"])
        self.phase_preset_combo.currentIndexChanged.connect(self._on_phase_preset_changed)
        phase_row.addWidget(self.phase_preset_combo)
        wave_layout.addLayout(phase_row)

        wave_layout.addWidget(QLabel("Довжина (λ):"))
        self.n_periods_spin = QSpinBox()
        self.n_periods_spin.setRange(1, 10)
        self.n_periods_spin.setValue(2)
        self.n_periods_spin.valueChanged.connect(self._on_param_changed)
        wave_layout.addWidget(self.n_periods_spin)

        wave_layout.addWidget(QLabel("Напрямок:"))
        self.dir_mode_combo = QComboBox()
        self.dir_mode_combo.addItems(["Базова вісь", "Довільний вектор"])
        self.dir_mode_combo.currentTextChanged.connect(self._on_dir_mode_changed)
        wave_layout.addWidget(self.dir_mode_combo)

        self.wave_prop_dir_combo = QComboBox()
        self.wave_prop_dir_combo.addItems(["+Z", "-Z", "+X", "-X", "+Y", "-Y"])
        self.wave_prop_dir_combo.currentTextChanged.connect(self._on_param_changed)
        wave_layout.addWidget(self.wave_prop_dir_combo)

        self.dir_vec_widget = QWidget()
        dv_layout = QHBoxLayout(self.dir_vec_widget)
        dv_layout.setContentsMargins(0, 0, 0, 0)
        for lbl_txt, attr, default in [("X:", 'dir_x_spin', 0.), ("Y:", 'dir_y_spin', 0.), ("Z:", 'dir_z_spin', 1.)]:
            dv_layout.addWidget(QLabel(lbl_txt))
            sp = QDoubleSpinBox()
            sp.setRange(-5.0, 5.0); sp.setSingleStep(0.1); sp.setDecimals(1); sp.setValue(default)
            sp.setFixedWidth(52)
            sp.valueChanged.connect(self._on_param_changed)
            setattr(self, attr, sp)
            dv_layout.addWidget(sp)
        self.dir_vec_widget.setVisible(False)
        wave_layout.addWidget(self.dir_vec_widget)

        self.wave_group.setLayout(wave_layout)
        layout.addWidget(self.wave_group)

        # Source position (Tab 2 only — direction is in wave_group above)
        self.source_group = QGroupBox("Позиція джерела 2")
        src_layout = QVBoxLayout()

        src_layout.addWidget(QLabel("Координати (×λ):"))
        pos_row = QHBoxLayout()
        for lbl, attr, default_val in [("X:", 'x0_spin', 0.0), ("Y:", 'y0_spin', 0.0), ("Z:", 'z0_spin', 3.0)]:
            pos_row.addWidget(QLabel(lbl))
            spin = QDoubleSpinBox()
            spin.setRange(-10.0, 10.0)
            spin.setValue(default_val)
            spin.setSingleStep(0.25)
            spin.setFixedWidth(70)
            spin.valueChanged.connect(self._on_source_changed)
            setattr(self, attr, spin)
            pos_row.addWidget(spin)
        src_layout.addLayout(pos_row)

        self.source_group.setLayout(src_layout)
        self.source_group.setVisible(False)
        layout.addWidget(self.source_group)

        # Visualization
        self.viz_group = QGroupBox("Візуалізація")
        viz_layout = QVBoxLayout()

        viz_layout.addWidget(QLabel("Проекція:"))
        self.proj_combo = QComboBox()
        self.proj_combo.addItems(["3D", "XY (Top)", "XZ (Side)", "YZ (Front)"])
        self.proj_combo.currentIndexChanged.connect(self._on_projection_changed)
        viz_layout.addWidget(self.proj_combo)

        viz_layout.addWidget(QLabel("Пікселів на довжину хвилі:"))
        self.pxw_spin = QSpinBox()
        self.pxw_spin.setRange(10, 2000)
        self.pxw_spin.setValue(100)
        self.pxw_spin.setSingleStep(10)
        self.pxw_spin.valueChanged.connect(self._on_pxw_changed)
        viz_layout.addWidget(self.pxw_spin)

        viz_layout.addWidget(QLabel("Відображення компонент:"))
        self.comp_combo = QComboBox()
        self.comp_combo.addItems(["Узагальнений підхід", "Електрична (E)", "Магнітна (B)"])
        viz_layout.addWidget(self.comp_combo)

        viz_layout.addWidget(QLabel("Представлення:"))
        self.repr_combo = QComboBox()
        self.repr_combo.addItems(["Linear (3D)", "Surface (3D)"])
        self.repr_combo.currentTextChanged.connect(self._on_repr_changed)
        viz_layout.addWidget(self.repr_combo)

        viz_layout.addWidget(QLabel("Фон:"))
        self.bg_combo = QComboBox()
        self.bg_combo.addItems(list(_BG_COLORS.keys()))
        self.bg_combo.currentTextChanged.connect(self._apply_background)
        viz_layout.addWidget(self.bg_combo)

        self.viz_group.setLayout(viz_layout)
        layout.addWidget(self.viz_group)

        # Modulation
        self.mod_group = QGroupBox("Модуляція")
        mod_layout = QVBoxLayout()

        mod_layout.addWidget(QLabel("Тип модуляції:"))
        self.mod_combo = QComboBox()
        self.mod_combo.addItems(["none", "AM", "FM", "AFM"])
        self.mod_combo.currentTextChanged.connect(self._on_param_changed)
        mod_layout.addWidget(self.mod_combo)

        mod_layout.addWidget(QLabel("Частота модуляції (Гц):"))
        self.mod_freq_spin = QDoubleSpinBox()
        self.mod_freq_spin.setRange(0.01, 1e9)
        self.mod_freq_spin.setValue(0.1)
        self.mod_freq_spin.setSingleStep(0.1)
        self.mod_freq_spin.valueChanged.connect(self._on_param_changed)
        mod_layout.addWidget(self.mod_freq_spin)

        mod_layout.addWidget(QLabel("Індекс модуляції:"))
        self.mod_index_spin = QDoubleSpinBox()
        self.mod_index_spin.setRange(0.0, 1.0)
        self.mod_index_spin.setValue(0.5)
        self.mod_index_spin.setSingleStep(0.1)
        self.mod_index_spin.valueChanged.connect(self._on_param_changed)
        mod_layout.addWidget(self.mod_index_spin)

        self.mod_group.setLayout(mod_layout)
        layout.addWidget(self.mod_group)

        # Data & legend
        self.data_group = QGroupBox("Дані та легенда")
        data_layout = QVBoxLayout()

        self.legend_label = QLabel("")   # filled by _update_legend
        data_layout.addWidget(self.legend_label)

        data_layout.addWidget(QLabel("Компоненти полів (період λ/10):"))
        self.data_table = QTableWidget(11, 6)
        self.data_table.setHorizontalHeaderLabels(["Coord", "E₁", "E₂", "H₁", "H₂", "Sz"])
        
        vh = self.data_table.verticalHeader()
        if vh is not None:
            vh.setVisible(False)
            vh.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            
        hh = self.data_table.horizontalHeader()
        if hh is not None:
            hh.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            hh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        
        self.data_table.setFixedHeight(350) 
        self.data_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.data_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        data_layout.addWidget(self.data_table)
        self.data_group.setLayout(data_layout)
        layout.addWidget(self.data_group)

        self.pause_button = QPushButton("⏸ Пауза")
        self.pause_button.setCheckable(True)
        self.pause_button.setChecked(True)
        self.pause_button.toggled.connect(self._on_pause_toggled)
        self.pause_button.setText("▶ Відтворити")
        layout.addWidget(self.pause_button)

        self.reset_button = QPushButton("⟲ Повернутись до початкових налаштувань")
        self.reset_button.clicked.connect(self._on_reset)
        layout.addWidget(self.reset_button)

        self.save_button = QPushButton("💾 Зберегти малюнок")
        self.save_button.clicked.connect(self._on_save_image)
        layout.addWidget(self.save_button)

        # Parameters save / load
        params_io_group = QGroupBox("Файл параметрів")
        params_io_layout = QVBoxLayout()
        io_row = QHBoxLayout()
        self.save_params_btn = QPushButton("💾 Зберегти")
        self.load_params_btn = QPushButton("📂 Завантажити")
        self.save_params_btn.clicked.connect(self._on_save_params)
        self.load_params_btn.clicked.connect(self._on_load_params)
        io_row.addWidget(self.save_params_btn)
        io_row.addWidget(self.load_params_btn)
        params_io_layout.addLayout(io_row)
        self.params_status = QLabel("—")
        self.params_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.params_status.setWordWrap(True)
        self.params_status.setStyleSheet("font-size: 8pt; color: gray;")
        params_io_layout.addWidget(self.params_status)
        params_io_group.setLayout(params_io_layout)
        layout.addWidget(params_io_group)

        self.exit_button = QPushButton("Вихід")
        self.exit_button.clicked.connect(self.close)
        layout.addWidget(self.exit_button)
        layout.addStretch()

    # ------------------------------------------------------------------
    # Slot handlers
    # ------------------------------------------------------------------

    def _update_legend(self, idx):
        """Rebuild legend_strip and data-panel legend to match active scenario."""
        if not hasattr(self, 'legend_strip') or not hasattr(self, 'legend_label'):
            return
        e1h = _c2hex(*_W1_E); b1h = _c2hex(*_W1_B); s1h = _c2hex(*_W1_S)
        e2h = _c2hex(*_W2_E); b2h = _c2hex(*_W2_B); s2h = _c2hex(*_W2_S)
        xh = _c2hex(*_AXIS_X); yh = _c2hex(*_AXIS_Y); zh = _c2hex(*_AXIS_Z)
        axes = (f"<font color='{xh}'>X</font>&nbsp;"
                f"<font color='{yh}'>Y</font>&nbsp;"
                f"<font color='{zh}'>Z</font>")
        if idx == 0:
            strip = (f"<b>Хвиля&nbsp;1:</b>&nbsp;&nbsp;"
                     f"<font color='{e1h}'>■&nbsp;E</font>&nbsp;&nbsp;"
                     f"<font color='{b1h}'>■&nbsp;B</font>"
                     f"&nbsp;&nbsp;|&nbsp;&nbsp;Осі:&nbsp;{axes}")
            panel = (f"<b>Хвиля 1:</b><br>"
                     f"<font color='{e1h}'>■ E — електричне поле</font><br>"
                     f"<font color='{b1h}'>■ B — магнітне поле</font><br>"
                     "<font color='gray'>─ Осі: X(фіолет.) Y(малин.) Z(корич.)</font>")
        elif idx == 1:
            strip = (f"<b>Хвиля&nbsp;2:</b>&nbsp;&nbsp;"
                     f"<font color='{e2h}'>■&nbsp;E</font>&nbsp;&nbsp;"
                     f"<font color='{b2h}'>■&nbsp;B</font>&nbsp;&nbsp;"
                     f"<font color='{s2h}'>■&nbsp;S</font>"
                     f"&nbsp;&nbsp;|&nbsp;&nbsp;Осі:&nbsp;{axes}")
            panel = (f"<b>Хвиля 2:</b><br>"
                     f"<font color='{e2h}'>■ E — електричне поле</font><br>"
                     f"<font color='{b2h}'>■ B — магнітне поле</font><br>"
                     f"<font color='{s2h}'>■ S — вектор Пойнтінга</font><br>"
                     "<font color='gray'>─ Осі: X(фіолет.) Y(малин.) Z(корич.)</font>")
        else:
            strip = (f"<b>Хвиля&nbsp;1:</b>&nbsp;"
                     f"<font color='{e1h}'>■E₁</font>&nbsp;"
                     f"<font color='{b1h}'>■B₁</font>&nbsp;"
                     f"<font color='{s1h}'>■S₁</font>"
                     f"&nbsp;&nbsp;<b>Хвиля&nbsp;2:</b>&nbsp;"
                     f"<font color='{e2h}'>■E₂</font>&nbsp;"
                     f"<font color='{b2h}'>■B₂</font>&nbsp;"
                     f"<font color='{s2h}'>■S₂</font>"
                     f"&nbsp;&nbsp;|&nbsp;&nbsp;Осі:&nbsp;{axes}")
            panel = (f"<b>Суперпозиція (Хвиля 1 + Хвиля 2):</b><br>"
                     f"<font color='{e1h}'>■</font>/<font color='{e2h}'>■ E — електричне поле</font><br>"
                     f"<font color='{b1h}'>■</font>/<font color='{b2h}'>■ B — магнітне поле</font><br>"
                     f"<font color='{s1h}'>■</font>/<font color='{s2h}'>■ S — вектор Пойнтінга</font><br>"
                     "<font color='gray'>─ Осі: X(фіолет.) Y(малин.) Z(корич.)</font>")
        self.legend_strip.setText(strip)
        self.legend_label.setText(panel)

    def _on_pause_toggled(self, checked):
        self._paused = checked
        if checked:
            self.pause_button.setText("▶ Відтворити")
        else:
            self.pause_button.setText("⏸ Пауза")

    def _on_visibility_changed(self):
        self.wave_group.setVisible(self.chk_wave.isChecked())
        self.viz_group.setVisible(self.chk_viz.isChecked())
        self.mod_group.setVisible(self.chk_mod.isChecked())
        self.data_group.setVisible(self.chk_data.isChecked())

    def _on_phase_preset_changed(self, index):
        if index > 0:
            self.phase_spin.setValue(float(self.phase_preset_combo.currentText()))

    def _on_scenario_changed(self, index):
        if index == 0:
            self._apply_wave1_camera()
        elif index == 1:
            self._apply_wave2_camera()
        else:
            self._apply_default_camera()

        # Tab 2 (index==2) = superposition — no editable params
        editable = index < 2
        self.wave_group.setEnabled(editable)
        self.mod_group.setEnabled(editable)
        # Source group visible only for Tab 2 (index==1)
        self.source_group.setVisible(index == 1)

        self._update_legend(index)
        if not editable:
            self.update_plot()
            return

        params = self.scenario_params[index]
        self._is_updating = True
        try:
            self.pol_combo.setCurrentText(params['pol'])
            self.amp_spin.setValue(params.get('amplitude', 1.0))
            self.phase_spin.setValue(params['phase'])
            self.mod_combo.setCurrentText(params['mod_type'])
            self.mod_freq_spin.setValue(params['mod_freq'])
            self.mod_index_spin.setValue(params['mod_index'])
            self.n_periods_spin.setValue(int(params.get('n_periods', 2)))
            self.dir_mode_combo.setCurrentText(params.get('dir_mode', 'Базова вісь'))
            self.wave_prop_dir_combo.setCurrentText(params['prop_dir'])
            dv = params.get('dir_vec', [0., 0., 1.])
            self.dir_x_spin.setValue(dv[0]); self.dir_y_spin.setValue(dv[1]); self.dir_z_spin.setValue(dv[2])
            if index == 1:
                self.x0_spin.setValue(params['x0'])
                self.y0_spin.setValue(params['y0'])
                self.z0_spin.setValue(params['z0'])
        finally:
            self._is_updating = False
        self._sync_dir_visibility()
        self._update_legend(index)
        self.update_plot()

    def _on_param_changed(self):
        if self._is_updating:
            return
        index = self.scenario_tabs.currentIndex()
        if index >= 2:
            return
        p = self.scenario_params[index]
        p['pol']       = self.pol_combo.currentText()
        p['amplitude'] = self.amp_spin.value()
        p['phase']     = self.phase_spin.value()
        p['mod_type']  = self.mod_combo.currentText()
        p['mod_freq']  = self.mod_freq_spin.value()
        p['mod_index'] = self.mod_index_spin.value()
        p['n_periods'] = self.n_periods_spin.value()
        p['dir_mode']  = self.dir_mode_combo.currentText()
        p['prop_dir']  = self.wave_prop_dir_combo.currentText()
        p['dir_vec']   = [self.dir_x_spin.value(), self.dir_y_spin.value(), self.dir_z_spin.value()]
        self.update_plot()

    def _on_dir_mode_changed(self, text):
        self._sync_dir_visibility()
        self._on_param_changed()

    def _sync_dir_visibility(self):
        is_custom = self.dir_mode_combo.currentText() == "Довільний вектор"
        self.wave_prop_dir_combo.setVisible(not is_custom)
        self.dir_vec_widget.setVisible(is_custom)

    def _on_source_changed(self):
        if self._is_updating:
            return
        p = self.scenario_params[1]
        p['x0'] = self.x0_spin.value()
        p['y0'] = self.y0_spin.value()
        p['z0'] = self.z0_spin.value()
        self.update_plot()

    def _on_repr_changed(self, text):
        self.update_plot()

    # ------------------------------------------------------------------
    # Parameters save / load
    # ------------------------------------------------------------------

    _ALLOWED = {
        'pol':      ['linear', 'circular_right', 'circular_left', 'elliptical'],
        'mod_type': ['none', 'AM', 'FM', 'AFM'],
        'prop_dir': ['+Z', '-Z', '+X', '-X', '+Y', '-Y'],
        'dir_mode': ['Базова вісь', 'Довільний вектор'],
    }
    _RANGES = {   # (min, max) inclusive
        'phase':     (0.0,  360.0),
        'amplitude': (0.1,  100.0),
        'mod_freq':  (0.01, 1e9),
        'mod_index': (0.0,  1.0),
        'n_periods': (1,    10),
    }
    _POS_RANGE = (-10.0, 10.0)

    def _collect_params(self):
        """Serialize all user-configurable state into a plain dict."""
        return {
            "schema_version": 1,
            "waves": {
                "0": dict(self.scenario_params[0]),
                "1": dict(self.scenario_params[1]),
            },
            "visualization": {
                "comp_mode":  self.comp_combo.currentText(),
                "repr_mode":  self.repr_combo.currentText(),
                "background": self.bg_combo.currentText(),
                "projection": self.proj_combo.currentText(),
                "pixels_per_wavelength": self.pxw_spin.value(),
            },
        }

    @staticmethod
    def _default_params():
        """Initial wave/visualization configuration — same shape as _collect_params()."""
        return {
            "schema_version": 1,
            "waves": {
                "0": {'pol': 'linear', 'phase': 0.0, 'mod_type': 'none', 'mod_freq': 0.1, 'mod_index': 0.5,
                      'n_periods': 2, 'prop_dir': '+Z', 'dir_mode': 'Базова вісь', 'dir_vec': [0., 0., 1.]},
                "1": {'pol': 'linear', 'phase': 0.0, 'mod_type': 'none', 'mod_freq': 0.1, 'mod_index': 0.5,
                      'n_periods': 2, 'prop_dir': '-Z', 'dir_mode': 'Базова вісь', 'dir_vec': [0., 0., -1.],
                      'x0': 0.0, 'y0': 0.0, 'z0': 3.0},
            },
            "visualization": {
                "comp_mode":  "Узагальнений підхід",
                "repr_mode":  "Linear (3D)",
                "background": "Чорний",
                "projection": "3D",
                "pixels_per_wavelength": 100,
            },
        }

    def _on_reset(self):
        """Restore wave/visualization settings to their initial defaults."""
        self._apply_params(self._default_params())
        self.t = 0.0
        self.scenario_tabs.setCurrentIndex(0)

    def _validate_params(self, data):
        """Return list of error strings (empty list = data is valid)."""
        errs = []
        if not isinstance(data, dict):
            return ["Файл не містить JSON-об'єкт"]
        if data.get('schema_version') != 1:
            return [f"Невідома версія файлу (schema_version = {data.get('schema_version')!r})"]

        waves = data.get('waves')
        if not isinstance(waves, dict):
            return ["Відсутня секція 'waves'"]

        for wk in ('0', '1'):
            w = waves.get(wk)
            if not isinstance(w, dict):
                errs.append(f"Хвиля '{wk}': секцію не знайдено"); continue

            # Enum fields
            for key, allowed in self._ALLOWED.items():
                val = w.get(key)
                if val not in allowed:
                    errs.append(f"Хвиля {wk} › {key} = {val!r} — має бути одне з {allowed}")

            # Numeric range fields
            for key, (lo, hi) in self._RANGES.items():
                val = w.get(key)
                try:
                    v = float(val)
                    if not (lo <= v <= hi):
                        errs.append(f"Хвиля {wk} › {key} = {val!r} — поза межами [{lo}, {hi}]")
                except (TypeError, ValueError):
                    errs.append(f"Хвиля {wk} › {key} = {val!r} — не є числом")

            # dir_vec
            dv = w.get('dir_vec')
            if not (isinstance(dv, list) and len(dv) == 3 and
                    all(isinstance(x, (int, float)) for x in dv)):
                errs.append(f"Хвиля {wk} › dir_vec — має бути список з 3 чисел, отримано: {dv!r}")

        # Wave 2 position fields
        w2 = waves.get('1', {})
        lo, hi = self._POS_RANGE
        for key in ('x0', 'y0', 'z0'):
            val = w2.get(key)
            try:
                v = float(val)
                if not (lo <= v <= hi):
                    errs.append(f"Хвиля 1 › {key} = {val!r} — поза межами [{lo}, {hi}]")
            except (TypeError, ValueError):
                errs.append(f"Хвиля 1 › {key} = {val!r} — не є числом")

        return errs

    def _apply_params(self, data):
        """Restore scenario_params and UI widgets from a validated dict."""
        waves = data['waves']
        for wk in ('0', '1'):
            w = waves[wk]; p = self.scenario_params[int(wk)]
            p.update({
                'pol':       w['pol'],
                'amplitude': float(w.get('amplitude', 1.0)),
                'phase':     float(w['phase']),
                'mod_type':  w['mod_type'],
                'mod_freq':  float(w['mod_freq']),
                'mod_index': float(w['mod_index']),
                'n_periods': int(w['n_periods']),
                'prop_dir':  w['prop_dir'],
                'dir_mode':  w['dir_mode'],
                'dir_vec':   [float(x) for x in w['dir_vec']],
            })
        w2 = waves['1']
        self.scenario_params[1].update({
            'x0': float(w2['x0']), 'y0': float(w2['y0']), 'z0': float(w2['z0']),
        })

        viz = data.get('visualization', {})
        self._is_updating = True
        try:
            def _set(combo, key):
                v = viz.get(key, '')
                items = [combo.itemText(i) for i in range(combo.count())]
                if v in items:
                    combo.setCurrentText(v)
            _set(self.comp_combo,  'comp_mode')
            _set(self.repr_combo,  'repr_mode')
            _set(self.proj_combo,  'projection')
            bg = viz.get('background', '')
            if bg in _BG_COLORS:
                self.bg_combo.setCurrentText(bg)
            pxw = viz.get('pixels_per_wavelength')
            if isinstance(pxw, (int, float)) and self.pxw_spin.minimum() <= pxw <= self.pxw_spin.maximum():
                self.pxw_spin.setValue(int(pxw))
        finally:
            self._is_updating = False

        # Full UI re-sync for whichever tab is active
        self._on_scenario_changed(self.scenario_tabs.currentIndex())

    def _on_save_params(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Зберегти параметри", "", "JSON-файли (*.json);;Усі файли (*)")
        if not path:
            return
        if not path.lower().endswith('.json'):
            path += '.json'
        try:
            data = self._collect_params()
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.params_status.setText(f"✓ {pathlib.Path(path).name}")
            self.params_status.setStyleSheet("font-size: 8pt; color: #33dd33;")
        except Exception as exc:
            self.params_status.setText(f"Помилка збереження: {exc}")
            self.params_status.setStyleSheet("font-size: 8pt; color: #ff5555;")

    def _on_load_params(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Завантажити параметри", "", "JSON-файли (*.json);;Усі файли (*)")
        if not path:
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as exc:
            QMessageBox.critical(self, "Помилка читання",
                                 f"Не вдалося прочитати файл:\n{exc}")
            self.params_status.setText("Помилка читання")
            self.params_status.setStyleSheet("font-size: 8pt; color: #ff5555;")
            return

        errs = self._validate_params(data)
        if errs:
            QMessageBox.warning(self, "Некоректні дані",
                                "Файл містить помилки:\n• " + "\n• ".join(errs))
            self.params_status.setText("Помилка перевірки даних")
            self.params_status.setStyleSheet("font-size: 8pt; color: #ff5555;")
            return

        self._apply_params(data)
        self.params_status.setText(f"✓ {pathlib.Path(path).name}")
        self.params_status.setStyleSheet("font-size: 8pt; color: #33dd33;")

    def _on_save_image(self):
        """Save screenshot of the 3D view + data table."""
        # Grab plot image
        plot_pixmap = self.view.grab()

        # Grab table image
        table_pixmap = self.data_table.grab()
        
        # Combine them
        combined_width = max(plot_pixmap.width(), table_pixmap.width())
        combined_height = plot_pixmap.height() + table_pixmap.height() + 40
        
        final_pixmap = QPixmap(combined_width, combined_height)
        final_pixmap.fill(QColor("white"))
        
        painter = QPainter(final_pixmap)
        painter.drawPixmap(0, 0, plot_pixmap)
        painter.drawPixmap(0, plot_pixmap.height() + 20, table_pixmap)
        
        # Add labels
        painter.setPen(QColor("black"))
        painter.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        painter.drawText(10, plot_pixmap.height() + 15, "Таблиця розрахунків:")
        
        painter.end()
        
        path, _ = QFileDialog.getSaveFileName(self, "Зберегти малюнок", "", "Images (*.png *.jpg)")
        if path:
            final_pixmap.save(path)

    def _distance_for_pxw(self, px_per_wavelength):
        """Camera distance such that one wavelength (= 1 unit in the λ-normalized
        3D view) spans `px_per_wavelength` screen pixels, given the view's current
        width and field of view."""
        width = max(self.view.width(), 1)
        fov = math.radians(self.view.opts.get('fov', 60))
        return width / (px_per_wavelength * 2 * math.tan(fov / 2))

    def _pixels_per_unit(self):
        """Current on-screen pixels per 3D-view unit (= per λ), given the live
        camera distance — used to thin out axis tick labels so they don't overlap."""
        width = max(self.view.width(), 1)
        dist = max(self.view.opts.get('distance', 1.0), 1e-9)
        fov = math.radians(self.view.opts.get('fov', 60))
        return width / (dist * 2 * math.tan(fov / 2))

    def _on_pxw_changed(self):
        self.view.opts['distance'] = self._distance_for_pxw(self.pxw_spin.value())
        self.view.update()

    def _on_projection_changed(self, index):
        dist = self.view.opts['distance']
        if index == 0:    # 3D perspective
            self.view.setCameraPosition(distance=dist, elevation=30, azimuth=45)
        elif index == 1:  # XY top view
            self.view.setCameraPosition(distance=dist, elevation=90, azimuth=0)
        elif index == 2:  # XZ side view
            self.view.setCameraPosition(distance=dist, elevation=0, azimuth=0)
        elif index == 3:  # YZ front view
            self.view.setCameraPosition(distance=dist, elevation=0, azimuth=90)

    def _apply_wave1_camera(self):
        """Fixed camera for Tab 1: X up, Z left-to-right, Y toward the viewer.
        The look-at center is offset along +Z so the origin sits a quarter of the
        Z-axis length left of screen-middle, leaving more room for the wave."""
        self.view.opts['rotationMethod'] = 'quaternion'
        self.view.opts['rotation'] = _WAVE1_CAMERA_ROTATION
        z_axis_len = self.scenario_params[0].get('n_periods', 2) + 2
        self.view.opts['center'] = pg.Vector(0, 0, z_axis_len * 0.25)
        self.view.update()

    def _apply_wave2_camera(self):
        """Fixed camera for Tab 2: X down, Y away from the viewer, Z right-to-left.
        The look-at center is offset along -Z so the origin sits two-thirds of the
        screen width to the right, leaving room on the left for the wave to develop."""
        self.view.opts['rotationMethod'] = 'quaternion'
        self.view.opts['rotation'] = _WAVE2_CAMERA_ROTATION
        z_axis_len = self.scenario_params[1].get('n_periods', 2) + 2
        self.view.opts['center'] = pg.Vector(0, 0, -z_axis_len * (2.0 / 3.0))
        self.view.update()

    def _apply_default_camera(self):
        """Restore the standard elevation/azimuth camera (driven by proj_combo)."""
        self.view.opts['rotationMethod'] = 'euler'
        self._on_projection_changed(self.proj_combo.currentIndex())

    def _apply_background(self, name: str):
        cfg = _BG_COLORS.get(name, _BG_COLORS["Чорний"])
        self.view.setBackgroundColor(cfg["gl"])
        self.grid.setColor(cfg["grid"])

        self._txt_color = _txt_rgba((0.0, 0.0, 0.0, 1.0) if name == "Білий" else (1.0, 1.0, 1.0, 1.0))
        tick_color = (0.3, 0.3, 0.3, 0.7) if name == "Білий" else (0.5, 0.5, 0.5, 0.6)
        self.tick_marks.setData(color=tick_color)
        # Axis label colors are fixed per-axis (X/Y/Z) — no background adaptation needed

    # ------------------------------------------------------------------
    # Axis permutation: local (t1, t2, prop) → world (x, y, z)
    # ------------------------------------------------------------------

    def _get_perm(self):
        axis = self.prop_axis_combo.currentText()
        if axis == "X":
            return [2, 0, 1]   # prop→x, t1→y, t2→z
        if axis == "Y":
            return [0, 2, 1]   # t1→x, prop→y, t2→z
        return [0, 1, 2]       # t1→x, t2→y, prop→z  (Z default)

    # ------------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------------

    def _format_value(self, val, fmt):
        """Format value and remove negative sign from zero."""
        formatted = f"{val:{fmt}}"
        # If the value is close to zero, it might be formatted as "-0.00"
        if float(formatted) == 0:
            return f"{0.0:{fmt}}"
        return formatted

    def update_plot(self):
        if self._is_updating:
            return
        if not self.isVisible(): # Do not update if window is not visible or closing
            return
        self._is_updating = True
        try:
            freq      = 1.0  # Фіксована частота
            repr_mode = self.repr_combo.currentText()
            comp_mode = self.comp_combo.currentText()
            scenario_idx = self.scenario_tabs.currentIndex()

            if not self._paused:
                dt = 0.05 / freq if freq > 0 else 0.05
                self.t += dt

            wavelength = C / freq

            # --- wave 1: source at origin, configurable direction ---
            p1 = self.scenario_params[0]
            amp1 = p1.get('amplitude', 1.0)
            k1, e1_1, e2_1 = self._make_wave_frame(p1)
            M1 = np.column_stack([e1_1, e2_1, k1])   # local(t1,t2,prop) → world
            off1 = np.zeros(3)
            z_wave1 = np.linspace(0, int(p1.get('n_periods', 2)) * wavelength, self.z_steps)

            # --- wave 2: configurable source position + direction ---
            p2 = self.scenario_params[1]
            amp2 = p2.get('amplitude', 1.0)
            k2, e1_2, e2_2 = self._make_wave_frame(p2)
            M2 = np.column_stack([e1_2, e2_2, k2])
            x0_2 = p2['x0'] * wavelength
            y0_2 = p2['y0'] * wavelength
            z0_2 = p2['z0'] * wavelength
            off2 = np.array([x0_2, y0_2, z0_2], dtype=float)
            z_wave2 = np.linspace(0, int(p2.get('n_periods', 2)) * wavelength, self.z_steps)

            # Global coordinate-system size covers both waves' extents
            n1, n2 = int(p1.get('n_periods', 2)), int(p2.get('n_periods', 2))
            z_max = max(n1 * wavelength,
                        np.linalg.norm(off2) + n2 * wavelength) * 1.15

            # 3D-view positions are normalized to λ (not raw SI meters, which at 1 Hz means
            # λ=3e8 m): at that magnitude, the GPU-rendered axis lines (float32 vertex buffers)
            # and the CPU-rendered GLTextItem labels (float32 QMatrix4x4) round differently,
            # visibly detaching the X/Y/Z labels from their axis tips.
            z_max_v   = z_max / wavelength
            off1_v    = off1 / wavelength
            off2_v    = off2 / wavelength
            z_wave1_v = z_wave1 / wavelength
            z_wave2_v = z_wave2 / wavelength

            # Fixed physical→view-unit scale factors for E/B/S deflection (3D-view units
            # per V/m, etc.). These must NOT depend on amp1/amp2 — earlier they were
            # divided by max(amp1, amp2), which silently canceled out whichever wave
            # was largest, so raising that wave's amplitude had no visible effect.
            # Calibrated so amp = 1 V/m gives a modest, non-dominating deflection
            # relative to the Z (propagation) axis.
            vis_scale = z_max_v * 0.15
            scale_s   = z_max_v * 0.4 * MU0 * C

            # direction=1.0 always — sign is baked into k_hat
            Ex1, Ey1, Bx1, By1 = calculate_fields(
                z_wave1, self.t, frequency=freq, amplitude=amp1,
                polarization=p1['pol'], phase_shift=p1['phase'],
                modulation_type=p1['mod_type'], mod_freq=p1['mod_freq'], mod_index=p1['mod_index'],
                z_offset=0.0, direction=1.0,
            )
            Ex2, Ey2, Bx2, By2 = calculate_fields(
                z_wave2, self.t, frequency=freq, amplitude=amp2,
                polarization=p2['pol'], phase_shift=p2['phase'],
                modulation_type=p2['mod_type'], mod_freq=p2['mod_freq'], mod_index=p2['mod_index'],
                z_offset=0.0, direction=1.0,
            )

            # World-frame X/Y components for table and 2D (correct sign for all wave directions)
            E1xw = Ex1 * e1_1[0] + Ey1 * e2_1[0]
            E1yw = Ex1 * e1_1[1] + Ey1 * e2_1[1]
            H1xw = (Bx1 * e1_1[0] + By1 * e2_1[0]) / MU0
            H1yw = (Bx1 * e1_1[1] + By1 * e2_1[1]) / MU0
            _, _, _Sz1k = calculate_poynting(Ex1, Ey1, Bx1, By1)
            Sz1w = _Sz1k * k1[2]   # world-Z component: negative for -Z waves

            E2xw = Ex2 * e1_2[0] + Ey2 * e2_2[0]
            E2yw = Ex2 * e1_2[1] + Ey2 * e2_2[1]
            H2xw = (Bx2 * e1_2[0] + By2 * e2_2[0]) / MU0
            H2yw = (Bx2 * e1_2[1] + By2 * e2_2[1]) / MU0
            _, _, _Sz2k = calculate_poynting(Ex2, Ey2, Bx2, By2)
            Sz2w = _Sz2k * k2[2]

            if scenario_idx == 0:
                tE1, tE2, tH1, tH2, tSz = E1xw, E1yw, H1xw, H1yw, Sz1w
                z_table = z_wave1
                n_act   = int(p1['n_periods'])
            elif scenario_idx == 1:
                tE1, tE2, tH1, tH2, tSz = E2xw, E2yw, H2xw, H2yw, Sz2w
                z_table = z_wave2
                n_act   = int(p2['n_periods'])
            else:
                tE1 = E1xw + E2xw
                tE2 = E1yw + E2yw
                tH1 = H1xw + H2xw
                tH2 = H1yw + H2yw
                tSz = Sz1w + Sz2w
                z_table = z_wave1
                n_act   = int(p1['n_periods'])

            # --- coordinate axis arrows (fixed colors, distinct from wave colors) ---
            # X/Y are sized to the field actually drawn (2× its peak excursion); Z is sized to
            # the configured wave length plus a 2λ margin — each axis fits its own data instead
            # of sharing one oversized cube.
            amp_extent = max(amp1, amp2, 1e-12) * vis_scale   # peak |E|/|B| reach actually drawn
            s_x = s_y = 2 * amp_extent
            # Tab 2: the Z arrow/ticks/label are drawn toward -Z so they run right-to-left
            # on screen under the fixed Tab-2 camera (wave 2's own curve already does this
            # naturally, since its default propagation direction is -Z).
            s_z = (n_act + 2) * (-1.0 if scenario_idx == 1 else 1.0)
            self.text_x.setData(pos=(s_x, 0, 0), color=_txt_rgba(_AXIS_X), text='X (E)' if scenario_idx in (0, 1) else 'X')
            self.text_y.setData(pos=(0, s_y, 0), color=_txt_rgba(_AXIS_Y), text='Y (H)' if scenario_idx in (0, 1) else 'Y')
            self.text_z.setData(pos=(0, 0, s_z), color=_txt_rgba(_AXIS_Z))
            hs_x, hw_x = s_x * 0.05, s_x * 0.02
            hs_y, hw_y = s_y * 0.05, s_y * 0.02
            hs_z, hw_z = s_z * 0.05, s_z * 0.02
            axis_pts = np.array([
                [0,0,0],[s_x,0,0], [s_x,0,0],[s_x-hs_x,hw_x,0], [s_x,0,0],[s_x-hs_x,-hw_x,0],
                [0,0,0],[0,s_y,0], [0,s_y,0],[hw_y,s_y-hs_y,0], [0,s_y,0],[-hw_y,s_y-hs_y,0],
                [0,0,0],[0,0,s_z], [0,0,s_z],[hw_z,0,s_z-hs_z], [0,0,s_z],[-hw_z,0,s_z-hs_z],
                [0,0,s_z],[0,hw_z,s_z-hs_z], [0,0,s_z],[0,-hw_z,s_z-hs_z],
            ], dtype=float)
            axis_colors = np.array(
                [_AXIS_X] * 6 +
                [_AXIS_Y] * 6 +
                [_AXIS_Z] * 10,
            )
            self.axis_arrows.setData(pos=axis_pts, color=axis_colors)

            # --- λ/4 tick marks on X, Y, Z axes ---
            tick_step = 0.25  # quarter-λ, in normalized (λ=1) 3D-view units
            tick_pts_list: list = []
            lbl_idx = 0

            # Tick marks are always drawn every λ/4, but text labels are thinned out
            # to whichever λ/4-multiple keeps consecutive labels at least MIN_LABEL_PX
            # apart on screen — otherwise at high zoom-out (or many periods) the labels
            # overlap each other.
            MIN_LABEL_PX = 45
            px_per_unit = max(self._pixels_per_unit(), 1e-9)
            _step_candidates = (1, 2, 4, 8, 16, 32, 64, 128)  # multiples of tick_step
            label_step_mult = next(
                (m for m in _step_candidates if m * tick_step * px_per_unit >= MIN_LABEL_PX),
                _step_candidates[-1])

            def _axis_ticks(axis_char, axis_len):
                # axis_len may be negative (Tab 2's mirrored Z axis) — ticks still
                # walk outward from the origin, just toward -axis_len instead of +axis_len.
                nonlocal lbl_idx
                axis_sign = -1.0 if axis_len < 0 else 1.0
                axis_mag = abs(axis_len)
                tick_sz = axis_mag * 0.018
                n = min(int(axis_mag / tick_step), 29)  # capped to pool size
                for i in range(1, n + 1):
                    p = i * tick_step * axis_sign
                    if abs(p) > axis_mag * 1.01:
                        break
                    show_label = (i % label_step_mult == 0)
                    text = _TICK_LABELS[i] if i < len(_TICK_LABELS) else f"{i}λ/4"
                    if axis_char == 'Z':
                        tick_pts_list.extend([[-tick_sz, 0, p], [tick_sz, 0, p],
                                              [0, -tick_sz, p], [0, tick_sz, p]])
                        lbl_pos = (-tick_sz * 3.5, 0.0, p)
                    elif axis_char == 'X':
                        tick_pts_list.extend([[p, -tick_sz, 0], [p, tick_sz, 0],
                                              [p, 0, -tick_sz], [p, 0, tick_sz]])
                        lbl_pos = (p, -tick_sz * 3.5, 0.0)
                    else:  # Y
                        tick_pts_list.extend([[-tick_sz, p, 0], [tick_sz, p, 0],
                                              [0, p, -tick_sz], [0, p, tick_sz]])
                        lbl_pos = (-tick_sz * 3.5, p, 0.0)
                    if show_label and lbl_idx < len(self.tick_label_pool):
                        lbl = self.tick_label_pool[lbl_idx]
                        lbl.setData(pos=lbl_pos, text=text, color=self._txt_color)
                        lbl.setVisible(True)
                        lbl_idx += 1

            _axis_ticks('Z', s_z)
            _axis_ticks('X', s_x)
            _axis_ticks('Y', s_y)
            for i in range(lbl_idx, len(self.tick_label_pool)):
                self.tick_label_pool[i].setVisible(False)
            self.tick_marks.setData(
                pos=np.array(tick_pts_list, dtype=float) if tick_pts_list else np.zeros((2, 3)))

            # --- wave direction indicators (k, E, B arrows near source) ---
            L = 0.25   # local frame size = λ/4, in normalized 3D-view units (matches tick scale)
            hl = L  * 0.25      # arrowhead length
            hw_ind = L * 0.08   # arrowhead width

            def _arrow_pts(origin, dvec):
                """Return 10 points forming a labelled arrow (body + 4-line head)."""
                tip   = origin + dvec
                d_hat = dvec / max(np.linalg.norm(dvec), 1e-12)
                base  = tip - d_hat * hl
                up = np.array([0., 0., 1.])
                if abs(np.dot(d_hat, up)) > 0.9:
                    up = np.array([0., 1., 0.])
                p1_ = np.cross(d_hat, up); p1_ /= np.linalg.norm(p1_)
                p2_ = np.cross(d_hat, p1_)
                p1_ *= hw_ind; p2_ *= hw_ind
                return np.array([
                    origin, tip,
                    tip, base + p1_, tip, base - p1_,
                    tip, base + p2_, tip, base - p2_,
                ])

            def _build_indicators(origin, k_hat, e1_hat, e2_hat):
                k_pts  = _arrow_pts(origin, k_hat * L)
                e_pts  = _arrow_pts(origin, e1_hat * L)
                b_pts  = _arrow_pts(origin, e2_hat * L)
                return k_pts, e_pts, b_pts, origin + k_hat*L*1.2, origin + e1_hat*L*1.2, origin + e2_hat*L*1.2

            show_source2 = scenario_idx in (1, 2)
            self.source2_marker.setData(pos=off2_v.reshape(1, 3))
            self.source2_marker.setVisible(show_source2)
            self.text_source2.setData(pos=tuple(off2_v + k2 * L * 0.6), text='Дж.2')
            self.text_source2.setVisible(show_source2)

            # hide wave-2 text labels by default; superposition re-enables them
            self.text_e2.setVisible(False); self.text_b2.setVisible(False)
            if scenario_idx == 0:
                # Tab 1: wave components only — no k/E/B reference arrows, labels sit on the axes
                self.k_indicator.setVisible(False)
                self.e_indicator.setVisible(False)
                self.b_indicator.setVisible(False)
                self.text_k.setVisible(False)
                el, bl = off1_v + e1_1 * L * 1.2, off1_v + e2_1 * L * 1.2
                self.text_e.setData(pos=tuple(el), color=_txt_rgba(_W1_E), text='E₁'); self.text_e.setVisible(True)
                self.text_b.setData(pos=tuple(bl), color=_txt_rgba(_W1_B), text='B₁'); self.text_b.setVisible(True)
            elif scenario_idx == 1:
                self.k_indicator.setVisible(True)
                self.e_indicator.setVisible(True)
                self.b_indicator.setVisible(True)
                self.text_k.setVisible(True)
                kp, ep, bp, kl, el, bl = _build_indicators(off2_v, k2, e1_2, e2_2)
                self.k_indicator.setData(pos=kp, color=np.tile(_W2_K, (len(kp), 1)))
                self.e_indicator.setData(pos=ep, color=np.tile(_W2_E, (len(ep), 1)))
                self.b_indicator.setData(pos=bp, color=np.tile(_W2_B, (len(bp), 1)))
                self.text_k.setData(pos=tuple(kl), color=_txt_rgba(_W2_K), text='k₂')
                self.text_e.setData(pos=tuple(el), color=_txt_rgba(_W2_E), text='E₂'); self.text_e.setVisible(True)
                self.text_b.setData(pos=tuple(bl), color=_txt_rgba(_W2_B), text='B₂'); self.text_b.setVisible(True)
            else:  # Tab 3 — indicators for both waves, color-coded
                self.k_indicator.setVisible(True)
                self.e_indicator.setVisible(True)
                self.b_indicator.setVisible(True)
                self.text_k.setVisible(True)
                kp1, ep1, bp1, kl1, el1, bl1 = _build_indicators(off1_v, k1, e1_1, e2_1)
                kp2, ep2, bp2, kl2, el2, bl2 = _build_indicators(off2_v, k2, e1_2, e2_2)
                k_clr = np.vstack([np.tile(_W1_K, (len(kp1), 1)), np.tile(_W2_K, (len(kp2), 1))])
                e_clr = np.vstack([np.tile(_W1_E, (len(ep1), 1)), np.tile(_W2_E, (len(ep2), 1))])
                b_clr = np.vstack([np.tile(_W1_B, (len(bp1), 1)), np.tile(_W2_B, (len(bp2), 1))])
                self.k_indicator.setData(pos=np.vstack([kp1, kp2]), color=k_clr)
                self.e_indicator.setData(pos=np.vstack([ep1, ep2]), color=e_clr)
                self.b_indicator.setData(pos=np.vstack([bp1, bp2]), color=b_clr)
                self.text_k.setData(pos=tuple(kl1), color=_txt_rgba(_W1_K), text='k₁')
                self.text_e.setData(pos=tuple(el1),  color=_txt_rgba(_W1_E), text='E₁'); self.text_e.setVisible(True)
                self.text_b.setData(pos=tuple(bl1),  color=_txt_rgba(_W1_B), text='B₁'); self.text_b.setVisible(True)
                self.text_e2.setData(pos=tuple(el2), color=_txt_rgba(_W2_E), text='E₂'); self.text_e2.setVisible(True)
                self.text_b2.setData(pos=tuple(bl2), color=_txt_rgba(_W2_B), text='B₂'); self.text_b2.setVisible(True)

            if not hasattr(self, '_cam_init'):
                self.view.opts['distance'] = self._distance_for_pxw(self.pxw_spin.value())
                if scenario_idx == 0:
                    self._apply_wave1_camera()
                elif scenario_idx == 1:
                    self._apply_wave2_camera()
                else:
                    self.view.setCameraPosition(elevation=30, azimuth=45)
                self.grid.scale(z_max_v / 10, z_max_v / 10, 1)
                self._cam_init = True

            # --- data table ---
            step_idx = self.z_steps // (n_act * 10) if n_act > 0 else self.z_steps // 10
            if step_idx < 1: step_idx = 1

            vals_all = []
            for i in range(11):
                idx = min(i * step_idx, self.z_steps - 1)
                vals_all.append([tE1[idx], tE2[idx], tH1[idx], tH2[idx], tSz[idx]])
            vals_all = np.array(vals_all)

            col_active = [True] * 5
            for j in range(4):
                if np.allclose(vals_all[:, j], 0, atol=1e-12):
                    col_active[j] = False

            all_headers   = ["Coord", "Ex(В/м)", "Ey(В/м)", "Hx(А/м)", "Hy(А/м)", "Sz(Вт/м²)"]
            active_headers = ["Coord"] + [all_headers[j+1] for j in range(5) if col_active[j]]

            if self.data_table.columnCount() != len(active_headers):
                self.data_table.setColumnCount(len(active_headers))
                self.data_table.setHorizontalHeaderLabels(active_headers)
                hh = self.data_table.horizontalHeader()
                if hh:
                    hh.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
                    hh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)

            fmt_list = [".2f", ".2f", ".2e", ".2e", ".2f"]
            for i in range(11):
                z_frac = "0" if i == 0 else ("1.0λ" if i == 10 else f"{i/10:.1f}λ")
                ci = self.data_table.item(i, 0) or QTableWidgetItem()
                ci.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.data_table.setItem(i, 0, ci)
                ci.setText(z_frac)
                curr_col = 1
                for j in range(5):
                    if col_active[j]:
                        cell = self.data_table.item(i, curr_col) or QTableWidgetItem()
                        cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                        self.data_table.setItem(i, curr_col, cell)
                        cell.setText(self._format_value(vals_all[i, j], fmt_list[j]))
                        curr_col += 1

            # --- visualization scaling (3D-view units, i.e. normalized to λ) ---
            h_size    = z_max_v * 0.03
            s_sc      = z_max_v * 0.02

            show_e = comp_mode in ["Узагальнений підхід", "Електрична (E)"]
            show_b = comp_mode in ["Узагальнений підхід", "Магнітна (B)"]
            show_s = comp_mode == "Узагальнений підхід"
            show_s_w1 = show_s and scenario_idx != 0  # Tab 1: no separate Poynting arrows

            Ex1v = Ex1 * vis_scale;  Ey1v = Ey1 * vis_scale
            vBx1 = Bx1 * C * vis_scale; vBy1 = By1 * C * vis_scale
            _, _, Sz1 = calculate_poynting(Ex1, Ey1, Bx1, By1)
            sv1 = Sz1 * scale_s

            Ex2v = Ex2 * vis_scale;  Ey2v = Ey2 * vis_scale
            vBx2 = Bx2 * C * vis_scale; vBy2 = By2 * C * vis_scale
            _, _, Sz2 = calculate_poynting(Ex2, Ey2, Bx2, By2)
            sv2 = Sz2 * scale_s

            draw_wave1 = scenario_idx in (0, 2)
            draw_wave2 = scenario_idx in (1, 2)

            # ---- 3-D modes ----
            def w2w(local_pts, M, origin):
                return local_pts @ M.T + origin

            def make_wave_funcs(M, origin, z_arr):
                N = len(z_arr)
                def local_vecs(t1, t2):
                    pts = np.zeros((N * 2, 3))
                    pts[0::2, 2] = z_arr
                    pts[1::2, 0] = t1; pts[1::2, 1] = t2; pts[1::2, 2] = z_arr
                    return w2w(pts, M, origin)
                def local_curve(t1, t2):
                    return w2w(np.column_stack([t1, t2, z_arr]), M, origin)
                return local_vecs, local_curve

            lv1, _ = make_wave_funcs(M1, off1_v, z_wave1_v)
            lv2, _ = make_wave_funcs(M2, off2_v, z_wave2_v)

            def draw_s_arrows(s_val, M, origin, z_arr):
                N = len(z_arr)
                s_loc = np.zeros((N * 2, 3))
                s_loc[0::2, 2] = z_arr
                s_loc[1::2, 2] = z_arr + s_val
                idx_s = np.arange(0, N, 10)
                sv_s = s_val[idx_s]; z_s = z_arr[idx_s]
                mask = np.abs(sv_s) >= (z_max_v * 0.001)
                if np.any(mask):
                    sv_s = sv_s[mask]; z_s = z_s[mask]
                    d = np.sign(sv_s); ze = z_s + sv_s; zb = ze - d * h_size
                    hw_s = s_sc * 0.5; n_s = len(ze)
                    sa = np.empty((n_s * 8, 3))
                    sa[0::8] = np.c_[np.zeros(n_s), np.zeros(n_s), ze]
                    sa[1::8] = np.c_[np.full(n_s, hw_s), np.zeros(n_s), zb]
                    sa[2::8] = sa[0::8]
                    sa[3::8] = np.c_[np.full(n_s, -hw_s), np.zeros(n_s), zb]
                    sa[4::8] = sa[0::8]
                    sa[5::8] = np.c_[np.zeros(n_s), np.full(n_s, hw_s), zb]
                    sa[6::8] = sa[0::8]
                    sa[7::8] = np.c_[np.zeros(n_s), np.full(n_s, -hw_s), zb]
                    s_loc = np.vstack([s_loc, sa])
                return w2w(s_loc, M, origin)

            def draw_linear_wave(Ex_v, Ey_v, vBx, vBy, sv, M, origin, z_arr,
                                  e_line, b_line, s_line, e_curve, b_curve, visible, show_s_flag):
                # e_line/b_line (per-point field-vector stalks) are retired — curves only
                e_line.setVisible(False)
                b_line.setVisible(False)
                e_curve.setVisible(show_e and visible)
                b_curve.setVisible(show_b and visible)
                s_line.setVisible(show_s_flag and visible)
                if not visible:
                    return
                e_curve.setData(pos=w2w(np.column_stack([Ex_v, Ey_v, z_arr]), M, origin))
                b_curve.setData(pos=w2w(np.column_stack([vBx, vBy, z_arr]), M, origin))
                s_line.setData(pos=draw_s_arrows(sv, M, origin, z_arr))

            if repr_mode == "Linear (3D)":
                self.e_surf.setVisible(False); self.b_surf.setVisible(False); self.s_surf.setVisible(False)
                draw_linear_wave(Ex1v, Ey1v, vBx1, vBy1, sv1, M1, off1_v, z_wave1_v,
                                 self.e_line, self.b_line, self.s_line, self.e_curve, self.b_curve, draw_wave1, show_s_w1)
                draw_linear_wave(Ex2v, Ey2v, vBx2, vBy2, sv2, M2, off2_v, z_wave2_v,
                                 self.e_line2, self.b_line2, self.s_line2, self.e_curve2, self.b_curve2, draw_wave2, show_s)

            else:  # Surface (3D)
                for it in (self.e_line, self.b_line, self.s_line, self.e_curve, self.b_curve,
                            self.e_line2, self.b_line2, self.s_line2, self.e_curve2, self.b_curve2):
                    it.setVisible(False)
                self.e_surf.setVisible(show_e and draw_wave1)
                self.b_surf.setVisible(show_b and draw_wave1)
                self.s_surf.setVisible(show_s and draw_wave1 and scenario_idx != 0)
                if show_e and draw_wave1:
                    self.e_surf.setMeshData(vertexes=lv1(Ex1v, Ey1v), faces=self.surface_faces)
                if show_b and draw_wave1:
                    self.b_surf.setMeshData(vertexes=lv1(vBx1, vBy1), faces=self.surface_faces)
                if show_s and draw_wave1:
                    n_seg = 4
                    v_s = np.zeros((len(z_wave1_v) * n_seg, 3))
                    r = sv1 * 0.5
                    for i in range(n_seg):
                        angle = i * (2 * np.pi / n_seg)
                        v_s[i::n_seg, 0] = r * np.cos(angle)
                        v_s[i::n_seg, 1] = r * np.sin(angle)
                        v_s[i::n_seg, 2] = z_wave1_v
                    self.s_surf.setMeshData(vertexes=w2w(v_s, M1, off1_v), faces=self.volume_faces)

        finally:
            self._is_updating = False


if __name__ == "__main__":
    app = QApplication(sys.argv)
    try:
        window = MainWindow()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        import traceback
        print("Виникла фатальна помилка:")
        traceback.print_exc()
        input("Натисніть Enter для виходу...")