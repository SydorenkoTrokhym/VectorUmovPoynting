import json
from dataclasses import replace
import math
import pathlib
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QComboBox, QDoubleSpinBox, QSpinBox,
                             QTableWidget, QTableWidgetItem, QHeaderView, QPushButton,
                             QGroupBox, QCheckBox, QTabWidget, QFileDialog, QMessageBox)
from PyQt6.QtGui import QPixmap, QPainter, QColor, QFont, QQuaternion
from PyQt6.QtCore import Qt, QTimer, QRect
import pyqtgraph.opengl as gl

import camera
from camera import (BG_COLORS as _BG_COLORS, txt_rgba as _txt_rgba,
                    _WAVE1_CAMERA_ROTATION, _WAVE2_CAMERA_ROTATION)
from controls import build_controls
from geometry import build_surface_faces
from models import ScenarioConfiguration, VisualizationSettings
from persistence import ConfigurationError, configuration_from_dict, configuration_to_dict
from physics import C, MU0
from renderer import WaveRenderObjects
from simulation import simulate_counterpropagating, simulate_wave

# Propagation frames: (k_hat, e1_hat, e2_hat) where e2 = k × e1 (right-hand rule for B)
_BASE_FRAMES = {
    '+Z': (np.array([0.,0.,1.]),  np.array([1.,0.,0.]), np.array([0., 1.,0.])),
    '-Z': (np.array([0.,0.,-1.]), np.array([1.,0.,0.]), np.array([0.,-1.,0.])),
    '+X': (np.array([1.,0.,0.]),  np.array([0.,1.,0.]), np.array([0., 0.,1.])),
    '-X': (np.array([-1.,0.,0.]), np.array([0.,1.,0.]), np.array([0., 0.,-1.])),
    '+Y': (np.array([0.,1.,0.]),  np.array([1.,0.,0.]), np.array([0., 0.,-1.])),
    '-Y': (np.array([0.,-1.,0.]), np.array([1.,0.,0.]), np.array([0., 0.,1.])),
}

# Wave colors — kept here so both __init__ and update_plot use the same values
FIELD_VIEW_SCALE = 0.30  # 3D-view units per V/m (E) and per V/m-equivalent c·B

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
        self.configuration = ScenarioConfiguration()
        self.scenario_params = self.configuration.waves
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
        build_controls(self, control_layout)
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

        self.e_surf2 = gl.GLMeshItem(smooth=False, color=_W2_E[:3]+(0.5,), shader='shaded', glOptions='translucent')
        self.b_surf2 = gl.GLMeshItem(smooth=False, color=_W2_B[:3]+(0.5,), shader='shaded', glOptions='translucent')
        self.s_surf2 = gl.GLMeshItem(smooth=False, color=_W2_S[:3]+(0.5,), shader='shaded', glOptions='translucent')
        for surf in (self.e_surf2, self.b_surf2, self.s_surf2):
            surf.setVisible(False)
            self.view.addItem(surf)

        # One renderer-owned bundle per wave, covering lines, curves and surfaces.
        self.wave_renderers = (
            WaveRenderObjects(self.e_line, self.b_line, self.s_line, self.e_curve, self.b_curve,
                              self.e_surf, self.b_surf, self.s_surf),
            WaveRenderObjects(self.e_line2, self.b_line2, self.s_line2, self.e_curve2, self.b_curve2,
                              self.e_surf2, self.b_surf2, self.s_surf2),
        )

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

        # Render once after the window enters the event loop. The animation timer
        # starts only when Play is pressed.
        self._update_legend(0)
        QTimer.singleShot(0, self.update_plot)

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
        return build_surface_faces(z_steps, closed)

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
            self.timer.stop()
            self.update_plot()
        else:
            self.pause_button.setText("⏸ Пауза")
            self.timer.start(50)

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

        # Tab 2 (index==2) = superposition
        is_superposition = index == 2
        editable = index < 2
        self.wave_group.setEnabled(editable)
        self.mod_group.setEnabled(editable)
        # Wave 2 uses a free source position; superposition uses only source separation.
        self.source_group.setVisible(index >= 1)
        self.source_group.setTitle("Позиція джерела 2" if index == 1 else "Параметри суперпозиції")
        self.source_position_widget.setVisible(index >= 1)

        self._update_legend(index)
        if is_superposition:
            self._is_updating = True
            source = self.scenario_params[1].source_position
            self.x0_spin.setValue(source[0])
            self.y0_spin.setValue(source[1])
            self.z0_spin.setValue(source[2])
            self._is_updating = False
            self.update_plot()
            return

        params = self.scenario_params[index]
        self._is_updating = True
        try:
            self.pol_combo.setCurrentText(params.polarization)
            self.amp_spin.setValue(params.amplitude)
            self.phase_spin.setValue(params.phase_degrees)
            self.mod_combo.setCurrentText(params.modulation_type)
            self.mod_freq_spin.setValue(params.modulation_frequency)
            self.mod_index_spin.setValue(params.modulation_index)
            self.n_periods_spin.setValue(params.periods)
            self.dir_mode_combo.setCurrentText(params.direction_mode)
            self.wave_prop_dir_combo.setCurrentText(params.propagation_direction)
            self.e_dir_combo.setCurrentText(params.electric_direction)
            self._rebuild_magnetic_direction_choices(params.magnetic_direction)
            dv = params.direction_vector
            self.dir_x_spin.setValue(dv[0]); self.dir_y_spin.setValue(dv[1]); self.dir_z_spin.setValue(dv[2])
            if index == 1:
                self.x0_spin.setValue(params.source_position[0])
                self.y0_spin.setValue(params.source_position[1])
                self.z0_spin.setValue(params.source_position[2])
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
        p.polarization = self.pol_combo.currentText()
        p.amplitude = self.amp_spin.value()
        p.phase_degrees = self.phase_spin.value()
        p.modulation_type = self.mod_combo.currentText()
        p.modulation_frequency = self.mod_freq_spin.value()
        p.modulation_index = self.mod_index_spin.value()
        p.periods = self.n_periods_spin.value()
        p.direction_mode = self.dir_mode_combo.currentText()
        p.propagation_direction = self.wave_prop_dir_combo.currentText()
        p.direction_vector = (self.dir_x_spin.value(), self.dir_y_spin.value(), self.dir_z_spin.value())
        p.electric_direction = self.e_dir_combo.currentText()
        p.magnetic_direction = self.b_dir_combo.currentText()
        self.update_plot()

    def _rebuild_magnetic_direction_choices(self, preferred=None):
        choices = ["+Y", "-Y"] if self.e_dir_combo.currentText().endswith("X") else ["+X", "-X"]
        current = preferred or self.b_dir_combo.currentText()
        self.b_dir_combo.blockSignals(True)
        self.b_dir_combo.clear()
        self.b_dir_combo.addItems(choices)
        if current in choices:
            self.b_dir_combo.setCurrentText(current)
        self.b_dir_combo.blockSignals(False)

    def _on_field_direction_changed(self):
        if self._is_updating:
            return
        if self.sender() is self.e_dir_combo:
            self._rebuild_magnetic_direction_choices()
        from geometry import field_directions
        k_hat, _, _ = field_directions(self.e_dir_combo.currentText(), self.b_dir_combo.currentText())
        direction = "+Z" if k_hat[2] > 0 else "-Z"
        self.dir_mode_combo.setCurrentText("Базова вісь")
        self.wave_prop_dir_combo.setCurrentText(direction)
        self.field_direction_status.setText(f"E ⊥ B;  E × B = {direction}")
        self.field_direction_status.setStyleSheet("font-size: 8pt; color: #33aa55;")
        self._on_param_changed()

    def _on_propagation_changed(self):
        if self._is_updating:
            return
        direction = self.wave_prop_dir_combo.currentText()
        enabled = direction in ("+Z", "-Z") and self.dir_mode_combo.currentText() == "Базова вісь"
        self.e_dir_combo.setEnabled(enabled)
        self.b_dir_combo.setEnabled(enabled)
        if enabled:
            from geometry import field_directions
            desired = 1 if direction == "+Z" else -1
            for candidate in [self.b_dir_combo.itemText(i) for i in range(self.b_dir_combo.count())]:
                if np.sign(field_directions(self.e_dir_combo.currentText(), candidate)[0][2]) == desired:
                    self.b_dir_combo.setCurrentText(candidate)
                    break
            self.field_direction_status.setText(f"E ⊥ B;  E × B = {direction}")
        else:
            self.field_direction_status.setText("Вибір ±X/±Y доступний для потоку вздовж ±Z")
        self._on_param_changed()

    def _on_dir_mode_changed(self, text):
        self._sync_dir_visibility()
        self._on_param_changed()

    def _sync_dir_visibility(self):
        is_custom = self.dir_mode_combo.currentText() == "Довільний вектор"
        self.wave_prop_dir_combo.setVisible(not is_custom)
        self.dir_vec_widget.setVisible(is_custom)
        enabled = not is_custom and self.wave_prop_dir_combo.currentText() in ("+Z", "-Z")
        self.e_dir_combo.setEnabled(enabled)
        self.b_dir_combo.setEnabled(enabled)

    def _on_source_changed(self):
        if self._is_updating:
            return
        p = self.scenario_params[1]

        pos = (self.x0_spin.value(), self.y0_spin.value(), self.z0_spin.value())
        if np.linalg.norm(pos) <= 1e-9:
            # Source 2 can't sit on source 1 (origin) — the propagation direction
            # between them would be undefined. Nudge it back off the origin.
            self._is_updating = True
            try:
                self.z0_spin.setValue(0.25)
            finally:
                self._is_updating = False
            pos = (pos[0], pos[1], 0.25)
        p.source_position = pos
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
        self.configuration.visualization = VisualizationSettings(
            self.comp_combo.currentText(), self.repr_combo.currentText(),
            self.bg_combo.currentText(), self.proj_combo.currentText(),
            self.pxw_spin.value(),
        )
        return configuration_to_dict(self.configuration)

    @staticmethod
    def _default_params():
        return configuration_to_dict(ScenarioConfiguration())

    def _on_reset(self):
        """Restore wave/visualization settings to their initial defaults."""
        self._apply_params(self._default_params())
        self.t = 0.0
        self.scenario_tabs.setCurrentIndex(0)

    def _validate_params(self, data):
        """Return list of error strings (empty list = data is valid)."""
        try:
            configuration_from_dict(data)
            return []
        except ConfigurationError as exc:
            return [str(exc)]

    def _apply_params(self, data):
        """Restore scenario_params and UI widgets from a validated dict."""
        self.configuration = configuration_from_dict(data)
        self.scenario_params = self.configuration.waves
        viz = self.configuration.visualization
        self._is_updating = True
        try:
            def _set(combo, key):
                v = getattr(viz, key)
                items = [combo.itemText(i) for i in range(combo.count())]
                if v in items:
                    combo.setCurrentText(v)
            _set(self.comp_combo,  'component_mode')
            _set(self.repr_combo,  'representation_mode')
            _set(self.proj_combo,  'projection')
            bg = viz.background
            if bg in _BG_COLORS:
                self.bg_combo.setCurrentText(bg)
            pxw = viz.pixels_per_wavelength
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
        z_axis_len = self.scenario_params[0].periods + 2
        self.view.opts['center'] = pg.Vector(0, 0, z_axis_len * 0.25)
        self.view.update()

    def _apply_wave2_camera(self):
        """Fixed camera for Tab 2: X down, Y away from the viewer, Z right-to-left.
        The look-at center is offset along +Z so the origin sits a sixth of the
        screen width to the left, leaving room on the right for the wave to develop."""
        self.view.opts['rotationMethod'] = 'quaternion'
        self.view.opts['rotation'] = _WAVE2_CAMERA_ROTATION
        z_axis_len = self.scenario_params[1].periods + 2
        self.view.opts['center'] = pg.Vector(0, 0, z_axis_len * (1.0 / 6.0))
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
            p1 = self.scenario_params[0]
            p2 = self.scenario_params[1]
            amp1, amp2 = p1.amplitude, p2.amplitude

            # All field data comes from simulation.py; this method only lays it out.
            if scenario_idx == 2:
                # Wave 1 starts at the origin; wave 2 starts at its XYZ point in lambda.
                # Their directions are derived from the line connecting the two sources.
                combined = simulate_counterpropagating(
                    p1, p2, self.t, frequency=freq, steps=self.z_steps)
                wave1, wave2 = combined.wave1, combined.wave2
                table_vectors = (combined.electric, combined.magnetic_h, combined.poynting)
                n_act = float(np.linalg.norm(p2.source_position))
                z_max = (n_act + 1.0) * wavelength
            else:
                # Tab 1 is the wave from the origin, whatever source a loaded file names.
                wave1 = simulate_wave(replace(p1, source_position=(0.0, 0.0, 0.0)),
                                      self.t, frequency=freq, steps=self.z_steps)
                wave2 = simulate_wave(p2, self.t, frequency=freq, steps=self.z_steps)
                shown, shown_params = (wave1, p1) if scenario_idx == 0 else (wave2, p2)
                table_vectors = (shown.electric, shown.magnetic_h, shown.poynting)
                n_act = shown_params.periods
                # Global coordinate-system size covers both waves' extents
                source2_distance = np.linalg.norm(np.asarray(p2.source_position) * wavelength)
                z_max = max(p1.periods * wavelength,
                            source2_distance + p2.periods * wavelength) * 1.15

            k1, e1_1, e2_1 = wave1.frame
            k2, e1_2, e2_2 = wave2.frame

            # 3D-view positions are normalized to λ (not raw SI meters, which at 1 Hz means
            # λ=3e8 m): at that magnitude, the GPU-rendered axis lines (float32 vertex buffers)
            # and the CPU-rendered GLTextItem labels (float32 QMatrix4x4) round differently,
            # visibly detaching the X/Y/Z labels from their axis tips.
            z_max_v = z_max / wavelength
            pos1_v  = wave1.positions / wavelength
            pos2_v  = wave2.positions / wavelength
            off1_v  = np.zeros(3)
            off2_v  = np.asarray(p2.source_position, dtype=float)

            # Fixed physical→view-unit scale factors for E/B/S deflection (3D-view units
            # per V/m, etc.). These must NOT depend on amp1/amp2 — earlier they were
            # divided by max(amp1, amp2), which silently canceled out whichever wave
            # was largest, so raising that wave's amplitude had no visible effect.
            # Calibrated so amp = 1 V/m gives a modest, non-dominating deflection
            # relative to the Z (propagation) axis.
            # Transverse field scale is independent of the displayed segment length.
            # Changing `periods` therefore extends only the propagation axis.
            vis_scale = FIELD_VIEW_SCALE
            scale_s = 0.80 * MU0 * C
            
            # Common scale for all modes
            max_amp_for_scale = max(amp1, amp2, 1e-12)

            # --- coordinate axis arrows (fixed colors, distinct from wave colors) ---
            # X/Y are sized to the field actually drawn (2× its peak excursion); Z is sized to
            # the configured wave length plus a 2λ margin — each axis fits its own data instead
            # of sharing one oversized cube.
            amp_extent = max_amp_for_scale * vis_scale   # peak |E|/|B| reach actually drawn
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
            if scenario_idx == 2:
                table_indices = np.linspace(0, self.z_steps - 1, 11, dtype=int)
            else:
                step_idx = max(int(self.z_steps // (n_act * 10)), 1) if n_act > 0 else self.z_steps // 10
                table_indices = [min(i * step_idx, self.z_steps - 1) for i in range(11)]

            electric_table, magnetic_table, poynting_table = table_vectors
            vals_all = []
            for idx in table_indices:
                vals_all.append(np.concatenate((
                    electric_table[idx], magnetic_table[idx], poynting_table[idx])))
            vals_all = np.array(vals_all)

            col_active = [not np.allclose(vals_all[:, j], 0, atol=1e-12) for j in range(9)]
            all_headers = [
                "Coord", "Ex(В/м)", "Ey(В/м)", "Ez(В/м)",
                "Hx(А/м)", "Hy(А/м)", "Hz(А/м)",
                "Sx(Вт/м²)", "Sy(Вт/м²)", "Sz(Вт/м²)",
            ]
            active_headers = ["Coord"] + [all_headers[j + 1] for j in range(9) if col_active[j]]

            if self.data_table.columnCount() != len(active_headers):
                self.data_table.setColumnCount(len(active_headers))
                self.data_table.setHorizontalHeaderLabels(active_headers)
                hh = self.data_table.horizontalHeader()
                if hh:
                    hh.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
                    hh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)

            fmt_list = [".2f"] * 3 + [".2e"] * 3 + [".2f"] * 3
            for i in range(11):
                coordinate = (n_act * i / 10) if scenario_idx == 2 else (i / 10)
                z_frac = "0" if i == 0 else f"{coordinate:.2g}λ"
                ci = self.data_table.item(i, 0) or QTableWidgetItem()
                ci.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.data_table.setItem(i, 0, ci)
                ci.setText(z_frac)
                curr_col = 1
                for j in range(9):
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
            show_s_w2 = show_s and scenario_idx != 1  # Wave 2 tab: hide Umov-Poynting vector

            # Field vectors in 3D-view units, drawn at the simulated sample positions
            e1_v = wave1.electric * vis_scale;  b1_v = wave1.magnetic * C * vis_scale
            e2_v = wave2.electric * vis_scale;  b2_v = wave2.magnetic * C * vis_scale
            sv1 = (wave1.poynting @ k1) * scale_s
            sv2 = (wave2.poynting @ k2) * scale_s

            draw_wave1 = scenario_idx in (0, 2)
            draw_wave2 = scenario_idx in (1, 2)

            # ---- 3-D modes ----
            def stalks(base, vectors):
                """Interleave base points with their tips: one line segment per sample."""
                pts = np.empty((len(base) * 2, 3))
                pts[0::2] = base
                pts[1::2] = base + vectors
                return pts

            def draw_s_arrows(s_val, pos, frame):
                k_hat, t1, t2 = frame
                lines = stalks(pos, np.outer(s_val, k_hat))
                idx_s = np.arange(0, len(pos), 10)
                sv_s = s_val[idx_s]; base = pos[idx_s]
                mask = np.abs(sv_s) >= (z_max_v * 0.001)
                if np.any(mask):
                    sv_s = sv_s[mask]; base = base[mask]
                    tip = base + np.outer(sv_s, k_hat)
                    head = tip - np.outer(np.sign(sv_s) * h_size, k_hat)
                    hw_s = s_sc * 0.5
                    sa = np.empty((len(tip) * 8, 3))
                    for n, side in enumerate((t1, -t1, t2, -t2)):
                        sa[2 * n::8] = tip
                        sa[2 * n + 1::8] = head + hw_s * side
                    lines = np.vstack([lines, sa])
                return lines

            def s_tube(s_val, pos, frame, n_seg=4):
                _, t1, t2 = frame
                r = s_val * 0.5
                verts = np.empty((len(pos) * n_seg, 3))
                for i in range(n_seg):
                    angle = i * (2 * np.pi / n_seg)
                    verts[i::n_seg] = (pos + np.outer(r * np.cos(angle), t1)
                                       + np.outer(r * np.sin(angle), t2))
                return verts

            def draw_linear_wave(e_v, b_v, sv, pos, frame,
                                  e_line, b_line, s_line, e_curve, b_curve, visible, show_s_flag):
                # e_line/b_line (per-point field-vector stalks) are retired — curves only
                e_line.setVisible(False)
                b_line.setVisible(False)
                e_curve.setVisible(show_e and visible)
                b_curve.setVisible(show_b and visible)
                s_line.setVisible(show_s_flag and visible)
                if not visible:
                    return
                e_curve.setData(pos=pos + e_v)
                b_curve.setData(pos=pos + b_v)
                s_line.setData(pos=draw_s_arrows(sv, pos, frame))

            if repr_mode == "Linear (3D)":
                for surf in (self.e_surf, self.b_surf, self.s_surf,
                             self.e_surf2, self.b_surf2, self.s_surf2):
                    surf.setVisible(False)
                draw_linear_wave(e1_v, b1_v, sv1, pos1_v, wave1.frame,
                                 self.e_line, self.b_line, self.s_line, self.e_curve, self.b_curve, draw_wave1, show_s_w1)
                draw_linear_wave(e2_v, b2_v, sv2, pos2_v, wave2.frame,
                                 self.e_line2, self.b_line2, self.s_line2, self.e_curve2, self.b_curve2, draw_wave2, show_s_w2)

            else:  # Surface (3D)
                for it in (self.e_line, self.b_line, self.s_line, self.e_curve, self.b_curve,
                            self.e_line2, self.b_line2, self.s_line2, self.e_curve2, self.b_curve2):
                    it.setVisible(False)
                self.e_surf.setVisible(show_e and draw_wave1)
                self.b_surf.setVisible(show_b and draw_wave1)
                self.s_surf.setVisible(show_s and draw_wave1 and scenario_idx != 0)
                self.e_surf2.setVisible(show_e and draw_wave2)
                self.b_surf2.setVisible(show_b and draw_wave2)
                self.s_surf2.setVisible(show_s_w2 and draw_wave2)
                if show_e and draw_wave1:
                    self.e_surf.setMeshData(vertexes=stalks(pos1_v, e1_v), faces=self.surface_faces)
                if show_b and draw_wave1:
                    self.b_surf.setMeshData(vertexes=stalks(pos1_v, b1_v), faces=self.surface_faces)
                if show_s and draw_wave1:
                    self.s_surf.setMeshData(vertexes=s_tube(sv1, pos1_v, wave1.frame),
                                            faces=self.volume_faces)
                if show_e and draw_wave2:
                    self.e_surf2.setMeshData(vertexes=stalks(pos2_v, e2_v), faces=self.surface_faces)
                if show_b and draw_wave2:
                    self.b_surf2.setMeshData(vertexes=stalks(pos2_v, b2_v), faces=self.surface_faces)
                if show_s and draw_wave2:
                    self.s_surf2.setMeshData(vertexes=s_tube(sv2, pos2_v, wave2.frame),
                                             faces=self.volume_faces)

        except ValueError as exc:
            self.params_status.setText(f"Помилка обчислення: {exc}")
            self.params_status.setStyleSheet("font-size: 8pt; color: #ff5555;")
        finally:
            self._is_updating = False


