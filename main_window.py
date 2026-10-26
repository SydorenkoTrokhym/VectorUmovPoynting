import json
import pathlib
from dataclasses import replace

import numpy as np
import pyqtgraph.opengl as gl
from PyQt6.QtCore import QEvent, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QPainter, QPixmap
from PyQt6.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
                             QTabWidget, QVBoxLayout, QWidget)

import camera
from controls import build_controls, choice, set_choice
from data_table import fill_table
from geometry import field_directions, wave_frame
from legend import legend_html
from models import (Background, ComponentMode, DirectionMode, Modulation, Polarization,
                    Projection, Representation, ScenarioConfiguration, VisualizationSettings)
from persistence import ConfigurationError, configuration_from_dict, configuration_to_dict
from physics import C
from scene import (FIELD_VIEW_SCALE, AxesLayer, DrawStyle, FieldLayer, IndicatorLayer,
                   MeetingFieldArrows, SceneFrame, axis_extents, field_axis_notes)
from simulation import simulate_counterpropagating, simulate_wave
from superposition import (CARRIER_FREQUENCY, TIME_STEP, place_source2, preset_periods,
                           time_text)

WAVE1_TAB, WAVE2_TAB, SUPERPOSITION_TAB = 0, 1, 2
Z_STEPS = 200        # samples along each wave
TABLE_ROWS = 11


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Візуалізація вектора Пойнтінга")
        self.resize(1200, 800)

        # Animation and update state
        self.t = 0.0
        self._paused = True
        self._is_updating = False
        self._camera_ready = False   # the first frame sets the camera from px/λ
        self._calculation_failed = False   # the status line shows a calculation error
        self.z_steps = Z_STEPS

        # Scenario tabs — each stores independent wave/modulation + source parameters
        self.configuration = ScenarioConfiguration()
        self.scenario_params = self.configuration.waves

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left: control panel
        control_panel = QWidget()
        control_panel.setFixedWidth(300)
        control_layout = QVBoxLayout(control_panel)
        main_layout.addWidget(control_panel)

        # Right: 3D view and its layers
        self.view = gl.GLViewWidget()
        self.view.opts['distance'] = 20
        self.axes = AxesLayer(self.view)
        self.fields = FieldLayer(self.view, self.z_steps)
        self.indicators = IndicatorLayer(self.view)
        self.meeting_arrows = MeetingFieldArrows(self.view)
        # Wheel zoom and resizes change px/λ behind the spinbox's back
        self.view.installEventFilter(self)

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

        build_controls(self, control_layout)
        self._on_visibility_changed()
        self._on_background_changed()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)

        # Render once after the window enters the event loop. The animation timer
        # starts only when Play is pressed.
        self._update_legend(WAVE1_TAB)
        QTimer.singleShot(0, self.update_plot)

    # ------------------------------------------------------------------
    # Animation and time
    # ------------------------------------------------------------------

    def _on_pause_toggled(self, checked):
        self._paused = checked
        if checked:
            self.pause_button.setText("▶ Відтворити")
            self.timer.stop()
            self.update_plot()
        else:
            self.pause_button.setText("⏸ Пауза")
            self.timer.start(50)

    def _on_time_preset(self, preset):
        """Jump to a characteristic moment of the superposition: fronts just left their
        sources, fronts meet midway, or the waves overlap along the whole gap."""
        periods = preset_periods(preset, self.configuration.source_separation)
        self._set_time(periods / CARRIER_FREQUENCY)

    def _on_time_step(self, sign):
        self._set_time(max(self.t + sign * TIME_STEP / CARRIER_FREQUENCY, 0.0))

    def _set_time(self, t):
        """Pause the animation and show the given moment."""
        self.pause_button.setChecked(True)
        self.t = t
        self.update_plot()

    # ------------------------------------------------------------------
    # Scenario and parameter controls
    # ------------------------------------------------------------------

    def _update_legend(self, tab):
        strip, panel = legend_html(tab)
        self.legend_strip.setText(strip)
        self.legend_label.setText(panel)

    def _on_visibility_changed(self):
        self.wave_group.setVisible(self.chk_wave.isChecked())
        self.viz_group.setVisible(self.chk_viz.isChecked())
        self.mod_group.setVisible(self.chk_mod.isChecked())
        self.data_group.setVisible(self.chk_data.isChecked())

    def _on_phase_preset_changed(self, index):
        if index > 0:
            self.phase_spin.setValue(float(self.phase_preset_combo.currentText()))

    def _on_scenario_changed(self, index):
        self._apply_camera(index)

        is_superposition = index == SUPERPOSITION_TAB
        self.wave_group.setEnabled(not is_superposition)
        self.mod_group.setEnabled(not is_superposition)
        # Wave 2 uses a free source position; superposition uses only source separation.
        self.source_group.setVisible(index != WAVE1_TAB)
        self.source_group.setTitle(
            "Позиція джерела 2" if index == WAVE2_TAB else "Параметри суперпозиції")
        self.source_position_widget.setVisible(index == WAVE2_TAB)
        self.separation_widget.setVisible(is_superposition)
        self.superposition_time_widget.setVisible(is_superposition)

        self._update_legend(index)
        if is_superposition:
            self._is_updating = True
            try:
                self.separation_spin.setValue(self.configuration.source_separation)
            finally:
                self._is_updating = False
            self.update_plot()
            return

        params = self.scenario_params[index]
        self._is_updating = True
        try:
            set_choice(self.pol_combo, params.polarization)
            self.amp_spin.setValue(params.amplitude)
            self.phase_spin.setValue(params.phase_degrees)
            set_choice(self.mod_combo, params.modulation_type)
            self.mod_freq_spin.setValue(params.modulation_frequency)
            self.mod_index_spin.setValue(params.modulation_index)
            self.n_periods_spin.setValue(params.periods)
            set_choice(self.dir_mode_combo, params.direction_mode)
            self.wave_prop_dir_combo.setCurrentText(params.propagation_direction)
            self.e_dir_combo.setCurrentText(params.electric_direction)
            self._rebuild_magnetic_direction_choices(params.magnetic_direction)
            dv = params.direction_vector
            self.dir_x_spin.setValue(dv[0]); self.dir_y_spin.setValue(dv[1]); self.dir_z_spin.setValue(dv[2])
            if index == WAVE2_TAB:
                self.x0_spin.setValue(params.source_position[0])
                self.y0_spin.setValue(params.source_position[1])
                self.z0_spin.setValue(params.source_position[2])
        finally:
            self._is_updating = False
        self._sync_dir_visibility()
        self.update_plot()

    def _on_param_changed(self):
        if self._is_updating:
            return
        index = self.scenario_tabs.currentIndex()
        if index == SUPERPOSITION_TAB:
            return
        p = self.scenario_params[index]
        direction = (p.direction_mode, p.propagation_direction, p.direction_vector)
        p.polarization = choice(self.pol_combo, Polarization)
        p.amplitude = self.amp_spin.value()
        p.phase_degrees = self.phase_spin.value()
        p.modulation_type = choice(self.mod_combo, Modulation)
        p.modulation_frequency = self.mod_freq_spin.value()
        p.modulation_index = self.mod_index_spin.value()
        p.periods = self.n_periods_spin.value()
        p.direction_mode = choice(self.dir_mode_combo, DirectionMode)
        p.propagation_direction = self.wave_prop_dir_combo.currentText()
        p.direction_vector = (self.dir_x_spin.value(), self.dir_y_spin.value(), self.dir_z_spin.value())
        p.electric_direction = self.e_dir_combo.currentText()
        p.magnetic_direction = self.b_dir_combo.currentText()
        if direction != (p.direction_mode, p.propagation_direction, p.direction_vector):
            self._apply_camera(index)   # the tab's camera turns with the wave's axis
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
        k_hat, _, _ = field_directions(self.e_dir_combo.currentText(), self.b_dir_combo.currentText())
        direction = "+Z" if k_hat[2] > 0 else "-Z"
        # Adjust the direction widgets without their handlers each redrawing.
        self._is_updating = True
        try:
            set_choice(self.dir_mode_combo, DirectionMode.AXIS)
            self.wave_prop_dir_combo.setCurrentText(direction)
        finally:
            self._is_updating = False
        self._sync_dir_visibility()
        self.field_direction_status.setText(f"E ⊥ B;  E × B = {direction}")
        self.field_direction_status.setStyleSheet("font-size: 8pt; color: #33aa55;")
        self._on_param_changed()

    def _on_propagation_changed(self):
        if self._is_updating:
            return
        direction = self.wave_prop_dir_combo.currentText()
        enabled = (direction in ("+Z", "-Z")
                   and choice(self.dir_mode_combo, DirectionMode) == DirectionMode.AXIS)
        self.e_dir_combo.setEnabled(enabled)
        self.b_dir_combo.setEnabled(enabled)
        if enabled:
            desired = 1 if direction == "+Z" else -1
            self._is_updating = True   # one redraw below, not another from the B combo
            try:
                for candidate in [self.b_dir_combo.itemText(i) for i in range(self.b_dir_combo.count())]:
                    if np.sign(field_directions(self.e_dir_combo.currentText(), candidate)[0][2]) == desired:
                        self.b_dir_combo.setCurrentText(candidate)
                        break
            finally:
                self._is_updating = False
            self.field_direction_status.setText(f"E ⊥ B;  E × B = {direction}")
        else:
            self.field_direction_status.setText("Вибір ±X/±Y доступний для потоку вздовж ±Z")
        self._on_param_changed()

    def _on_dir_mode_changed(self, _text):
        self._sync_dir_visibility()
        self._on_param_changed()

    def _sync_dir_visibility(self):
        is_custom = choice(self.dir_mode_combo, DirectionMode) == DirectionMode.VECTOR
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

    def _on_separation_changed(self, value):
        if self._is_updating:
            return
        self.configuration.source_separation = value
        camera.look_at(self.view, self._camera_target(SUPERPOSITION_TAB))
        self.update_plot()

    def _on_repr_changed(self, _text):
        self.update_plot()

    def _on_background_changed(self, _index=None):
        self.axes.set_background(choice(self.bg_combo, Background))

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    def _apply_camera(self, tab):
        """With the "3D" projection, Tabs 1 and 2 use their fixed cameras:
        Tab 1 — X up, Z left-to-right, Y toward the viewer;
        Tab 2 — X down, Y away from the viewer, Z left-to-right;
        for a ±X/±Y wave the same layout is turned onto its axis (camera.rotation_along).
        The superposition, and every tab under XY/XZ/YZ, uses the projection's
        elevation/azimuth camera. Either way it looks at the tab's _camera_target."""
        projection = choice(self.proj_combo, Projection)
        target = self._camera_target(tab)
        if projection == Projection.PERSPECTIVE and tab != SUPERPOSITION_TAB:
            camera.set_fixed_camera(self.view, self._wave_rotation(tab), target)
        else:
            camera.set_default_camera(self.view, projection)
            camera.look_at(self.view, target)

    def _wave_rotation(self, tab):
        try:
            return camera.rotation_along(wave_frame(self.scenario_params[tab])[0],
                                         upside_down=tab == WAVE2_TAB)
        except ValueError:   # the redraw reports the invalid direction
            return camera.WAVE1_ROTATION if tab == WAVE1_TAB else camera.WAVE2_ROTATION

    def _camera_target(self, tab):
        """Tab 1: a quarter of the axis length (wave plus margin) along k from the
        origin, leaving more room for the wave than for the space behind its source.
        Tab 2: the middle of wave 2, wherever its source and direction put it.
        Superposition: the midpoint between the sources."""
        if tab == SUPERPOSITION_TAB:
            source2 = place_source2(self.scenario_params[1], self.configuration.source_separation)
            return np.asarray(source2.source_position) / 2
        params = self.scenario_params[tab]
        try:
            k_hat = wave_frame(params)[0]
        except ValueError:
            k_hat = np.zeros(3)   # the redraw reports the invalid direction
        if tab == WAVE1_TAB:
            return k_hat * (params.periods + 2) * 0.25
        return np.asarray(params.source_position, dtype=float) + k_hat * params.periods / 2

    def _on_projection_changed(self, _index):
        self._apply_camera(self.scenario_tabs.currentIndex())

    def _distance_for_pxw(self, px_per_wavelength):
        return camera.distance_for_pxw(self.view, px_per_wavelength)

    def _pixels_per_unit(self):
        return camera.pixels_per_unit(self.view)

    def _on_pxw_changed(self):
        self.view.opts['distance'] = self._distance_for_pxw(self.pxw_spin.value())
        self.view.update()
        if self._paused:
            self.update_plot()   # re-thin the tick labels for the new zoom

    def eventFilter(self, obj, event):
        if obj is self.view and event.type() in (QEvent.Type.Wheel, QEvent.Type.Resize):
            # The view applies the zoom after this filter returns, so sync afterwards.
            QTimer.singleShot(0, self._sync_pxw_from_camera)
        return super().eventFilter(obj, event)

    def _sync_pxw_from_camera(self):
        """Show the camera's actual px/λ in the spinbox without moving the camera."""
        if not self._camera_ready:
            return   # the first frame sets the camera from the spinbox, not the reverse
        pxw = round(self._pixels_per_unit())
        pxw = min(max(pxw, self.pxw_spin.minimum()), self.pxw_spin.maximum())
        if pxw != self.pxw_spin.value():
            self.pxw_spin.blockSignals(True)
            self.pxw_spin.setValue(pxw)
            self.pxw_spin.blockSignals(False)
        if self._paused:
            self.update_plot()

    def _init_camera(self, tab):
        self.view.opts['distance'] = self._distance_for_pxw(self.pxw_spin.value())
        self._apply_camera(tab)
        self._camera_ready = True

    # ------------------------------------------------------------------
    # Parameters save / load, image export
    # ------------------------------------------------------------------

    def _collect_params(self):
        """Serialize all user-configurable state into a plain dict."""
        self.configuration.visualization = VisualizationSettings(
            choice(self.comp_combo, ComponentMode), choice(self.repr_combo, Representation),
            choice(self.bg_combo, Background), choice(self.proj_combo, Projection),
            self.pxw_spin.value(),
        )
        return configuration_to_dict(self.configuration)

    def _on_reset(self):
        """Restore wave/visualization settings to their initial defaults."""
        self._apply_params(configuration_to_dict(ScenarioConfiguration()))
        self.t = 0.0
        self.scenario_tabs.setCurrentIndex(WAVE1_TAB)

    def _apply_params(self, data):
        """Restore scenario_params and UI widgets from a validated dict."""
        self.configuration = configuration_from_dict(data)
        self.scenario_params = self.configuration.waves
        viz = self.configuration.visualization
        self._is_updating = True
        try:
            set_choice(self.comp_combo, viz.component_mode)
            set_choice(self.repr_combo, viz.representation_mode)
            set_choice(self.proj_combo, viz.projection)
            set_choice(self.bg_combo, viz.background)
            pxw = viz.pixels_per_wavelength
            if self.pxw_spin.minimum() <= pxw <= self.pxw_spin.maximum():
                self.pxw_spin.setValue(int(pxw))
        finally:
            self._is_updating = False

        # Full UI re-sync for whichever tab is active
        self._on_scenario_changed(self.scenario_tabs.currentIndex())

    def _show_status(self, text, ok):
        self.params_status.setText(text)
        self.params_status.setStyleSheet(
            f"font-size: 8pt; color: {'#33dd33' if ok else '#ff5555'};")

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
            self._show_status(f"✓ {pathlib.Path(path).name}", ok=True)
        except Exception as exc:
            self._show_status(f"Помилка збереження: {exc}", ok=False)

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
            self._show_status("Помилка читання", ok=False)
            return
        try:
            configuration_from_dict(data)
        except ConfigurationError as exc:
            QMessageBox.warning(self, "Некоректні дані", f"Файл містить помилки:\n• {exc}")
            self._show_status("Помилка перевірки даних", ok=False)
            return
        self._apply_params(data)
        self._show_status(f"✓ {pathlib.Path(path).name}", ok=True)

    def _on_save_image(self):
        path, selected = QFileDialog.getSaveFileName(
            self, "Зберегти малюнок", "", "PNG (*.png);;JPEG (*.jpg *.jpeg)")
        if path:
            self._save_image(path, ".jpg" if selected.startswith("JPEG") else ".png")

    def _save_image(self, path, default_suffix=".png"):
        """Save a screenshot of the 3D view + data table. A path without an image
        extension gets default_suffix, since the extension picks the format."""
        if pathlib.Path(path).suffix.lower() not in (".png", ".jpg", ".jpeg"):
            path += default_suffix
        plot_pixmap = self.view.grab()
        table_pixmap = self.data_table.grab()
        combined_width = max(plot_pixmap.width(), table_pixmap.width())
        combined_height = plot_pixmap.height() + table_pixmap.height() + 40

        final_pixmap = QPixmap(combined_width, combined_height)
        final_pixmap.fill(QColor("white"))
        painter = QPainter(final_pixmap)
        painter.drawPixmap(0, 0, plot_pixmap)
        painter.drawPixmap(0, plot_pixmap.height() + 20, table_pixmap)
        painter.setPen(QColor("black"))
        painter.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        painter.drawText(10, plot_pixmap.height() + 15, "Таблиця розрахунків:")
        painter.end()

        name = pathlib.Path(path).name
        if final_pixmap.save(path):
            self._show_status(f"✓ {name}", ok=True)
        else:
            self._show_status(f"Не вдалося зберегти малюнок {name}", ok=False)
        return path

    # ------------------------------------------------------------------
    # Redraw
    # ------------------------------------------------------------------

    def update_plot(self):
        if self._is_updating or not self.isVisible():
            return
        self._is_updating = True
        try:
            if not self._paused:
                self.t += TIME_STEP / CARRIER_FREQUENCY
            frame = self._simulate_frame()
            if not self._camera_ready:
                self._init_camera(frame.tab)
            if frame.tab == SUPERPOSITION_TAB:
                self.time_status.setText(time_text(frame.extent, frame.periods))
            self._draw_axes(frame)
            self.indicators.draw(frame)
            self._fill_data_table(frame)
            style = self._draw_style()
            self.fields.draw(frame, style)
            self.meeting_arrows.draw(frame, style)
            if self._calculation_failed:
                # The parameters are valid again: drop the stale error message.
                self._calculation_failed = False
                self.params_status.setText("—")
                self.params_status.setStyleSheet("font-size: 8pt; color: gray;")
        except ValueError as exc:
            self._calculation_failed = True
            self._show_status(f"Помилка обчислення: {exc}", ok=False)
        finally:
            self._is_updating = False

    def _simulate_frame(self) -> SceneFrame:
        """All field data comes from simulation.py; the scene only lays it out."""
        tab = self.scenario_tabs.currentIndex()
        frequency = CARRIER_FREQUENCY
        p1, p2 = self.scenario_params
        if tab == SUPERPOSITION_TAB:
            # Wave 1 starts at the origin and wave 2 at the configured separation; their
            # directions follow the line between the sources. Both switch on at t = 0,
            # so each front leaves its own source.
            source2_params = place_source2(p2, self.configuration.source_separation)
            combined = simulate_counterpropagating(
                p1, source2_params, self.t, frequency=frequency, steps=self.z_steps,
                switch_on_time=0.0)
            # The fields exactly at the meeting point: sample 1 of [source 1, midpoint,
            # source 2]. The 200-sample grid has no sample there.
            meeting = simulate_counterpropagating(
                p1, source2_params, self.t, frequency=frequency, steps=3,
                switch_on_time=0.0)
            wave1, wave2 = combined.wave1, combined.wave2
            source2 = np.asarray(source2_params.source_position)
            extent = self.configuration.source_separation
            z_max = extent + 1.0
        else:
            # Tab 1 is the wave from the origin, whatever source a loaded file names.
            combined = meeting = None
            wave1 = simulate_wave(replace(p1, source_position=(0.0, 0.0, 0.0)),
                                  self.t, frequency=frequency, steps=self.z_steps)
            wave2 = simulate_wave(p2, self.t, frequency=frequency, steps=self.z_steps)
            source2 = np.asarray(p2.source_position, dtype=float)
            extent = (p1 if tab == WAVE1_TAB else p2).periods
            # The coordinate system covers both waves' extents
            z_max = max(p1.periods, np.linalg.norm(source2) + p2.periods) * 1.15
        return SceneFrame(tab, wave1, wave2, combined, source2, extent, z_max,
                          self.t * frequency, C / frequency, meeting)

    def _draw_axes(self, frame):
        # Each axis reaches at least 2× the peak field excursion, and covers the shown
        # wave on whichever side of the origin it runs (plus a 2 λ margin). Every arrow
        # points to its + direction, so the axes stay right-handed under every camera.
        p1, p2 = self.scenario_params
        minimum = 2 * max(p1.amplitude, p2.amplitude, 1e-12) * FIELD_VIEW_SCALE
        # The superposition's wave 1 already spans the whole gap between the sources.
        wave = frame.wave2 if frame.tab == WAVE2_TAB else frame.wave1
        plus, minus = axis_extents(frame.view_positions(wave), minimum)
        notes = field_axis_notes(wave.frame) if frame.tab != SUPERPOSITION_TAB else ("",) * 3
        self.axes.draw(plus, minus, notes, px_per_unit=self._pixels_per_unit(),
                       grid_extent=frame.z_max)

    def _draw_style(self) -> DrawStyle:
        component = choice(self.comp_combo, ComponentMode)
        return DrawStyle(
            linear=choice(self.repr_combo, Representation) == Representation.LINEAR,
            show_e=component in (ComponentMode.COMBINED, ComponentMode.ELECTRIC),
            show_b=component in (ComponentMode.COMBINED, ComponentMode.MAGNETIC),
            show_s=component == ComponentMode.COMBINED,
            overlap_waves=self.overlap_waves_check.isChecked(),
        )

    def _fill_data_table(self, frame):
        # The rows are evaluated on their own grid, exactly at each labelled point: the
        # drawn curve's samples fall on tenths of λ only for some segment lengths.
        p1, p2 = self.scenario_params
        if frame.tab == SUPERPOSITION_TAB:
            # Tenths of the gap between the sources
            result = simulate_counterpropagating(
                p1, place_source2(p2, self.configuration.source_separation), self.t,
                frequency=CARRIER_FREQUENCY, steps=TABLE_ROWS, switch_on_time=0.0)
            coordinates = [frame.extent * i / 10 for i in range(TABLE_ROWS)]
        else:
            # Rows every λ/10 over the first λ from the source
            params = p1 if frame.tab == WAVE1_TAB else p2
            result = simulate_wave(replace(params, periods=1), self.t,
                                   frequency=CARRIER_FREQUENCY, steps=TABLE_ROWS)
            coordinates = [i / 10 for i in range(TABLE_ROWS)]
        rows = np.hstack([result.electric, result.magnetic * 1e9,
                          result.poynting])   # B in nT
        labels = ["0" if i == 0 else f"{c:.2g}λ" for i, c in enumerate(coordinates)]
        fill_table(self.data_table, rows, labels)
