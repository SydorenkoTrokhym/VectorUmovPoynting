import os
import sys
import tempfile
import unittest
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt6.QtCore import QPoint, QPointF, Qt, qInstallMessageHandler
from PyQt6.QtGui import QQuaternion, QVector3D, QWheelEvent
from PyQt6.QtWidgets import QApplication

from controls import set_choice
from geometry import wave_frame
import camera
from main_window import MainWindow
from models import Background, ComponentMode, DirectionMode, Projection, Representation
from physics import C, calculate_fields
from scene import FIELD_VIEW_SCALE
from simulation import simulate_counterpropagating, simulate_wave

WAVELENGTH = C / 1.0   # the window's carrier is fixed at 1 Hz


class MainWindowSmokeTests(unittest.TestCase):
    """Build the real window and drive it through every tab and mode.

    Exceptions raised inside Qt slots never reach the test directly, so they are
    captured through sys.excepthook and asserted on afterwards.
    """

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.errors = []
        self._previous_hook = sys.excepthook
        sys.excepthook = lambda *exc_info: self.errors.append(exc_info)
        self.window = MainWindow()
        self.window.show()
        self.app.processEvents()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        sys.excepthook = self._previous_hook

    def assert_no_errors(self):
        messages = [f"{exc_type.__name__}: {exc}" for exc_type, exc, _ in self.errors]
        self.assertEqual(messages, [])
        self.assertFalse(self.window.params_status.text().startswith("Помилка"),
                         self.window.params_status.text())

    def test_initial_render(self):
        self.window.update_plot()
        self.assert_no_errors()

    def test_every_tab_and_representation_renders(self):
        for representation in Representation:
            set_choice(self.window.repr_combo, representation)
            for index in (0, 1, 2, 0):
                with self.subTest(representation=representation, tab=index):
                    self.window.scenario_tabs.setCurrentIndex(index)
                    self.app.processEvents()
                    self.window.update_plot()
                    self.assert_no_errors()

    def test_drawn_curves_match_simulation(self):
        window = self.window
        window.t = 0.137
        p1, p2 = window.scenario_params
        superposition = simulate_counterpropagating(
            p1, replace(p2, source_position=(0.0, 0.0, 8.0)), window.t,
            steps=window.z_steps, switch_on_time=0.0)
        cases = (
            (0, window.fields.waves[0].electric_curve, simulate_wave(p1, window.t, steps=window.z_steps)),
            (1, window.fields.waves[1].electric_curve, simulate_wave(p2, window.t, steps=window.z_steps)),
            (2, window.fields.waves[0].electric_curve, superposition.wave1),
            (2, window.fields.waves[1].electric_curve, superposition.wave2),
        )
        for index, curve, field in cases:
            with self.subTest(tab=index, curve=curve):
                window.scenario_tabs.setCurrentIndex(index)
                window.t = 0.137
                window.update_plot()
                expected = (field.positions / WAVELENGTH
                            + field.electric * FIELD_VIEW_SCALE)[field.active]
                np.testing.assert_allclose(curve.pos, expected, atol=1e-9)
        self.assert_no_errors()

    def test_superposition_presets_show_fronts_then_combined_field(self):
        window = self.window
        window.scenario_tabs.setCurrentIndex(2)
        p1, p2 = window.scenario_params
        gap = 8.0   # default distance between the sources, in λ
        self.assertEqual(window.separation_spin.value(), gap)
        p2 = replace(p2, source_position=(0.0, 0.0, gap))
        combined_e = window.fields.combined.electric_curve

        window._on_time_preset("start")
        self.assertAlmostEqual(window.t, 2.0)
        self.assertTrue(window.fields.waves[0].electric_curve.visible() and window.fields.waves[1].electric_curve.visible())
        self.assertFalse(combined_e.visible())   # the fronts have not met yet

        self.assertFalse(any(label.visible() for label in window.indicators.meet_labels))

        window._on_time_preset("meet")
        self.assertAlmostEqual(window.t, 4.0)   # 8 λ apart → fronts meet after 4 T
        self.assertIn("зустрічаються", window.time_status.text())
        meet_labels = {label.text: np.asarray(label.pos)
                       for label in window.indicators.meet_labels if label.visible()}
        self.assertEqual(sorted(meet_labels), ["B₁", "B₂", "E₁", "E₂", "k₁", "k₂"])
        for position in meet_labels.values():
            self.assertLess(np.linalg.norm(position - [0.0, 0.0, gap / 2]), 0.5)
        # The two waves' k vectors at the meeting point face each other.
        self.assertGreater(meet_labels["k₁"][2], gap / 2)
        self.assertLess(meet_labels["k₂"][2], gap / 2)

        window._on_time_preset("later")
        self.assertTrue(window._paused)
        self.assertTrue(combined_e.visible())
        result = simulate_counterpropagating(
            p1, p2, window.t, steps=window.z_steps, switch_on_time=0.0)
        overlap = result.wave1.active & result.wave2.active
        self.assertTrue(overlap.all())
        expected = (result.positions / WAVELENGTH
                    + result.electric * FIELD_VIEW_SCALE)[overlap]
        np.testing.assert_allclose(combined_e.pos, expected, atol=1e-9)

        window._on_time_step(-1)
        self.assertAlmostEqual(window.t, gap + 0.625 - 0.05)
        window.scenario_tabs.setCurrentIndex(0)
        window.update_plot()
        self.assertFalse(combined_e.visible())
        self.assert_no_errors()

    def test_meeting_point_field_arrows_match_the_fields_there(self):
        window = self.window
        window.scenario_tabs.setCurrentIndex(2)
        gap = window.configuration.source_separation
        window._on_time_preset("start")
        self.assertFalse(window.meeting_arrows.wave_arrows.visible())
        self.assertFalse(window.meeting_arrows.sum_arrows.visible())

        window._set_time(gap / 2 + 0.3)   # fronts met 0.3 T ago; fields are non-zero there
        p1, p2 = window.scenario_params
        # Steps 3 puts sample 1 exactly on the meeting point, between two samples
        # of the drawn curves.
        result = simulate_counterpropagating(
            p1, replace(p2, source_position=(0.0, 0.0, gap)), window.t,
            steps=3, switch_on_time=0.0)
        idx = 1
        base = np.array([0.0, 0.0, gap / 2])
        np.testing.assert_allclose(result.positions[idx] / WAVELENGTH, base)
        # Σ arrows start on the axis and end on the combined E/B curves.
        sum_pos = window.meeting_arrows.sum_arrows.pos
        expected_tips = [base + result.electric[idx] * FIELD_VIEW_SCALE,
                         base + result.magnetic[idx] * C * FIELD_VIEW_SCALE]
        tips = [sum_pos[i + 1] for i in range(0, len(sum_pos), 10)]
        np.testing.assert_allclose(tips, expected_tips[:len(tips)], atol=1e-9)
        # Each wave's E arrow has that wave's E there, shifted toward its own source.
        field_pos = window.meeting_arrows.wave_arrows.pos
        e1_start, e1_tip = field_pos[0], field_pos[1]
        np.testing.assert_allclose(e1_tip - e1_start,
                                   result.wave1.electric[idx] * FIELD_VIEW_SCALE, atol=1e-9)
        self.assertLess(e1_start[2], base[2])
        texts = {label.text for label in window.meeting_arrows.labels if label.visible()}
        self.assertTrue({"E₁", "E₂", "EΣ"} <= texts, texts)

        set_choice(window.comp_combo, ComponentMode.MAGNETIC)
        window.update_plot()
        texts = {label.text for label in window.meeting_arrows.labels if label.visible()}
        self.assertFalse(texts & {"E₁", "E₂", "EΣ"})
        window.scenario_tabs.setCurrentIndex(0)
        window.update_plot()
        self.assertFalse(window.meeting_arrows.sum_arrows.visible())
        self.assertFalse(any(label.visible() for label in window.meeting_arrows.labels))
        self.assert_no_errors()

    def test_separation_moves_source_two_and_meeting_time(self):
        window = self.window
        window.scenario_tabs.setCurrentIndex(2)
        self.assertTrue(window.separation_widget.isVisible())
        self.assertFalse(window.source_position_widget.isVisible())
        window.separation_spin.setValue(5.0)
        self.assertEqual(window.configuration.source_separation, 5.0)
        np.testing.assert_allclose(window.indicators.source2_marker.pos, [[0.0, 0.0, 5.0]])
        np.testing.assert_allclose(window.indicators.meeting_marker.pos, [[0.0, 0.0, 2.5]])
        window._on_time_preset("meet")
        self.assertAlmostEqual(window.t, 2.5)
        # Wave 2's own tab keeps its own source position.
        window.scenario_tabs.setCurrentIndex(1)
        self.assertTrue(window.source_position_widget.isVisible())
        self.assertEqual(window.scenario_params[1].source_position, (0.0, 0.0, 2.0))
        self.assert_no_errors()

    def test_grid_follows_coordinate_system_size(self):
        window = self.window
        window.scenario_tabs.setCurrentIndex(2)
        grid_scale = lambda: window.axes.grid.transform().data()[0]
        window.update_plot()
        self.assertAlmostEqual(grid_scale(), (8.0 + 1.0) / 10)
        window.separation_spin.setValue(15.0)
        self.assertAlmostEqual(grid_scale(), (15.0 + 1.0) / 10)

    def test_poynting_tube_radius_is_never_negative(self):
        window = self.window
        set_choice(window.repr_combo, Representation.SURFACE)
        window.scenario_tabs.setCurrentIndex(2)
        window._on_time_preset("later")
        tube = window.fields.combined.poynting_surface
        self.assertTrue(tube.visible())
        p1, p2 = window.scenario_params
        result = simulate_counterpropagating(
            p1, replace(p2, source_position=(0.0, 0.0, 8.0)), window.t,
            steps=window.z_steps, switch_on_time=0.0)
        self.assertTrue((result.poynting[:, 2] < 0).any())   # S does change sign here
        t1 = result.wave1.frame[1]
        axis = result.positions / WAVELENGTH
        # Vertex 0 of each ring sits at +radius along t1; a negative radius flips it.
        ring_start = tube.opts["meshdata"].vertexes()[0::4]
        self.assertTrue(np.all((ring_start - axis) @ t1 >= -1e-6))
        self.assert_no_errors()

    def test_background_change_recolours_tick_labels_at_once(self):
        window = self.window
        window.update_plot()
        set_choice(window.bg_combo, Background.WHITE)
        labels = [label for label in window.axes.tick_labels if label.visible()]
        self.assertTrue(labels)
        for label in labels:
            self.assertEqual(label.color.getRgb()[:3], (0, 0, 0))

    def test_wheel_zoom_updates_pixels_per_wavelength(self):
        window = self.window
        window.update_plot()
        self.assertEqual(window.pxw_spin.value(), 100)
        distance = window.view.opts['distance']
        center = QPointF(window.view.width() / 2, window.view.height() / 2)
        wheel = QWheelEvent(center, window.view.mapToGlobal(center), QPoint(0, 0),
                            QPoint(0, -240), Qt.MouseButton.NoButton,
                            Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
        QApplication.sendEvent(window.view, wheel)
        zoomed = window.view.opts['distance']
        self.assertGreater(zoomed, distance)   # zoomed out
        self.app.processEvents()
        self.assertEqual(window.pxw_spin.value(), round(window._pixels_per_unit()))
        self.assertLess(window.pxw_spin.value(), 100)
        # Syncing the spinbox must not move the camera again.
        self.assertEqual(window.view.opts['distance'], zoomed)
        self.assert_no_errors()

    def test_drawn_axes_are_right_handed_in_every_tab(self):
        window = self.window
        for tab in (0, 1, 2):
            with self.subTest(tab=tab):
                window.scenario_tabs.setCurrentIndex(tab)
                window.update_plot()
                pos = np.asarray(window.axes.arrows.pos)
                x_tip, y_tip, z_tip = pos[1], pos[7], pos[13]
                self.assertGreater(np.linalg.det([x_tip, y_tip, z_tip]), 0)
                self.assertGreater(z_tip[2], 0)   # the arrow labelled Z points to +Z
        self.assert_no_errors()

    def test_wave2_camera_looks_at_the_middle_of_wave2(self):
        window = self.window
        window.scenario_tabs.setCurrentIndex(1)
        # default: source at z = 2, propagating −Z for 2 λ
        np.testing.assert_allclose(list(window.view.opts['center']), [0.0, 0.0, 1.0])

    def test_single_wave_tabs_show_poynting_vector(self):
        window = self.window
        for tab, wave in ((0, 0), (1, 1)):
            with self.subTest(tab=tab):
                window.scenario_tabs.setCurrentIndex(tab)
                set_choice(window.comp_combo, ComponentMode.COMBINED)
                window._set_time(0.3)
                self.assertTrue(window.fields.waves[wave].poynting_vectors.visible())
                self.assertIn("S", window.legend_strip.text())
                set_choice(window.comp_combo, ComponentMode.ELECTRIC)
                self.assertFalse(window.fields.waves[wave].poynting_vectors.visible())
        self.assert_no_errors()

    def test_data_table_shows_b_in_nanotesla(self):
        window = self.window
        window.chk_data.setChecked(True)
        window._set_time(0.3)
        table = window.data_table
        headers = [table.horizontalHeaderItem(j).text() for j in range(table.columnCount())]
        self.assertIn("By(нТл)", headers)
        self.assertFalse(any(h.startswith("H") for h in headers))
        self.assert_table_matches_exact_fields(window.scenario_params[0])

    def assert_table_matches_exact_fields(self, params):
        """Row i must hold the field exactly i·λ/10 from the source, not at a nearby sample."""
        table, t = self.window.data_table, self.window.t
        headers = [table.horizontalHeaderItem(j).text() for j in range(table.columnCount())]
        column = headers.index("By(нТл)")
        _, _, bx, by = calculate_fields(np.arange(11) / 10 * WAVELENGTH, t)
        _, e1, e2 = wave_frame(params)
        exact_y = (np.outer(bx, e1) + np.outer(by, e2))[:, 1] * 1e9
        for row in range(11):
            self.assertEqual(table.item(row, 0).text(), "0" if row == 0 else f"{row / 10:.2g}λ")
            self.assertEqual(float(table.item(row, column).text()),
                             float(f"{exact_y[row]:.2f}") + 0.0, f"row {row}")

    def test_data_table_rows_sit_on_tenths_of_lambda_for_any_length(self):
        window = self.window
        window.chk_data.setChecked(True)
        for periods in (3, 7):
            with self.subTest(periods=periods):
                window.n_periods_spin.setValue(periods)
                window._set_time(0.37)
                self.assert_table_matches_exact_fields(window.scenario_params[0])
        self.assert_no_errors()

    def test_table_refresh_raises_no_qt_warnings(self):
        window = self.window
        window.chk_data.setChecked(True)
        messages = []
        previous = qInstallMessageHandler(lambda _type, _context, text: messages.append(text))
        try:
            for t in (0.1, 0.2, 0.3):   # refresh cells that already exist
                window._set_time(t)
        finally:
            qInstallMessageHandler(previous)
        self.assertEqual([m for m in messages if "QTableWidget" in m], [])
        self.assertEqual(window.data_table.item(0, 0).textAlignment(),
                         Qt.AlignmentFlag.AlignCenter.value)

    def test_calculation_error_clears_once_parameters_are_valid(self):
        window = self.window
        set_choice(window.dir_mode_combo, DirectionMode.VECTOR)
        for spin in (window.dir_x_spin, window.dir_y_spin, window.dir_z_spin):
            spin.setValue(0.0)
        self.assertTrue(window.params_status.text().startswith("Помилка обчислення"))
        window.dir_x_spin.setValue(1.0)
        self.assertEqual(window.params_status.text(), "—")
        self.assert_no_errors()

    def test_axis_labels_name_the_axes_the_fields_lie_along(self):
        window = self.window
        axis_texts = lambda: [label.text for label in window.axes.labels]
        window.update_plot()
        self.assertEqual(axis_texts(), ["X (E)", "Y (B)", "Z"])
        window.wave_prop_dir_combo.setCurrentText("+X")   # frame k = X, E = Y, B = Z
        self.assertEqual(axis_texts(), ["X", "Y (E)", "Z (B)"])
        window.wave_prop_dir_combo.setCurrentText("+Z")
        window.e_dir_combo.setCurrentText("+Y")           # E = Y, B = −X
        self.assertEqual(axis_texts(), ["X (B)", "Y (E)", "Z"])
        window.scenario_tabs.setCurrentIndex(2)
        self.assertEqual(axis_texts(), ["X", "Y", "Z"])
        self.assert_no_errors()

    def test_axes_cover_a_wave_running_toward_minus(self):
        window = self.window
        window.n_periods_spin.setValue(3)
        for direction, axis in (("-Z", 2), ("-X", 0)):
            with self.subTest(direction=direction):
                window.wave_prop_dir_combo.setCurrentText(direction)
                pos = np.asarray(window.axes.arrows.pos)
                start = pos[6 * axis]   # each axis line starts its block of points
                self.assertAlmostEqual(start[axis], -(3 + 2))   # wave plus 2 λ margin
                texts = [label.text for label in window.axes.tick_labels if label.visible()]
                self.assertIn("−3λ", texts)
        self.assert_no_errors()

    def test_wave1_camera_looks_along_the_wave(self):
        window = self.window
        window.wave_prop_dir_combo.setCurrentText("-Z")
        window._apply_camera(0)
        np.testing.assert_allclose(list(window.view.opts['center']), [0.0, 0.0, -1.0])

    def screen(self, vector):
        """Where the view's rotation puts a world direction: x right, y up, z toward the viewer."""
        rotated = self.window.view.opts['rotation'].rotatedVector(QVector3D(*map(float, vector)))
        return np.array([rotated.x(), rotated.y(), rotated.z()])

    def test_wave1_camera_turns_with_the_wave_axis(self):
        window = self.window
        window.update_plot()
        for direction in ("+X", "-X", "+Y", "-Y", "-Z", "+Z"):
            with self.subTest(direction=direction):
                window.wave_prop_dir_combo.setCurrentText(direction)   # no manual camera call
                self.assertEqual(window.view.opts['rotationMethod'], 'quaternion')
                k_hat, e_hat, b_hat = wave_frame(window.scenario_params[0])
                sign = 1.0 if direction[0] == "+" else -1.0
                # The wave runs along the screen's horizontal (right-to-left toward −),
                # E points up and B along the line of sight: still the +Z layout.
                np.testing.assert_allclose(self.screen(k_hat), [sign, 0, 0], atol=1e-6)
                np.testing.assert_allclose(self.screen(e_hat), [0, 1, 0], atol=1e-6)
                np.testing.assert_allclose(self.screen(b_hat), [0, 0, sign], atol=1e-6)
        # Along ±Z it is exactly the original Tab 1 camera.
        for axis in np.eye(3):
            np.testing.assert_allclose(
                self.screen(axis),
                [getattr(camera.WAVE1_ROTATION.rotatedVector(QVector3D(*axis)), c)() for c in "xyz"],
                atol=1e-6)
        self.assert_no_errors()

    def test_wave2_camera_turns_with_the_wave_axis(self):
        window = self.window
        window.scenario_tabs.setCurrentIndex(1)
        window.update_plot()
        for direction in ("+X", "-X", "+Y", "-Y", "+Z", "-Z"):
            with self.subTest(direction=direction):
                window.wave_prop_dir_combo.setCurrentText(direction)   # no manual camera call
                self.assertEqual(window.view.opts['rotationMethod'], 'quaternion')
                k_hat, e_hat, b_hat = wave_frame(window.scenario_params[1])
                sign = 1.0 if direction[0] == "+" else -1.0
                # The −Z layout of Tab 2 turned onto the wave's axis: E points down.
                np.testing.assert_allclose(self.screen(k_hat), [sign, 0, 0], atol=1e-6)
                np.testing.assert_allclose(self.screen(e_hat), [0, -1, 0], atol=1e-6)
                np.testing.assert_allclose(self.screen(b_hat), [0, 0, -sign], atol=1e-6)
        # Along ±Z it is exactly the original Tab 2 camera.
        for axis in np.eye(3):
            np.testing.assert_allclose(
                self.screen(axis),
                [getattr(camera.WAVE2_ROTATION.rotatedVector(QVector3D(*axis)), c)() for c in "xyz"],
                atol=1e-6)
        window.scenario_tabs.setCurrentIndex(0)   # Tab 1 keeps its own wave's camera
        np.testing.assert_allclose(self.screen(wave_frame(window.scenario_params[0])[1]),
                                   [0, 1, 0], atol=1e-6)
        self.assert_no_errors()

    def test_wave1_camera_for_a_tilted_vector_stays_proper(self):
        window = self.window
        set_choice(window.dir_mode_combo, DirectionMode.VECTOR)
        window.dir_x_spin.setValue(1.0)
        window.dir_z_spin.setValue(0.0)
        window.dir_y_spin.setValue(1.0)   # k = (1, 1, 0)/√2
        k_hat = wave_frame(window.scenario_params[0])[0]
        np.testing.assert_allclose(self.screen(k_hat), [1, 0, 0], atol=1e-6)
        columns = np.array([self.screen(axis) for axis in np.eye(3)]).T
        self.assertAlmostEqual(np.linalg.det(columns), 1.0, places=5)   # no mirror image
        self.assert_no_errors()

    def test_other_parameters_keep_a_user_rotated_camera(self):
        window = self.window
        window.update_plot()
        turned = QQuaternion.fromEulerAngles(10.0, 20.0, 30.0)
        window.view.opts['rotation'] = turned
        window.amp_spin.setValue(2.0)
        window.n_periods_spin.setValue(3)
        self.assertEqual(window.view.opts['rotation'], turned)

    def test_projection_applies_in_every_tab_and_survives_tab_changes(self):
        window = self.window
        window.update_plot()
        set_choice(window.proj_combo, Projection.TOP)
        self.assertEqual(window.view.opts['rotationMethod'], 'euler')
        self.assertEqual((window.view.opts['elevation'], window.view.opts['azimuth']), (90, 0))
        window.scenario_tabs.setCurrentIndex(1)
        self.assertEqual(window.view.opts['rotationMethod'], 'euler')
        self.assertEqual(window.view.opts['elevation'], 90)
        # Still looking at the middle of wave 2 (source z = 2, −Z for 2 λ)
        np.testing.assert_allclose(list(window.view.opts['center']), [0.0, 0.0, 1.0])
        set_choice(window.proj_combo, Projection.PERSPECTIVE)   # back to the fixed camera
        self.assertEqual(window.view.opts['rotationMethod'], 'quaternion')
        for axis in np.eye(3):
            np.testing.assert_allclose(
                self.screen(axis),
                [getattr(camera.WAVE2_ROTATION.rotatedVector(QVector3D(*axis)), c)() for c in "xyz"],
                atol=1e-6)
        self.assert_no_errors()

    def test_direction_changes_redraw_once(self):
        window = self.window
        calls = []
        original = window.update_plot
        window.update_plot = lambda: (calls.append(1), original())
        # E = +Y (B becomes +X, k = −Z); B = −X (k = +Z); k = −Z (B back to +X)
        for change, expected in (
                (lambda: window.e_dir_combo.setCurrentText("+Y"), ("+Y", "+X", "-Z")),
                (lambda: window.b_dir_combo.setCurrentText("-X"), ("+Y", "-X", "+Z")),
                (lambda: window.wave_prop_dir_combo.setCurrentText("-Z"), ("+Y", "+X", "-Z"))):
            calls.clear()
            change()
            self.assertEqual(len(calls), 1)
            params = window.scenario_params[0]
            self.assertEqual((params.electric_direction, params.magnetic_direction,
                              params.propagation_direction), expected)
        self.assert_no_errors()

    def test_saved_image_gets_an_extension_and_failures_are_reported(self):
        window = self.window
        window.update_plot()
        with tempfile.TemporaryDirectory() as folder:
            saved = window._save_image(os.path.join(folder, "picture"))
            self.assertTrue(saved.endswith("picture.png"))
            self.assertTrue(os.path.getsize(saved) > 0)
            self.assertEqual(window.params_status.text(), "✓ picture.png")
            jpeg = window._save_image(os.path.join(folder, "photo"), ".jpg")
            self.assertTrue(os.path.exists(jpeg) and jpeg.endswith(".jpg"))
            window._save_image(os.path.join(folder, "missing", "picture.png"))
            self.assertTrue(window.params_status.text().startswith("Не вдалося"))

    def test_overlap_toggle_leaves_only_the_sum_in_the_overlap(self):
        window = self.window
        window.scenario_tabs.setCurrentIndex(2)
        window._set_time(5.0)   # fronts met 1 T ago: 2 λ of overlap in the middle
        p1, p2 = window.scenario_params
        result = simulate_counterpropagating(
            p1, replace(p2, source_position=(0.0, 0.0, 8.0)), window.t,
            steps=window.z_steps, switch_on_time=0.0)
        overlap = result.wave1.active & result.wave2.active
        curve = window.fields.waves[0].electric_curve
        self.assertEqual(len(curve.pos), result.wave1.active.sum())
        self.assertAlmostEqual(curve.color[3], 0.30)   # dimmed under the sum
        window.overlap_waves_check.setChecked(False)
        self.assertEqual(len(curve.pos), (result.wave1.active & ~overlap).sum())
        self.assertAlmostEqual(curve.color[3], 1.0)
        self.assertTrue(window.fields.combined.electric_curve.visible())
        self.assert_no_errors()

    def test_meeting_point_directions_fade_once_fields_grow(self):
        window = self.window
        window.scenario_tabs.setCurrentIndex(2)

        def meeting_e_alphas():
            colors = np.asarray(window.indicators.e_arrows.color)
            return colors[20:, 3]   # after the two source triads (10 points each)

        window._on_time_preset("meet")
        np.testing.assert_allclose(meeting_e_alphas(), 1.0)
        window._on_time_step(+1)
        window._on_time_step(+1)
        np.testing.assert_allclose(meeting_e_alphas(), 0.35)
        self.assert_no_errors()

    def test_animation_step_and_reset(self):
        self.window.pause_button.setChecked(False)
        self.window.update_plot()
        self.window.pause_button.setChecked(True)
        self.window.reset_button.click()
        self.app.processEvents()
        self.assert_no_errors()


if __name__ == "__main__":
    unittest.main()
