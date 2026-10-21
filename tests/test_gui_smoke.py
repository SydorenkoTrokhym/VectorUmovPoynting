import os
import sys
import unittest
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtWidgets import QApplication

from main_window import FIELD_VIEW_SCALE, MainWindow
from physics import C
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
        for representation in ("Linear (3D)", "Surface (3D)"):
            self.window.repr_combo.setCurrentText(representation)
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
            (0, window.e_curve, simulate_wave(p1, window.t, steps=window.z_steps)),
            (1, window.e_curve2, simulate_wave(p2, window.t, steps=window.z_steps)),
            (2, window.e_curve, superposition.wave1),
            (2, window.e_curve2, superposition.wave2),
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
        combined_e = window.combined_renderer.electric_curve

        window._on_time_preset("start")
        self.assertAlmostEqual(window.t, 2.0)
        self.assertTrue(window.e_curve.visible() and window.e_curve2.visible())
        self.assertFalse(combined_e.visible())   # the fronts have not met yet

        self.assertFalse(any(label.visible() for label in window.meet_labels))

        window._on_time_preset("meet")
        self.assertAlmostEqual(window.t, 4.0)   # 8 λ apart → fronts meet after 4 T
        self.assertIn("зустрічаються", window.time_status.text())
        meet_labels = {label.text: np.asarray(label.pos)
                       for label in window.meet_labels if label.visible()}
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
        self.assertFalse(window.meet_field_arrows.visible())
        self.assertFalse(window.meet_sum_arrows.visible())

        window._set_time(gap / 2 + 0.3)   # fronts met 0.3 T ago; fields are non-zero there
        p1, p2 = window.scenario_params
        result = simulate_counterpropagating(
            p1, replace(p2, source_position=(0.0, 0.0, gap)), window.t,
            steps=window.z_steps, switch_on_time=0.0)
        positions = result.positions / WAVELENGTH
        idx = int(np.argmin(np.linalg.norm(positions - [0.0, 0.0, gap / 2], axis=1)))
        base = positions[idx]
        # Σ arrows start on the axis and end on the combined E/B curves.
        sum_pos = window.meet_sum_arrows.pos
        expected_tips = [base + result.electric[idx] * FIELD_VIEW_SCALE,
                         base + result.magnetic[idx] * C * FIELD_VIEW_SCALE]
        tips = [sum_pos[i + 1] for i in range(0, len(sum_pos), 10)]
        np.testing.assert_allclose(tips, expected_tips[:len(tips)], atol=1e-9)
        # Each wave's E arrow has that wave's E there, shifted toward its own source.
        field_pos = window.meet_field_arrows.pos
        e1_start, e1_tip = field_pos[0], field_pos[1]
        np.testing.assert_allclose(e1_tip - e1_start,
                                   result.wave1.electric[idx] * FIELD_VIEW_SCALE, atol=1e-9)
        self.assertLess(e1_start[2], base[2])
        texts = {label.text for label in window.meet_field_labels if label.visible()}
        self.assertTrue({"E₁", "E₂", "EΣ"} <= texts, texts)

        window.comp_combo.setCurrentText("Магнітна (B)")
        window.update_plot()
        texts = {label.text for label in window.meet_field_labels if label.visible()}
        self.assertFalse(texts & {"E₁", "E₂", "EΣ"})
        window.scenario_tabs.setCurrentIndex(0)
        window.update_plot()
        self.assertFalse(window.meet_sum_arrows.visible())
        self.assertFalse(any(label.visible() for label in window.meet_field_labels))
        self.assert_no_errors()

    def test_separation_moves_source_two_and_meeting_time(self):
        window = self.window
        window.scenario_tabs.setCurrentIndex(2)
        self.assertTrue(window.separation_widget.isVisible())
        self.assertFalse(window.source_position_widget.isVisible())
        window.separation_spin.setValue(5.0)
        self.assertEqual(window.configuration.source_separation, 5.0)
        np.testing.assert_allclose(window.source2_marker.pos, [[0.0, 0.0, 5.0]])
        np.testing.assert_allclose(window.meeting_marker.pos, [[0.0, 0.0, 2.5]])
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
        grid_scale = lambda: window.grid.transform().data()[0]
        window.update_plot()
        self.assertAlmostEqual(grid_scale(), (8.0 + 1.0) / 10)
        window.separation_spin.setValue(15.0)
        self.assertAlmostEqual(grid_scale(), (15.0 + 1.0) / 10)

    def test_poynting_tube_radius_is_never_negative(self):
        window = self.window
        window.repr_combo.setCurrentText("Surface (3D)")
        window.scenario_tabs.setCurrentIndex(2)
        window._on_time_preset("later")
        tube = window.combined_renderer.poynting_surface
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
        window.bg_combo.setCurrentText("Білий")
        labels = [label for label in window.tick_label_pool if label.visible()]
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

    def test_animation_step_and_reset(self):
        self.window.pause_button.setChecked(False)
        self.window.update_plot()
        self.window.pause_button.setChecked(True)
        self.window.reset_button.click()
        self.app.processEvents()
        self.assert_no_errors()


if __name__ == "__main__":
    unittest.main()
