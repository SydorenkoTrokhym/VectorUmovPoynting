import os
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from main_window import MainWindow


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

    def test_animation_step_and_reset(self):
        self.window.pause_button.setChecked(False)
        self.window.update_plot()
        self.window.pause_button.setChecked(True)
        self.window.reset_button.click()
        self.app.processEvents()
        self.assert_no_errors()


if __name__ == "__main__":
    unittest.main()
