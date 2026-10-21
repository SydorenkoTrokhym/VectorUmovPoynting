import logging
import sys

from PyQt6.QtWidgets import QApplication, QMessageBox

from main_window import MainWindow


def report_unhandled_exception(exc_type, exc_value, traceback) -> None:
    """Log uncaught Qt callback failures and report them without blocking stdin."""
    logging.getLogger(__name__).critical(
        "Unhandled application error",
        exc_info=(exc_type, exc_value, traceback),
    )
    QMessageBox.critical(
        None,
        "Критична помилка",
        "Сталася неочікувана помилка. Докладні відомості записано в журнал.",
    )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app = QApplication(sys.argv)
    sys.excepthook = report_unhandled_exception
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
