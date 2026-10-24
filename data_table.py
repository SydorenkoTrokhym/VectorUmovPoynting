"""Fill the E/B/S data table, showing only the components that are non-zero."""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHeaderView, QTableWidgetItem

_HEADERS = ["Ex(В/м)", "Ey(В/м)", "Ez(В/м)",
            "Bx(нТл)", "By(нТл)", "Bz(нТл)",
            "Sx(Вт/м²)", "Sy(Вт/м²)", "Sz(Вт/м²)"]
_FORMATS = [".2f"] * 9


def format_value(value, fmt):
    """Format a value without a minus sign on zero ("-0.00")."""
    formatted = f"{value:{fmt}}"
    if float(formatted) == 0:
        return f"{0.0:{fmt}}"
    return formatted


def fill_table(table, rows, coordinates):
    """rows: (n, 9) array of E (V/m), B (nT), S (W/m²); coordinates: n row labels."""
    rows = np.asarray(rows)
    active = [not np.allclose(rows[:, j], 0, atol=1e-12) for j in range(9)]
    headers = ["Coord"] + [_HEADERS[j] for j in range(9) if active[j]]
    if table.columnCount() != len(headers):
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        header = table.horizontalHeader()
        if header:
            header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
    for i, coordinate in enumerate(coordinates):
        values = [coordinate] + [format_value(rows[i, j], _FORMATS[j])
                                 for j in range(9) if active[j]]
        for column, text in enumerate(values):
            cell = table.item(i, column)
            if cell is None:
                # Only new cells are inserted: re-inserting an item the table already
                # owns is rejected with a Qt warning on every redraw.
                cell = QTableWidgetItem()
                cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                table.setItem(i, column, cell)
            cell.setText(text)
