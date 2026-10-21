from __future__ import annotations

import math

import pyqtgraph as pg
from PyQt6.QtGui import QColor, QFont, QPainter, QPixmap, QQuaternion
from PyQt6.QtWidgets import QFileDialog

BG_COLORS = {
    "Чорний": {"gl": "k", "grid": (0.3, 0.3, 0.3, 1.0)},
    "Білий":  {"gl": "w", "grid": (0.5, 0.5, 0.5, 1.0)},
}


def txt_rgba(c):
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


def distance_for_pxw(window, px_per_wavelength):
    """Camera distance such that one wavelength (= 1 unit in the λ-normalized
    3D view) spans `px_per_wavelength` screen pixels, given the view's current
    width and field of view."""
    width = max(window.view.width(), 1)
    fov = math.radians(window.view.opts.get('fov', 60))
    return width / (px_per_wavelength * 2 * math.tan(fov / 2))


def pixels_per_unit(window):
    """Current on-screen pixels per 3D-view unit (= per λ), given the live
    camera distance — used to thin out axis tick labels so they don't overlap."""
    width = max(window.view.width(), 1)
    dist = max(window.view.opts.get('distance', 1.0), 1e-9)
    fov = math.radians(window.view.opts.get('fov', 60))
    return width / (dist * 2 * math.tan(fov / 2))


def on_pxw_changed(window):
    window.view.opts['distance'] = distance_for_pxw(window, window.pxw_spin.value())
    window.view.update()


def on_projection_changed(window, index):
    dist = window.view.opts['distance']
    if index == 0:    # 3D perspective
        window.view.setCameraPosition(distance=dist, elevation=30, azimuth=45)
    elif index == 1:  # XY top view
        window.view.setCameraPosition(distance=dist, elevation=90, azimuth=0)
    elif index == 2:  # XZ side view
        window.view.setCameraPosition(distance=dist, elevation=0, azimuth=0)
    elif index == 3:  # YZ front view
        window.view.setCameraPosition(distance=dist, elevation=0, azimuth=90)


def apply_wave1_camera(window):
    """Fixed camera for Tab 1: X up, Z left-to-right, Y toward the viewer.
    The look-at center is offset along +Z so the origin sits a quarter of the
    Z-axis length left of screen-middle, leaving more room for the wave."""
    window.view.opts['rotationMethod'] = 'quaternion'
    window.view.opts['rotation'] = _WAVE1_CAMERA_ROTATION
    z_axis_len = window.scenario_params[0].periods + 2
    window.view.opts['center'] = pg.Vector(0, 0, z_axis_len * 0.25)
    window.view.update()


def apply_wave2_camera(window):
    """Fixed camera for Tab 2: X down, Y away from the viewer, Z right-to-left.
    The look-at center is offset along +Z so the origin sits a sixth of the
    screen width to the left, leaving room on the right for the wave to develop."""
    window.view.opts['rotationMethod'] = 'quaternion'
    window.view.opts['rotation'] = _WAVE2_CAMERA_ROTATION
    z_axis_len = window.scenario_params[1].periods + 2
    window.view.opts['center'] = pg.Vector(0, 0, z_axis_len * (1.0 / 6.0))
    window.view.update()


def apply_default_camera(window):
    """Restore the standard elevation/azimuth camera (driven by proj_combo)."""
    window.view.opts['rotationMethod'] = 'euler'
    on_projection_changed(window, window.proj_combo.currentIndex())


def apply_background(window, name: str):
    cfg = BG_COLORS.get(name, BG_COLORS["Чорний"])
    window.view.setBackgroundColor(cfg["gl"])
    window.grid.setColor(cfg["grid"])

    window._txt_color = txt_rgba((0.0, 0.0, 0.0, 1.0) if name == "Білий" else (1.0, 1.0, 1.0, 1.0))
    tick_color = (0.3, 0.3, 0.3, 0.7) if name == "Білий" else (0.5, 0.5, 0.5, 0.6)
    window.tick_marks.setData(color=tick_color)
    # Axis label colors are fixed per-axis (X/Y/Z) — no background adaptation needed


def on_save_image(window):
    """Save screenshot of the 3D view + data table."""
    # Grab plot image
    plot_pixmap = window.view.grab()

    # Grab table image
    table_pixmap = window.data_table.grab()

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

    path, _ = QFileDialog.getSaveFileName(window, "Зберегти малюнок", "", "Images (*.png *.jpg)")
    if path:
        final_pixmap.save(path)