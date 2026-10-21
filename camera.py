"""Camera placement for the GLViewWidget; the view is in λ units (1 λ = 1 unit)."""
from __future__ import annotations

import math

import pyqtgraph as pg
from PyQt6.QtGui import QQuaternion

from models import Projection

# Camera orientation for Tab 1 ("Хвиля 1"): X up, Z left-to-right, Y toward the viewer.
# A plain elevation/azimuth pair can't reach this exact axis mapping (it's the "wrong-handed"
# 3-cycle of the two reachable via X-then-Z Euler rotation), so it's set as a fixed quaternion.
WAVE1_ROTATION = QQuaternion(0.5, 0.5, 0.5, 0.5)

# Camera orientation for Tab 2 ("Хвиля 2"): X down, Y away from the viewer, Z left-to-right.
# This is a proper rotation, so the axes on screen stay right-handed (X × Y = Z), as the
# E × B = S right-hand rule needs; wave 2 (−Z by default) then runs right-to-left.
WAVE2_ROTATION = QQuaternion(-0.5, 0.5, -0.5, 0.5)

# (elevation, azimuth) for each standard projection
_PROJECTIONS = {
    Projection.PERSPECTIVE: (30, 45),
    Projection.TOP: (90, 0),
    Projection.SIDE: (0, 0),
    Projection.FRONT: (0, 90),
}


def distance_for_pxw(view, px_per_wavelength):
    """Camera distance such that one wavelength (= 1 unit in the λ-normalized
    3D view) spans `px_per_wavelength` screen pixels, given the view's current
    width and field of view."""
    width = max(view.width(), 1)
    fov = math.radians(view.opts.get('fov', 60))
    return width / (px_per_wavelength * 2 * math.tan(fov / 2))


def pixels_per_unit(view):
    """Current on-screen pixels per 3D-view unit (= per λ), given the live
    camera distance — used to thin out axis tick labels so they don't overlap."""
    width = max(view.width(), 1)
    dist = max(view.opts.get('distance', 1.0), 1e-9)
    fov = math.radians(view.opts.get('fov', 60))
    return width / (dist * 2 * math.tan(fov / 2))


def set_projection(view, projection: Projection):
    """Apply a projection's elevation/azimuth, keeping the current distance."""
    elevation, azimuth = _PROJECTIONS[projection]
    view.setCameraPosition(distance=view.opts['distance'], elevation=elevation, azimuth=azimuth)


def set_default_camera(view, projection: Projection):
    """Leave a fixed quaternion camera for the standard elevation/azimuth one."""
    view.opts['rotationMethod'] = 'euler'
    set_projection(view, projection)


def set_fixed_camera(view, rotation: QQuaternion, center):
    """Fixed quaternion camera looking at the given point."""
    view.opts['rotationMethod'] = 'quaternion'
    view.opts['rotation'] = rotation
    view.opts['center'] = pg.Vector(*center)
    view.update()


def look_at(view, point):
    view.opts['center'] = pg.Vector(*point)
    view.update()
