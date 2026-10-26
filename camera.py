"""Camera placement for the GLViewWidget; the view is in λ units (1 λ = 1 unit)."""
from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PyQt6.QtGui import QMatrix4x4, QQuaternion, QVector3D

from models import Projection

# A straight side view puts B along the line of sight, where it shrinks to a dot, so
# the single-wave cameras look at the side view from a little above and to the side:
# turned 30° about the screen's vertical, then 20° about its horizontal.
SIDE_VIEW_TILT = (QQuaternion.fromAxisAndAngle(1, 0, 0, 20)
                  * QQuaternion.fromAxisAndAngle(0, 1, 0, -30))

# Camera orientation for Tab 1 ("Хвиля 1"): X up, Z left-to-right, Y toward the viewer,
# tilted. A plain elevation/azimuth pair can't reach this axis mapping (it's the
# "wrong-handed" 3-cycle of the two reachable via X-then-Z Euler rotation), so it's set
# as a fixed quaternion.
WAVE1_ROTATION = SIDE_VIEW_TILT * QQuaternion(0.5, 0.5, 0.5, 0.5)

# Camera orientation for Tab 2 ("Хвиля 2"): X down, Y away from the viewer, Z left-to-right,
# tilted. This is a proper rotation, so the axes on screen stay right-handed (X × Y = Z),
# as the E × B = S right-hand rule needs; wave 2 (−Z by default) then runs right-to-left.
WAVE2_ROTATION = SIDE_VIEW_TILT * QQuaternion(-0.5, 0.5, -0.5, 0.5)

# Tabs 1 and 2 for a wave along another axis: that axis takes Z's place (left-to-right)
# and the axis its E lies along takes X's place (up on Tab 1, down on Tab 2).
# Indexed by the wave's main axis.
_UP_FOR_AXIS = (np.array([0., 1., 0.]),   # along X: Y up (Z toward the viewer)
                np.array([1., 0., 0.]),   # along Y: X up (Z away from the viewer)
                np.array([1., 0., 0.]))   # along Z: X up (Y toward the viewer)

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


def screen_points(view, points):
    """Where the current camera puts 3D points in the view, in pixels from its top left
    (as GLTextItem places its text)."""
    viewport = view.getViewport()
    to_pixels = QMatrix4x4()
    to_pixels.viewport(0, viewport[3], viewport[2], -viewport[3])
    transform = to_pixels * view.projectionMatrix(viewport, viewport) * view.viewMatrix()
    mapped = (transform.map(QVector3D(*map(float, point))) for point in points)
    return np.array([(p.x(), p.y()) for p in mapped]).reshape(-1, 2)


def set_projection(view, projection: Projection):
    """Apply a projection's elevation/azimuth, keeping the current distance."""
    elevation, azimuth = _PROJECTIONS[projection]
    view.setCameraPosition(distance=view.opts['distance'], elevation=elevation, azimuth=azimuth)


def set_default_camera(view, projection: Projection):
    """Leave a fixed quaternion camera for the standard elevation/azimuth one."""
    view.opts['rotationMethod'] = 'euler'
    set_projection(view, projection)


def rotation_along(k_hat, upside_down=False) -> QQuaternion:
    """Camera for a wave travelling along k_hat: the + direction of its main axis runs
    left-to-right, the E axis of that direction points up (down if upside_down), and
    right × up points toward the viewer, so the rotation is proper and the axes stay
    right-handed; then SIDE_VIEW_TILT. Along ±Z this is WAVE1_ROTATION, or
    WAVE2_ROTATION upside down."""
    k_hat = np.asarray(k_hat, dtype=float)
    axis = int(np.argmax(np.abs(k_hat)))
    right = k_hat * np.sign(k_hat[axis])          # a wave along −k runs right-to-left
    up = _UP_FOR_AXIS[axis] * (-1.0 if upside_down else 1.0)
    up = up - np.dot(up, right) * right
    up /= np.linalg.norm(up)
    toward = np.cross(right, up)
    # fromAxes maps the camera's x/y/z to these world directions; the view needs
    # the opposite mapping, world → camera.
    side_view = QQuaternion.fromAxes(*(QVector3D(*map(float, v)) for v in (right, up, toward))
                                     ).conjugated()
    return SIDE_VIEW_TILT * side_view


def set_fixed_camera(view, rotation: QQuaternion, center):
    """Fixed quaternion camera looking at the given point."""
    view.opts['rotationMethod'] = 'quaternion'
    view.opts['rotation'] = rotation
    view.opts['center'] = pg.Vector(*center)
    view.update()


def look_at(view, point):
    view.opts['center'] = pg.Vector(*point)
    view.update()
