"""Colours shared by the 3D scene and the legend.

Palette rule: hue = physical quantity, shade = which wave. E is always blue, B red,
S (energy flux) gold and k green, so a quantity keeps its hue in every tab; wave 2
is the lighter shade and the combined field a distinct strong variant of each hue.
Every colour must read on both the black and the white background.
"""
from __future__ import annotations

from models import Background

W1_E = (0.10, 0.35, 1.00, 1.0)   # blue        — wave 1 electric field
W1_B = (1.00, 0.15, 0.15, 1.0)   # red         — wave 1 magnetic field
W1_S = (0.85, 0.60, 0.00, 1.0)   # amber       — wave 1 Poynting
W1_K = (0.10, 0.75, 0.30, 1.0)   # green       — wave 1 propagation
W2_E = (0.35, 0.80, 1.00, 1.0)   # sky blue    — wave 2 electric field
W2_B = (1.00, 0.55, 0.55, 1.0)   # salmon      — wave 2 magnetic field
W2_S = (0.90, 0.90, 0.30, 1.0)   # pale yellow — wave 2 Poynting
W2_K = (0.60, 1.00, 0.40, 1.0)   # light green — wave 2 propagation

# Combined field in the superposition tab — opaque and thicker; the individual waves
# are dimmed underneath it. E Σ leans to violet and B Σ to pink so neither can be
# mistaken for wave 1 or wave 2 of the same quantity.
SUM_E = (0.60, 0.40, 1.00, 1.0)  # violet      — E₁ + E₂
SUM_B = (1.00, 0.15, 0.60, 1.0)  # hot pink    — B₁ + B₂
SUM_S = (1.00, 0.80, 0.00, 1.0)  # gold        — S of the combined field

# Coordinate axes and markers are neutral grey: every hue is taken by a field quantity,
# and the X/Y/Z labels already tell the axes apart.
NEUTRAL = (0.60, 0.60, 0.60, 1.0)
MEETING_MARKER = (0.8, 0.8, 0.8, 0.8)

# Per wave: (E, B, S, k)
WAVE_COLORS = ((W1_E, W1_B, W1_S, W1_K), (W2_E, W2_B, W2_S, W2_K))
SUM_COLORS = (SUM_E, SUM_B, SUM_S)

BACKGROUNDS = {
    Background.BLACK: {"gl": "k", "grid": (0.3, 0.3, 0.3, 1.0),
                       "text": (1.0, 1.0, 1.0, 1.0), "ticks": (0.5, 0.5, 0.5, 0.6)},
    Background.WHITE: {"gl": "w", "grid": (0.5, 0.5, 0.5, 1.0),
                       "text": (0.0, 0.0, 0.0, 1.0), "ticks": (0.3, 0.3, 0.3, 0.7)},
}


def txt_rgba(c):
    """GLTextItem routes its color through pg.mkColor(), which treats a 4-tuple
    as 0-255 ints — unlike GLLinePlotItem/GLMeshItem, which take 0-1 floats for
    OpenGL vertex colors. Our palette constants are all 0-1 floats, so any color
    handed to a GLTextItem must be rescaled here or it renders as near-invisible
    (truncated to 0 and alpha ~1/255)."""
    return tuple(int(round(v * 255)) for v in c)


def to_hex(color) -> str:
    r, g, b = color[:3]
    return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
