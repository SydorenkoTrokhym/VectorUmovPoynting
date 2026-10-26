"""The 3D scene: axes, direction indicators, field layers and meeting-point arrows.

Every layer owns its GL items and is redrawn from a SceneFrame. Positions are in λ
units (1 λ = 1 view unit), not raw SI metres: at 1 Hz λ = 3e8 m, and at that
magnitude the GPU-rendered axis lines (float32 vertex buffers) and the CPU-rendered
GLTextItem labels (float32 QMatrix4x4) round differently, visibly detaching the
X/Y/Z labels from their axis tips.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyqtgraph.opengl as gl

from camera import screen_points
from geometry import arrow_points, build_surface_faces
from models import Background
from palette import (BACKGROUNDS, MEETING_MARKER, NEUTRAL, SUM_COLORS, WAVE_COLORS,
                     txt_rgba)
from physics import C, MU0
from renderer import WaveRenderObjects
from simulation import CounterpropagatingResult, FieldResult
from superposition import TIME_STEP, fronts_met

# Fixed physical→view scale for the transverse E/B deflection: 3D-view units per V/m
# (E) and per V/m-equivalent c·B. It must not depend on the amplitudes — dividing by
# the largest amplitude used to cancel out that wave's amplitude change — and it is
# independent of the displayed segment length, so `periods` only extends the axis.
FIELD_VIEW_SCALE = 0.30
POYNTING_VIEW_SCALE = 0.80 * MU0 * C   # 3D-view units per W/m² of S
DIM_ALPHA = 0.30            # individual waves while the combined field is shown

INDICATOR_SIZE = 0.5        # λ/2: k/E/B direction arrows
LABEL_CLEARANCE = (40, 16)  # px; a tick label this close on screen to another label is dropped
MEET_TRIAD_NUDGE = 0.06     # λ; each wave's direction triad at the meeting point
MEET_TRIAD_FADED = 0.35     # alpha of the meeting-point E/B directions once fields grow
MEET_FIELD_SHIFT = 0.15     # λ; offset of each wave's E/B arrows at the meeting point
MEET_FIELD_MIN = 1e-3       # λ; shorter field arrows (e.g. at a node) are not drawn

TICK_STEP = 0.25            # λ/4
AXIS_MARGIN = 2.0           # λ past the far end of a wave along the axes it runs along
MIN_LABEL_PX = 45           # thin tick labels so they stay at least this far apart
_LABEL_STEP_CANDIDATES = (1, 2, 4, 8, 16, 32, 64, 128)   # multiples of TICK_STEP
_TICKS_PER_AXIS = 29

# Precomputed λ/4 tick labels: index i → text for position i*λ/4
_TICK_LABELS = [
    "0", "λ/4", "λ/2", "3λ/4", "λ",
    "5λ/4", "3λ/2", "7λ/4", "2λ",
    "9λ/4", "5λ/2", "11λ/4", "3λ",
    "13λ/4", "7λ/2", "15λ/4", "4λ",
    "17λ/4", "9λ/2", "19λ/4", "5λ",
    "21λ/4", "11λ/2", "23λ/4", "6λ",
    "25λ/4", "13λ/2", "27λ/4", "7λ", "29λ/4",
]


@dataclass(slots=True)
class SceneFrame:
    """Everything one redraw needs, in λ units."""

    tab: int                                   # 0: wave 1, 1: wave 2, 2: superposition
    wave1: FieldResult
    wave2: FieldResult
    combined: CounterpropagatingResult | None  # superposition tab only
    source2: np.ndarray                        # wave 2's source position
    extent: float                              # shown length along k (periods or gap)
    z_max: float                               # coordinate-system size
    periods: float                             # elapsed time in carrier periods
    wavelength: float                          # metres per view unit
    # Superposition only: the fields at [source 1, meeting point, source 2]
    meeting: CounterpropagatingResult | None = None

    @property
    def meeting_point(self) -> np.ndarray:
        return self.source2 / 2

    @property
    def fronts_met(self) -> bool:
        return fronts_met(self.periods, self.extent)

    def view_positions(self, wave: FieldResult) -> np.ndarray:
        return wave.positions / self.wavelength


@dataclass(slots=True)
class DrawStyle:
    linear: bool    # curves and arrows; otherwise ribbons and tubes
    show_e: bool
    show_b: bool
    show_s: bool
    overlap_waves: bool = True   # draw each wave inside the overlap, under the sum


def _new_labels(view, count):
    labels = []
    for _ in range(count):
        label = gl.GLTextItem(pos=(0, 0, 0), text='')
        label.setVisible(False)
        view.addItem(label)
        labels.append(label)
    return labels


def _show_labels(pool, labels):
    """labels: (position, text, colour) triples; unused pool entries are hidden."""
    for label, (pos, text, color) in zip(pool, labels):
        label.setData(pos=tuple(pos), text=text, color=txt_rgba(color))
        label.setVisible(True)
    for label in pool[len(labels):]:
        label.setVisible(False)


class AxesLayer:
    """Coordinate axes, λ/4 ticks with thinned labels, and the floor grid."""

    def __init__(self, view):
        self.view = view
        self.grid = gl.GLGridItem(glOptions='translucent')
        self.grid.scale(2, 2, 1)
        self._grid_extent = None   # z_max (in λ) the grid is currently scaled to
        view.addItem(self.grid)
        self.arrows = gl.GLLinePlotItem(width=3, mode='lines', glOptions='translucent')
        self.ticks = gl.GLLinePlotItem(
            width=1, color=BACKGROUNDS[Background.BLACK]["ticks"], mode='lines',
            glOptions='translucent')
        self.labels = [gl.GLTextItem(pos=(12, 0, 0), text='X', color=txt_rgba(NEUTRAL)),
                       gl.GLTextItem(pos=(0, 12, 0), text='Y', color=txt_rgba(NEUTRAL)),
                       gl.GLTextItem(pos=(0, 0, 12), text='Z', color=txt_rgba(NEUTRAL))]
        for item in (self.arrows, self.ticks, *self.labels):
            view.addItem(item)
        # Label pool — up to _TICKS_PER_AXIS on each side of each of the 3 axes
        self.tick_labels = []
        for _ in range(6 * _TICKS_PER_AXIS):
            label = gl.GLTextItem(pos=(0, 0, 0), text='', color=txt_rgba((0.7, 0.7, 0.7, 0.9)))
            label.setVisible(False)
            view.addItem(label)
            self.tick_labels.append(label)
        self._text_color = txt_rgba(BACKGROUNDS[Background.BLACK]["text"])
        self.draw((12, 12, 12), (0, 0, 0), ("", "", ""), px_per_unit=100, grid_extent=None)

    def set_background(self, background: Background):
        cfg = BACKGROUNDS[background]
        self.view.setBackgroundColor(cfg["gl"])
        self.grid.setColor(cfg["grid"])
        self._text_color = txt_rgba(cfg["text"])
        self.ticks.setData(color=cfg["ticks"])
        for label in self.tick_labels:
            label.setData(color=self._text_color)

    def draw(self, plus, minus, notes, px_per_unit, grid_extent, avoid=()):
        """Axis lengths in λ toward + (plus) and toward − (minus), per X/Y/Z; notes
        name the field each axis carries ("E", "B" or ""). Tick labels that would
        overlap on screen the axis names, the `avoid` points (other labels) or each
        other are left out."""
        points = []
        for axis, (name, length, back, note) in enumerate(zip("XYZ", plus, minus, notes)):
            unit = np.eye(3)[axis]
            tip = unit * length
            self.labels[axis].setData(pos=tuple(tip), color=txt_rgba(NEUTRAL),
                                      text=f"{name} ({note})" if note else name)
            head, half_width = length * 0.05, length * 0.02
            # The X and Y heads open in the XY plane; Z gets a head in both planes.
            sides = [np.eye(3)[1 - axis]] if axis < 2 else [np.eye(3)[0], np.eye(3)[1]]
            points += [-unit * back, tip]
            for side in sides:
                points += [tip, tip - unit * head + side * half_width,
                           tip, tip - unit * head - side * half_width]
        points = np.array(points, dtype=float)
        self.arrows.setData(pos=points, color=np.array([NEUTRAL] * len(points)))
        names = [np.eye(3)[axis] * length for axis, length in enumerate(plus)]
        self._draw_ticks(plus, minus, px_per_unit, [*names, *avoid])
        if grid_extent is not None and grid_extent != self._grid_extent:
            self.grid.resetTransform()
            self.grid.scale(grid_extent / 10, grid_extent / 10, 1)
            self._grid_extent = grid_extent

    def _draw_ticks(self, plus, minus, px_per_unit, avoid):
        # Ticks are drawn every λ/4, but labels only every λ/4-multiple that keeps
        # them MIN_LABEL_PX apart on screen, so they don't overlap when zoomed out.
        px_per_unit = max(px_per_unit, 1e-9)
        label_step = next(
            (m for m in _LABEL_STEP_CANDIDATES if m * TICK_STEP * px_per_unit >= MIN_LABEL_PX),
            _LABEL_STEP_CANDIDATES[-1])
        points, labels = [], []
        for axis in (2, 0, 1):
            unit = np.eye(3)[axis]
            across = [np.eye(3)[other] for other in range(3) if other != axis]
            # Labels sit beside Z and Y toward −X, and beside X toward −Y.
            label_side = np.eye(3)[1 if axis == 0 else 0]
            size = max(plus[axis], minus[axis]) * 0.018
            for sign, magnitude in ((1.0, plus[axis]), (-1.0, minus[axis])):
                for i in range(1, min(int(magnitude / TICK_STEP), _TICKS_PER_AXIS) + 1):
                    p = unit * (i * TICK_STEP * sign)
                    for side in across:
                        points += [p - side * size, p + side * size]
                    if i % label_step == 0:
                        text = _TICK_LABELS[i] if i < len(_TICK_LABELS) else f"{i}λ/4"
                        labels.append((tuple(p - label_side * size * 3.5),
                                       text if sign > 0 else "−" + text))
        labels = self._clear_of(labels, avoid)
        for label, (pos, text) in zip(self.tick_labels, labels):
            label.setData(pos=pos, text=text, color=self._text_color)
            label.setVisible(True)
        for label in self.tick_labels[len(labels):]:
            label.setVisible(False)
        self.ticks.setData(pos=np.array(points, dtype=float) if points else np.zeros((2, 3)))


    def _clear_of(self, labels, avoid):
        """The labels, in order, without those that would overlap on screen a label
        already placed: one of `avoid`, or a tick label kept before it."""
        if not labels:
            return labels
        screen = screen_points(self.view, [pos for pos, _ in labels])
        taken = list(screen_points(self.view, avoid))
        kept = []
        for label, point in zip(labels, screen):
            if all(np.any(np.abs(point - other) >= LABEL_CLEARANCE) for other in taken):
                kept.append(label)
                taken.append(point)
        return kept


def axis_extents(points, minimum):
    """(plus, minus): how far each axis must reach toward + and − to cover the
    points (λ units). An axis the points run along beyond `minimum` gets AXIS_MARGIN
    past their far end; every axis reaches at least `minimum` toward +."""
    points = np.asarray(points, dtype=float)
    high = np.maximum(points.max(axis=0), 0.0)
    low = np.maximum(-points.min(axis=0), 0.0)
    plus = np.where(high > minimum, high + AXIS_MARGIN, minimum)
    minus = np.where(low > minimum, low + AXIS_MARGIN, 0.0)
    return plus, minus


def field_axis_notes(frame):
    """"E"/"B" for the axes the wave's initial E and B lie along, given its (k, E, B)
    frame; an axis no field is aligned with (e.g. a tilted vector frame) gets ""."""
    notes = ["", "", ""]
    for name, direction in zip("EB", frame[1:]):
        axis = int(np.argmax(np.abs(direction)))
        if np.isclose(abs(direction[axis]), 1.0):
            notes[axis] = name
    return tuple(notes)


class IndicatorLayer:
    """k/E/B direction triads at the sources and at the meeting point, the source-2
    marker and the meeting-point marker."""

    def __init__(self, view):
        (w1_e, w1_b, _, w1_k), (w2_e, w2_b, _, _) = WAVE_COLORS
        self.k_arrows = gl.GLLinePlotItem(width=6, color=w1_k, mode='lines', glOptions='translucent')
        self.e_arrows = gl.GLLinePlotItem(width=4, color=w1_e, mode='lines', glOptions='translucent')
        self.b_arrows = gl.GLLinePlotItem(width=4, color=w1_b, mode='lines', glOptions='translucent')
        self.text_k = gl.GLTextItem(pos=(0, 0, 0), text='k', color=txt_rgba(w1_k))
        self.text_e = gl.GLTextItem(pos=(5, 0, 0), text='E₁', color=txt_rgba(w1_e))
        self.text_b = gl.GLTextItem(pos=(0, 5, 0), text='B₁', color=txt_rgba(w1_b))
        self.text_e2 = gl.GLTextItem(pos=(5, 0, 0), text='E₂', color=txt_rgba(w2_e))
        self.text_b2 = gl.GLTextItem(pos=(0, 5, 0), text='B₂', color=txt_rgba(w2_b))
        self.source2_marker = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3)), color=NEUTRAL, size=14, pxMode=True)
        self.text_source2 = gl.GLTextItem(pos=(0, 0, 0), text='Дж.2', color=txt_rgba(NEUTRAL))
        # Midpoint between the sources, where the two fronts meet
        self.meeting_marker = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3)), color=MEETING_MARKER, size=9, pxMode=True)
        for item in (self.k_arrows, self.e_arrows, self.b_arrows, self.text_k, self.text_e,
                     self.text_b, self.text_e2, self.text_b2, self.source2_marker,
                     self.text_source2, self.meeting_marker):
            view.addItem(item)
        for item in (self.text_e, self.text_b, self.text_e2, self.text_b2,
                     self.source2_marker, self.text_source2, self.meeting_marker):
            item.setVisible(False)
        # k/E/B labels for both waves' triads at the meeting point
        self.meet_labels = _new_labels(view, 6)

    @staticmethod
    def _triad(origin, frame):
        """Arrow points for k, E and B, and the label positions just past their tips."""
        size = INDICATOR_SIZE
        arrows = [arrow_points(origin, direction * size) for direction in frame]
        tips = [origin + direction * size * 1.2 for direction in frame]
        return arrows, tips

    def draw(self, frame: SceneFrame):
        """Returns where the visible labels sit, for the axes to keep their tick labels
        clear of them."""
        tab, source2 = frame.tab, frame.source2
        k2 = frame.wave2.frame[0]
        self.meeting_marker.setData(pos=frame.meeting_point.reshape(1, 3))
        self.meeting_marker.setVisible(tab == 2)
        self.source2_marker.setData(pos=source2.reshape(1, 3))
        self.source2_marker.setVisible(tab in (1, 2))
        # Behind the source, clear of the triad's labels in front of it
        self.text_source2.setData(pos=tuple(source2 - k2 * INDICATOR_SIZE * 0.5), text='Дж.2')
        self.text_source2.setVisible(tab in (1, 2))
        self.text_e2.setVisible(False)
        self.text_b2.setVisible(False)
        if tab == 0:
            self._draw_single(np.zeros(3), frame.wave1.frame, WAVE_COLORS[0], '₁')
        elif tab == 1:
            self._draw_single(source2, frame.wave2.frame, WAVE_COLORS[1], '₂')
        else:
            self._draw_superposition(frame)
        texts = [self.text_k, self.text_e, self.text_b, self.text_e2, self.text_b2,
                 self.text_source2, *self.meet_labels]
        return [np.asarray(item.pos, dtype=float) for item in texts if item.visible()]

    def _draw_single(self, source, wave_frame, colors, index):
        """A single-wave tab: the wave's k/E/B triad at its source, so the initial
        directions show before the front has left the source."""
        self._draw_triads([(source, wave_frame, colors)])
        e_color, b_color, _, k_color = colors
        _, (kl, el, bl) = self._triad(source, wave_frame)
        self._set_text(self.text_k, kl, 'k' + index, k_color)
        self._set_text(self.text_e, el, 'E' + index, e_color)
        self._set_text(self.text_b, bl, 'B' + index, b_color)
        _show_labels(self.meet_labels, [])

    def _draw_superposition(self, frame):
        triads = [(np.zeros(3), frame.wave1.frame, WAVE_COLORS[0]),
                  (frame.source2, frame.wave2.frame, WAVE_COLORS[1])]
        labels = []
        if frame.fronts_met:
            # Repeat both waves' triads at the meeting point, each nudged toward its own
            # source so the parallel E arrows of the two waves stay distinguishable.
            nudge = MEET_TRIAD_NUDGE * frame.wave1.frame[0]
            # After the meeting moment the instantaneous field arrows are drawn there
            # too; fading these fixed E/B directions keeps the two kinds apart (they
            # can point opposite ways when the field is negative).
            faded = frame.periods > frame.extent / 2 + TIME_STEP / 2
            for origin, wave, colors, index in (
                    (frame.meeting_point - nudge, frame.wave1, WAVE_COLORS[0], '₁'),
                    (frame.meeting_point + nudge, frame.wave2, WAVE_COLORS[1], '₂')):
                e_color, b_color, s_color, k_color = colors
                if faded:
                    e_color, b_color = (c[:3] + (MEET_TRIAD_FADED,) for c in (e_color, b_color))
                triads.append((origin, wave.frame, (e_color, b_color, s_color, k_color)))
                _, (kl, el, bl) = self._triad(origin, wave.frame)
                labels += [(kl, 'k' + index, k_color), (el, 'E' + index, e_color),
                           (bl, 'B' + index, b_color)]
        _show_labels(self.meet_labels, labels)
        self._draw_triads(triads)
        (w1_e, w1_b, _, w1_k), (w2_e, w2_b, _, _) = WAVE_COLORS
        _, (kl1, el1, bl1) = self._triad(np.zeros(3), frame.wave1.frame)
        _, (_, el2, bl2) = self._triad(frame.source2, frame.wave2.frame)
        self._set_text(self.text_k, kl1, 'k₁', w1_k)
        self._set_text(self.text_e, el1, 'E₁', w1_e)
        self._set_text(self.text_b, bl1, 'B₁', w1_b)
        self._set_text(self.text_e2, el2, 'E₂', w2_e)
        self._set_text(self.text_b2, bl2, 'B₂', w2_b)

    def _draw_triads(self, triads):
        """triads: (origin, (k, E, B) frame, wave colours) for each triad to draw."""
        points, colors = [[], [], []], [[], [], []]
        for origin, frame, (e_color, b_color, _, k_color) in triads:
            arrows, _ = self._triad(origin, frame)
            for n, (pts, color) in enumerate(zip(arrows, (k_color, e_color, b_color))):
                points[n].append(pts)
                colors[n].append(np.tile(color, (len(pts), 1)))
        for item, pts, color in zip((self.k_arrows, self.e_arrows, self.b_arrows), points, colors):
            item.setData(pos=np.vstack(pts), color=np.vstack(color))
            item.setVisible(True)
        self.text_k.setVisible(True)

    @staticmethod
    def _set_text(item, pos, text, color):
        item.setData(pos=tuple(pos), color=txt_rgba(color), text=text)
        item.setVisible(True)


class FieldLayer:
    """E/B curves or ribbons and S arrows or tubes, for each wave and the combined field."""

    def __init__(self, view, steps):
        self.waves = tuple(WaveRenderObjects.create(view, e, b, s)
                           for e, b, s, _ in WAVE_COLORS)
        self.combined = WaveRenderObjects.create(view, *SUM_COLORS, curve_width=5)
        self.surface_faces = build_surface_faces(steps, closed=False)
        self.volume_faces = build_surface_faces(steps, closed=True)

    def draw(self, frame: SceneFrame, style: DrawStyle):
        tab, wave1, wave2 = frame.tab, frame.wave1, frame.wave2
        # In the superposition tab the combined field is drawn where both fronts have
        # arrived; each wave keeps its own S arrows only where it is still alone.
        if tab == 2:
            overlap = wave1.active & wave2.active
            alone1, alone2 = wave1.active & ~overlap, wave2.active & ~overlap
            # Dim the waves only under a combined field. Without them in the overlap,
            # what remains of each wave is alone and keeps full colour.
            dim = DIM_ALPHA if overlap.any() and style.overlap_waves else 1.0
        else:
            overlap = np.zeros(len(wave1.active), dtype=bool)
            alone1, alone2 = wave1.active, wave2.active
            dim = 1.0
        show_s = style.show_s
        for objects, wave, colors, visible, alone in (
                (self.waves[0], wave1, WAVE_COLORS[0], tab in (0, 2), alone1),
                (self.waves[1], wave2, WAVE_COLORS[1], tab in (1, 2), alone2)):
            field_mask = wave.active if style.overlap_waves else alone
            self._draw_layer(objects, colors[:3], wave_vectors(wave), frame.view_positions(wave),
                             wave.frame, visible, show_s, field_mask, alone, dim,
                             style, frame.z_max)
        if tab == 2:
            # Combined S lies along the source-to-source line, k₁ = −k₂.
            combined = frame.combined
            vectors = field_vectors(combined.electric, combined.magnetic, combined.poynting,
                                    wave1.frame[0])
            self._draw_layer(self.combined, SUM_COLORS, vectors, frame.view_positions(wave1),
                             wave1.frame, True, show_s, overlap, overlap, 1.0,
                             style, frame.z_max)
        else:
            self.combined.hide()

    def _draw_layer(self, objects, colors, vectors, pos, frame, visible, show_flux,
                    field_mask, s_mask, alpha, style, z_max):
        """Draw one wave (or the combined field) over the samples in its masks.

        field_mask selects where E/B are drawn and s_mask where S is drawn; both are
        contiguous runs, so the precomputed strip faces can be sliced to fit.
        """
        linear, show_e, show_b = style.linear, style.show_e, style.show_b
        e_v, b_v, sv = vectors
        n_field, n_s = int(field_mask.sum()), int(s_mask.sum())
        fields_on = visible and n_field >= 2
        flux_on = visible and show_flux and n_s >= 2
        objects.electric_curve.setVisible(linear and show_e and fields_on)
        objects.magnetic_curve.setVisible(linear and show_b and fields_on)
        objects.poynting_vectors.setVisible(linear and flux_on)
        objects.electric_surface.setVisible(not linear and show_e and fields_on)
        objects.magnetic_surface.setVisible(not linear and show_b and fields_on)
        objects.poynting_surface.setVisible(not linear and flux_on)
        e_color, b_color, s_color = (c[:3] + (c[3] * alpha,) for c in colors)
        if fields_on:
            p_f, e_f, b_f = pos[field_mask], e_v[field_mask], b_v[field_mask]
            if linear:
                objects.electric_curve.setData(pos=p_f + e_f, color=e_color)
                objects.magnetic_curve.setData(pos=p_f + b_f, color=b_color)
            else:
                faces = self.surface_faces[:2 * (n_field - 1)]
                objects.electric_surface.setMeshData(vertexes=_stalks(p_f, e_f), faces=faces)
                objects.magnetic_surface.setMeshData(vertexes=_stalks(p_f, b_f), faces=faces)
                objects.electric_surface.setColor(e_color[:3] + (0.5 * alpha,))
                objects.magnetic_surface.setColor(b_color[:3] + (0.5 * alpha,))
        if flux_on:
            p_s, s_s = pos[s_mask], sv[s_mask]
            if linear:
                objects.poynting_vectors.setData(
                    pos=_poynting_arrows(s_s, p_s, frame, z_max), color=s_color)
            else:
                objects.poynting_surface.setMeshData(
                    vertexes=_poynting_tube(s_s, p_s, frame),
                    faces=self.volume_faces[:8 * (n_s - 1)])
                objects.poynting_surface.setColor(s_color[:3] + (0.5 * alpha,))


def field_vectors(electric, magnetic, poynting, k_hat):
    """E and c·B in view units, and S along k_hat in view units."""
    return (electric * FIELD_VIEW_SCALE, magnetic * C * FIELD_VIEW_SCALE,
            (poynting @ k_hat) * POYNTING_VIEW_SCALE)


def wave_vectors(wave: FieldResult):
    return field_vectors(wave.electric, wave.magnetic, wave.poynting, wave.frame[0])


def _stalks(base, vectors):
    """Interleave base points with their tips: one line segment per sample."""
    points = np.empty((len(base) * 2, 3))
    points[0::2] = base
    points[1::2] = base + vectors
    return points


def _poynting_arrows(s_val, pos, frame, z_max):
    k_hat, t1, t2 = frame
    lines = _stalks(pos, np.outer(s_val, k_hat))
    head_size, head_width = z_max * 0.03, z_max * 0.02 * 0.5
    every = np.arange(0, len(pos), 10)
    s_heads, base = s_val[every], pos[every]
    mask = np.abs(s_heads) >= (z_max * 0.001)
    if np.any(mask):
        s_heads, base = s_heads[mask], base[mask]
        tip = base + np.outer(s_heads, k_hat)
        head = tip - np.outer(np.sign(s_heads) * head_size, k_hat)
        heads = np.empty((len(tip) * 8, 3))
        for n, side in enumerate((t1, -t1, t2, -t2)):
            heads[2 * n::8] = tip
            heads[2 * n + 1::8] = head + head_width * side
        lines = np.vstack([lines, heads])
    return lines


def _poynting_tube(s_val, pos, frame, n_seg=4):
    _, t1, t2 = frame
    radius = np.abs(s_val) * 0.5   # a negative radius would turn the tube inside out
    verts = np.empty((len(pos) * n_seg, 3))
    for i in range(n_seg):
        angle = i * (2 * np.pi / n_seg)
        verts[i::n_seg] = (pos + np.outer(radius * np.cos(angle), t1)
                           + np.outer(radius * np.sin(angle), t2))
    return verts


class MeetingFieldArrows:
    """Instantaneous E/B of both waves and their sum at the meeting point."""

    def __init__(self, view):
        self.wave_arrows = gl.GLLinePlotItem(width=3, mode='lines', glOptions='translucent')
        self.sum_arrows = gl.GLLinePlotItem(width=5, mode='lines', glOptions='translucent')
        for item in (self.wave_arrows, self.sum_arrows):
            item.setVisible(False)
            view.addItem(item)
        self.labels = _new_labels(view, 6)

    def draw(self, frame: SceneFrame, style: DrawStyle):
        """The vectors are the fields exactly at the meeting point, at the same scale as
        the field curves, so the Σ arrows end on the combined curves. Each
        wave's own arrows are shifted toward its source (wave 1 at −k₁, wave 2 at +k₁),
        otherwise the equal E₁ and E₂ there would be drawn on top of each other.
        """
        groups = ((self.wave_arrows, ()), (self.sum_arrows, ()))
        meeting = frame.meeting
        idx = 1   # the meeting point in frame.meeting
        if (frame.tab == 2 and meeting is not None
                and meeting.wave1.active[idx] and meeting.wave2.active[idx]):
            wave1, wave2 = meeting.wave1, meeting.wave2
            base = frame.meeting_point
            e1, b1, _ = (v[idx] for v in wave_vectors(wave1))
            e2, b2, _ = (v[idx] for v in wave_vectors(wave2))
            combined = meeting
            se, sb, _ = (v[idx] for v in field_vectors(
                combined.electric, combined.magnetic, combined.poynting, wave1.frame[0]))
            shift = MEET_FIELD_SHIFT * wave1.frame[0]
            (w1_e, w1_b, _, _), (w2_e, w2_b, _, _) = WAVE_COLORS
            sum_e, sum_b, _ = SUM_COLORS
            show_e, show_b = style.show_e, style.show_b
            groups = (
                (self.wave_arrows, (
                    (base - shift, e1, w1_e, 'E₁', show_e), (base - shift, b1, w1_b, 'B₁', show_b),
                    (base + shift, e2, w2_e, 'E₂', show_e), (base + shift, b2, w2_b, 'B₂', show_b))),
                (self.sum_arrows, (
                    (base, se, sum_e, 'EΣ', show_e), (base, sb, sum_b, 'BΣ', show_b))),
            )
        labels = []
        for item, arrows in groups:
            points, colors = [], []
            for origin, vector, color, text, shown in arrows:
                length = np.linalg.norm(vector)
                if not shown or length < MEET_FIELD_MIN:
                    continue
                pts = arrow_points(origin, vector, head_fraction=0.3, max_head=0.06,
                                   width_ratio=0.4)
                points.append(pts)
                colors.append(np.tile(color, (len(pts), 1)))
                labels.append((origin + vector * (1 + 0.08 / length), text, color))
            item.setVisible(bool(points))
            if points:
                item.setData(pos=np.vstack(points), color=np.vstack(colors))
        _show_labels(self.labels, labels)
