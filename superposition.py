"""Timing and source placement for the superposition tab (no Qt dependencies)."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from models import WaveParameters

# The carrier is fixed at 1 Hz: one period T is 1 s, and a front advances 1 λ per T.
CARRIER_FREQUENCY = 1.0
TIME_STEP = 0.05                 # T/20 — one animation tick and one manual step


def preset_periods(preset: str, gap: float) -> float:
    """Elapsed periods for a characteristic moment; gap is the source distance in λ.

    "start": fronts have covered a quarter of the gap; "meet": they meet midway;
    "later": full overlap, offset T/8 from the instant where in-phase sources give
    E Σ = 0 everywhere, so both E Σ and B Σ are visible.
    """
    return {"start": gap / 4, "meet": gap / 2, "later": gap + 0.625}[preset]


def fronts_met(periods: float, gap: float) -> bool:
    return periods >= gap / 2 - 1e-9


def place_source2(wave2: WaveParameters, separation: float) -> WaveParameters:
    """Wave 2 as used by the superposition tab: its source sits at the configured
    separation, in the direction of wave 2's own source position (+Z if unset)."""
    direction = np.asarray(wave2.source_position, dtype=float)
    norm = np.linalg.norm(direction)
    unit = direction / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])
    position = unit * separation
    return replace(wave2, source_position=tuple(float(v) for v in position))


def time_text(gap: float, periods: float) -> str:
    """Readout for the superposition tab; gap in λ, elapsed time in periods T."""
    front = min(max(periods, 0.0), gap)   # each front advances 1 λ per T
    meet = gap / 2
    if periods <= 0:
        stage = "джерела ще не випромінюють"
    elif abs(periods - meet) < TIME_STEP / 2:
        stage = "фронти зустрічаються"
    elif periods < meet:
        stage = f"ще {meet - periods:.2f} T"
    elif periods < gap:
        stage = f"перекриття {2 * (periods - meet):.2f}λ"
    else:
        stage = "повне перекриття"
    return (f"t = {periods:.2f} T;  фронти: {front:.2f}λ з {gap:.2f}λ\n"
            f"зустріч при {meet:.2f} T — {stage}")
