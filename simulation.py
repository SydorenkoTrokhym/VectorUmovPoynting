from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from geometry import wave_frame
from models import WaveParameters
from physics import C, MU0, calculate_fields, calculate_poynting_3d


@dataclass(slots=True)
class FieldResult:
    positions: NDArray[np.float64]
    electric: NDArray[np.float64]
    magnetic: NDArray[np.float64]
    magnetic_h: NDArray[np.float64]
    poynting: NDArray[np.float64]
    frame: tuple[NDArray, NDArray, NDArray]
    # True where the wave has reached the sample; fields are zero elsewhere.
    active: NDArray[np.bool_]


@dataclass(slots=True)
class CounterpropagatingResult:
    positions: NDArray[np.float64]
    wave1: FieldResult
    wave2: FieldResult
    electric: NDArray[np.float64]
    magnetic: NDArray[np.float64]
    magnetic_h: NDArray[np.float64]
    poynting: NDArray[np.float64]


def simulate_wave(params: WaveParameters, time: float, frequency: float = 1.0, steps: int = 200) -> FieldResult:
    wavelength = C / frequency
    distance = np.linspace(0.0, params.periods * wavelength, steps)
    ex, ey, bx, by = calculate_fields(
        distance, time, frequency, params.amplitude, params.polarization,
        params.phase_degrees, params.modulation_type,
        params.modulation_frequency, params.modulation_index,
    )
    frame = wave_frame(params)
    k_hat, e1_hat, e2_hat = frame
    origin = np.asarray(params.source_position) * wavelength
    electric = np.outer(ex, e1_hat) + np.outer(ey, e2_hat)
    magnetic = np.outer(bx, e1_hat) + np.outer(by, e2_hat)
    positions = origin + np.outer(distance, k_hat)
    return FieldResult(positions, electric, magnetic, magnetic / MU0,
                       calculate_poynting_3d(electric, magnetic), frame,
                       np.ones(steps, dtype=bool))


def superpose(*fields: FieldResult) -> tuple[NDArray, NDArray, NDArray]:
    if not fields:
        raise ValueError("At least one field is required")
    electric = np.sum([field.electric for field in fields], axis=0)
    magnetic = np.sum([field.magnetic for field in fields], axis=0)
    return electric, magnetic / MU0, calculate_poynting_3d(electric, magnetic)


def simulate_counterpropagating(
    wave1: WaveParameters,
    wave2: WaveParameters,
    time: float,
    frequency: float = 1.0,
    steps: int = 200,
    switch_on_time: float | None = None,
) -> CounterpropagatingResult:
    """Evaluate two waves travelling toward each other between their sources.

    Source 1 is at the origin. Source 2 is positioned by ``wave2.source_position``
    in wavelength units. Directions are derived from the line connecting the sources;
    configured propagation directions are deliberately ignored in this scenario.

    With ``switch_on_time`` set, both sources start emitting at that moment: each
    wave is zero beyond its front, which leaves its own source at speed c. Without
    it, both waves fill the whole gap (steady state).
    """
    wavelength = C / frequency
    source2 = np.asarray(wave2.source_position, dtype=float) * wavelength
    distance = np.linalg.norm(source2)
    if distance <= 1e-12:
        raise ValueError("The second source must differ from the origin")
    k1 = source2 / distance
    reference = np.array([1., 0., 0.])
    if abs(np.dot(reference, k1)) > 0.9:
        reference = np.array([0., 1., 0.])
    e1 = reference - np.dot(reference, k1) * k1
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(k1, e1)
    travelled1 = np.linspace(0.0, distance, steps)
    positions = np.outer(travelled1, k1)
    distances = (travelled1, distance - travelled1)
    frames = (
        (k1, e1, e2),
        (-k1, e1, -e2),
    )
    results = []
    for params, travelled, frame in zip((wave1, wave2), distances, frames):
        ex, ey, bx, by = calculate_fields(
            travelled, time, frequency, params.amplitude, params.polarization,
            params.phase_degrees, params.modulation_type,
            params.modulation_frequency, params.modulation_index,
        )
        if switch_on_time is None:
            active = np.ones(steps, dtype=bool)
        else:
            # Small tolerance so a front landing exactly on a sample counts as arrived.
            front = C * (time - switch_on_time) + 1e-9 * wavelength
            active = travelled <= front
        k_hat, e1_hat, e2_hat = frame
        electric = np.outer(ex * active, e1_hat) + np.outer(ey * active, e2_hat)
        magnetic = np.outer(bx * active, e1_hat) + np.outer(by * active, e2_hat)
        results.append(FieldResult(positions, electric, magnetic, magnetic / MU0,
                                   calculate_poynting_3d(electric, magnetic), frame, active))
    electric = results[0].electric + results[1].electric
    magnetic = results[0].magnetic + results[1].magnetic
    return CounterpropagatingResult(
        results[0].positions, results[0], results[1], electric, magnetic,
        magnetic / MU0, calculate_poynting_3d(electric, magnetic),
    )
