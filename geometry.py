from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from models import WaveParameters


BASE_FRAMES = {
    "+Z": (np.array([0., 0., 1.]), np.array([1., 0., 0.]), np.array([0., 1., 0.])),
    "-Z": (np.array([0., 0., -1.]), np.array([1., 0., 0.]), np.array([0., -1., 0.])),
    "+X": (np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.])),
    "-X": (np.array([-1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., -1.])),
    "+Y": (np.array([0., 1., 0.]), np.array([1., 0., 0.]), np.array([0., 0., -1.])),
    "-Y": (np.array([0., -1., 0.]), np.array([1., 0., 0.]), np.array([0., 0., 1.])),
}

FIELD_DIRECTIONS = {
    "+X": np.array([1., 0., 0.]), "-X": np.array([-1., 0., 0.]),
    "+Y": np.array([0., 1., 0.]), "-Y": np.array([0., -1., 0.]),
}


def field_directions(electric: str, magnetic: str) -> tuple[NDArray, NDArray, NDArray]:
    """Return (k, E, B), validating E perpendicular B and k = E cross B."""
    try:
        e_hat = FIELD_DIRECTIONS[electric]
        b_hat = FIELD_DIRECTIONS[magnetic]
    except KeyError as exc:
        raise ValueError(f"Unknown field direction: {exc.args[0]}") from exc
    if not np.isclose(np.dot(e_hat, b_hat), 0.0):
        raise ValueError("Initial E and B directions must be perpendicular")
    return np.cross(e_hat, b_hat), e_hat.copy(), b_hat.copy()


def wave_frame(params: WaveParameters) -> tuple[NDArray, NDArray, NDArray]:
    if params.direction_mode == "Довільний вектор":
        vector = np.asarray(params.direction_vector, dtype=float)
        norm = np.linalg.norm(vector)
        if norm <= 1e-9:
            raise ValueError("The propagation direction vector cannot be zero")
        k_hat = vector / norm
        reference = np.array([1., 0., 0.]) if abs(k_hat[0]) < 0.9 else np.array([0., 1., 0.])
        e1_hat = np.cross(reference, k_hat)
        e1_hat /= np.linalg.norm(e1_hat)
        return k_hat, e1_hat, np.cross(k_hat, e1_hat)
    if params.propagation_direction in ("+Z", "-Z"):
        k_hat, e_hat, b_hat = field_directions(
            params.electric_direction, params.magnetic_direction)
        expected = BASE_FRAMES[params.propagation_direction][0]
        if not np.allclose(k_hat, expected):
            raise ValueError("E × B does not match the selected propagation direction")
        return k_hat, e_hat, b_hat
    return BASE_FRAMES[params.propagation_direction]


def transform_points(local_points: NDArray, frame: tuple[NDArray, NDArray, NDArray], origin: NDArray) -> NDArray:
    matrix = np.column_stack(frame)
    return np.asarray(local_points) @ matrix.T + np.asarray(origin)


def build_surface_faces(steps: int, closed: bool = False) -> NDArray[np.int_]:
    faces: list[list[int]] = []
    width = 4 if closed else 2
    for i in range(steps - 1):
        if closed:
            for j in range(4):
                a, b = i * 4 + j, i * 4 + (j + 1) % 4
                c, d = (i + 1) * 4 + j, (i + 1) * 4 + (j + 1) % 4
                faces.extend(([a, b, c], [b, d, c]))
        else:
            a = i * width
            faces.extend(([a, a + 1, a + 2], [a + 1, a + 3, a + 2]))
    return np.asarray(faces, dtype=int)


def arrow_points(origin: NDArray, direction: NDArray, head_fraction: float = 0.25) -> NDArray:
    origin, direction = np.asarray(origin, float), np.asarray(direction, float)
    length = np.linalg.norm(direction)
    if length <= 1e-12:
        raise ValueError("Arrow direction cannot be zero")
    unit, tip = direction / length, origin + direction
    reference = np.array([0., 0., 1.]) if abs(unit[2]) < 0.9 else np.array([0., 1., 0.])
    side1 = np.cross(unit, reference); side1 /= np.linalg.norm(side1)
    side2 = np.cross(unit, side1)
    base = tip - unit * length * head_fraction
    width = length * head_fraction * 0.32
    return np.array([origin, tip, tip, base + side1 * width, tip, base - side1 * width,
                     tip, base + side2 * width, tip, base - side2 * width])
