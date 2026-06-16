import numpy as np
from numpy.typing import NDArray

C = 3e8                  # speed of light (m/s)
MU0 = 4 * np.pi * 1e-7  # vacuum permeability (H/m)


def calculate_fields(
    z: NDArray[np.float64],
    t: float,
    frequency: float = 1e9,
    amplitude: float = 1.0,
    polarization: str = 'linear',
    angle: float = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Розрахунок полів E та B для пласкої хвилі.
    angle: кут повороту вектора поля навколо осі Z у градусах.
    """
    omega = 2 * np.pi * frequency
    k = omega / C
    alpha = np.radians(angle)
    arg = omega * t - k * z

    if polarization == 'circular_right':
        E0x = amplitude * np.cos(arg)
        E0y = amplitude * np.sin(arg)
    elif polarization == 'circular_left':
        E0x = amplitude * np.cos(arg)
        E0y = -amplitude * np.sin(arg)
    elif polarization == 'elliptical':
        E0x = amplitude * np.cos(arg)
        E0y = 0.5 * amplitude * np.cos(arg + np.pi / 4)
    else:  # linear
        E0x = amplitude * np.cos(arg)
        E0y = np.zeros_like(E0x)

    # Rotate field vector by alpha around Z-axis
    Ex = E0x * np.cos(alpha) - E0y * np.sin(alpha)
    Ey = E0x * np.sin(alpha) + E0y * np.cos(alpha)

    # B = (1/c) * (ẑ × E)
    Bx = -Ey / C
    By = Ex / C

    return Ex, Ey, Bx, By


def calculate_h_field(
    Bx: NDArray[np.float64],
    By: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Розрахунок напруженості магнітного поля H = B / μ₀"""
    Hx = Bx / MU0
    Hy = By / MU0
    return Hx, Hy


def calculate_poynting(
    Ex: NDArray[np.float64],
    Ey: NDArray[np.float64],
    Bx: NDArray[np.float64],
    By: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Розрахунок вектора Пойнтінга S = (1/μ₀) × (E × B)"""
    Sx = np.zeros_like(Ex)
    Sy = np.zeros_like(Ex)
    Sz = (Ex * By - Ey * Bx) / MU0
    return Sx, Sy, Sz