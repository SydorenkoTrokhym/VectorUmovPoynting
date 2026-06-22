import numpy as np
from numpy.typing import NDArray

C = 3e8                  # speed of light (m/s)
MU0 = 4 * np.pi * 1e-7  # vacuum permeability (H/m)


def calculate_fields(
    z: NDArray[np.float64],
    t: float,
    frequency: float = 1.0,
    amplitude: float = 1.0,
    polarization: str = 'linear',
    phase_shift: float = 0,
    modulation_type: str = 'none',
    mod_freq: float = 0.1,
    mod_index: float = 0.5,
    z_offset: float = 0.0,
    direction: float = 1.0, # 1.0 for +z, -1.0 for -z
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Розрахунок полів E та B для пласкої хвилі з підтримкою модуляції, зміщення по Z та напрямку.
    """
    omega_c = 2 * np.pi * frequency
    omega_m = 2 * np.pi * mod_freq
    k_c = direction * (omega_c / C)
    phi = np.radians(phase_shift)

    # Adjusted z for the wave source position
    z_adj = z - z_offset
    
    if modulation_type == 'AM':
        amp_t = amplitude * (1 + mod_index * np.cos(omega_m * t))
        phase = omega_c * t - k_c * z_adj + phi
    elif modulation_type == 'FM':
        amp_t = amplitude
        phase = omega_c * t - k_c * z_adj + mod_index * np.sin(omega_m * t) + phi
    elif modulation_type == 'AFM':
        amp_t = amplitude * (1 + mod_index * np.cos(omega_m * t))
        phase = omega_c * t - k_c * z_adj + mod_index * np.sin(omega_m * t) + phi
    else:
        amp_t = amplitude
        phase = omega_c * t - k_c * z_adj + phi

    if polarization == 'circular_right':
        E0x = -amp_t * np.sin(phase)
        E0y = -amp_t * np.cos(phase)
    elif polarization == 'circular_left':
        E0x = -amp_t * np.sin(phase)
        E0y = amp_t * np.cos(phase)
    elif polarization == 'elliptical':
        E0x = -amp_t * np.sin(phase)
        E0y = -0.5 * amp_t * np.cos(phase + np.pi / 4)
    else:  # linear
        E0x = -amp_t * np.sin(phase)
        E0y = np.zeros_like(E0x)

    Ex = E0x
    Ey = E0y
    # B = (k x E) / omega
    # If direction is -1, k is -k_c*z_unit, so B = (-z_unit x E) / C
    Bx = -direction * Ey / C
    By = direction * Ex / C

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

