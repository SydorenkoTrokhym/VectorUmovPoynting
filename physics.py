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
    phi = np.radians(phase_shift)

    # Retarded time makes both the carrier and its modulation propagate.
    z_adj = z - z_offset
    tau = t - direction * z_adj / C
    
    if modulation_type == 'AM':
        amp_t = amplitude * (1 + mod_index * np.cos(omega_m * tau))
        phase = omega_c * tau + phi
    elif modulation_type == 'FM':
        amp_t = amplitude
        phase = omega_c * tau + mod_index * np.sin(omega_m * tau) + phi
    elif modulation_type == 'AFM':
        amp_t = amplitude * (1 + mod_index * np.cos(omega_m * tau))
        phase = omega_c * tau + mod_index * np.sin(omega_m * tau) + phi
    else:
        amp_t = amplitude
        phase = omega_c * tau + phi

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


def calculate_poynting_3d(
    electric: NDArray[np.float64],
    magnetic: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return S = E x B / mu0 for arrays whose final dimension is XYZ."""
    electric = np.asarray(electric, dtype=float)
    magnetic = np.asarray(magnetic, dtype=float)
    if electric.shape != magnetic.shape or electric.shape[-1] != 3:
        raise ValueError("electric and magnetic must have equal (..., 3) shapes")
    return np.cross(electric, magnetic) / MU0
