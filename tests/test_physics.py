import unittest

import numpy as np

from physics import C, MU0, calculate_fields, calculate_poynting_3d


class FieldTests(unittest.TestCase):
    def test_plane_wave_fields_are_orthogonal(self):
        z = np.linspace(0.0, C, 101)
        ex, ey, bx, by = calculate_fields(z, 0.17, polarization="circular_right")
        electric = np.column_stack((ex, ey, np.zeros_like(ex)))
        magnetic = np.column_stack((bx, by, np.zeros_like(bx)))

        np.testing.assert_allclose(np.sum(electric * magnetic, axis=1), 0.0, atol=1e-15)
        np.testing.assert_allclose(
            np.linalg.norm(magnetic, axis=1),
            np.linalg.norm(electric, axis=1) / C,
        )

    def test_poynting_direction_follows_wave_direction(self):
        z = np.linspace(0.0, C, 101)
        for direction in (1.0, -1.0):
            ex, ey, bx, by = calculate_fields(z, 0.11, direction=direction)
            electric = np.column_stack((ex, ey, np.zeros_like(ex)))
            magnetic = np.column_stack((bx, by, np.zeros_like(bx)))
            poynting = calculate_poynting_3d(electric, magnetic)
            self.assertGreater(direction * np.mean(poynting[:, 2]), 0.0)

    def test_superposition_contains_interference_terms(self):
        electric = np.array([[1.0, 0.0, 0.0]])
        magnetic = np.array([[0.0, 1.0 / C, 0.0]])
        individual = calculate_poynting_3d(electric, magnetic)
        total = calculate_poynting_3d(2.0 * electric, 2.0 * magnetic)

        np.testing.assert_allclose(total, 4.0 * individual)
        self.assertFalse(np.allclose(total, 2.0 * individual, rtol=1e-12, atol=0.0))
        np.testing.assert_allclose(individual[0, 2], 1.0 / (MU0 * C))

    def test_modulation_envelope_propagates(self):
        # The same retarded-time point one quarter wavelength later has the same field.
        frequency = 2.0
        wavelength = C / frequency
        dt = 0.25 / frequency
        z = np.array([0.0])
        shifted_z = np.array([0.25 * wavelength])
        field_a = calculate_fields(z, 0.0, frequency=frequency, modulation_type="AM")
        field_b = calculate_fields(shifted_z, dt, frequency=frequency, modulation_type="AM")
        for a, b in zip(field_a, field_b):
            np.testing.assert_allclose(a, b)

    def test_poynting_3d_rejects_invalid_shapes(self):
        with self.assertRaises(ValueError):
            calculate_poynting_3d(np.zeros((2, 2)), np.zeros((2, 2)))


if __name__ == "__main__":
    unittest.main()
