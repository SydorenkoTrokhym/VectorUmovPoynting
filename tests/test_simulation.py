import unittest

import numpy as np

from models import WaveParameters
from physics import C
from simulation import simulate_counterpropagating, simulate_wave, superpose


class SimulationTests(unittest.TestCase):
    def assert_vector_relation(self, field):
        k_hat = field.frame[0]
        k = np.broadcast_to(k_hat, field.electric.shape)
        np.testing.assert_allclose(np.sum(field.electric * field.magnetic, axis=1), 0.0, atol=1e-14)
        np.testing.assert_allclose(np.sum(field.electric * k, axis=1), 0.0, atol=1e-14)
        np.testing.assert_allclose(np.sum(field.magnetic * k, axis=1), 0.0, atol=1e-14)
        np.testing.assert_allclose(field.magnetic, np.cross(k, field.electric) / C, atol=1e-14)
        self.assertGreater(np.mean(field.poynting @ k_hat), 0.0)

    def test_cartesian_directions_obey_maxwell_relations(self):
        for direction in ("+X", "-X", "+Y", "-Y", "+Z", "-Z"):
            with self.subTest(direction=direction):
                magnetic_direction = "-Y" if direction == "-Z" else "+Y"
                self.assert_vector_relation(simulate_wave(
                    WaveParameters(propagation_direction=direction,
                                   magnetic_direction=magnetic_direction), 0.17))

    def test_arbitrary_direction_obeys_maxwell_relations(self):
        field = simulate_wave(WaveParameters(
            direction_mode="Довільний вектор",
            direction_vector=(1.0, -2.0, 3.0),
            polarization="elliptical",
        ), 0.31)
        self.assert_vector_relation(field)

    def test_circular_polarization_has_constant_magnitude(self):
        amplitude = 3.25
        field = simulate_wave(WaveParameters(
            amplitude=amplitude, polarization="circular_right"), 0.23)
        np.testing.assert_allclose(np.linalg.norm(field.electric, axis=1), amplitude)

    def test_in_phase_interference_is_quadratic(self):
        field = simulate_wave(WaveParameters(), 0.13)
        _, _, total_s = superpose(field, field)
        np.testing.assert_allclose(total_s, 4.0 * field.poynting)

    def test_opposite_phase_fields_cancel(self):
        first = simulate_wave(WaveParameters(phase_degrees=0), 0.13)
        second = simulate_wave(WaveParameters(phase_degrees=180), 0.13)
        electric, magnetic_h, poynting = superpose(first, second)
        np.testing.assert_allclose(electric, 0.0, atol=1e-14)
        np.testing.assert_allclose(magnetic_h, 0.0, atol=1e-14)
        np.testing.assert_allclose(poynting, 0.0, atol=1e-14)

    def test_counterpropagating_sources_are_separated_in_wavelengths(self):
        source = (1.0, -2.0, 3.0)
        result = simulate_counterpropagating(
            WaveParameters(propagation_direction="-X"),
            WaveParameters(propagation_direction="+Y", source_position=source),
            0.0, frequency=2.0,
        )
        wavelength = C / 2.0
        np.testing.assert_allclose(result.positions[0], [0, 0, 0])
        np.testing.assert_allclose(result.positions[-1], np.asarray(source) * wavelength)
        expected_direction = np.asarray(source) / np.linalg.norm(source)
        np.testing.assert_allclose(result.wave1.frame[0], expected_direction)
        np.testing.assert_allclose(result.wave2.frame[0], -expected_direction)

    def test_counterpropagating_poynting_vectors_face_each_other(self):
        result = simulate_counterpropagating(
            WaveParameters(), WaveParameters(source_position=(1, 2, 3)), 0.17)
        direction = result.wave1.frame[0]
        self.assertGreater(np.mean(result.wave1.poynting @ direction), 0.0)
        self.assertLess(np.mean(result.wave2.poynting @ direction), 0.0)

    def test_steady_state_waves_fill_the_gap(self):
        result = simulate_counterpropagating(
            WaveParameters(), WaveParameters(source_position=(0, 0, 2)), 0.3)
        self.assertTrue(result.wave1.active.all())
        self.assertTrue(result.wave2.active.all())

    def test_switched_on_waves_leave_their_own_sources(self):
        # Sources 2 lambda apart, 1 Hz: each front has travelled 0.5 lambda at t = 0.5 s.
        result = simulate_counterpropagating(
            WaveParameters(), WaveParameters(source_position=(0, 0, 2)),
            0.5, frequency=1.0, switch_on_time=0.0)
        along = result.positions[:, 2] / C
        np.testing.assert_array_equal(result.wave1.active, along <= 0.5 + 1e-9)
        np.testing.assert_array_equal(result.wave2.active, along >= 1.5 - 1e-9)
        self.assertFalse((result.wave1.active & result.wave2.active).any())
        for wave in (result.wave1, result.wave2):
            np.testing.assert_array_equal(wave.electric[~wave.active], 0.0)
            np.testing.assert_array_equal(wave.poynting[~wave.active], 0.0)
        # Each wave's energy flows away from its own source.
        self.assertGreater(np.sum(result.wave1.poynting[:, 2]), 0.0)
        self.assertLess(np.sum(result.wave2.poynting[:, 2]), 0.0)

    def test_switched_on_fronts_meet_midway_then_overlap_everywhere(self):
        params = (WaveParameters(), WaveParameters(source_position=(0, 0, 2)))
        before = simulate_counterpropagating(*params, 0.95, switch_on_time=0.0)
        after = simulate_counterpropagating(*params, 1.05, switch_on_time=0.0)
        later = simulate_counterpropagating(*params, 2.0, switch_on_time=0.0)
        self.assertFalse((before.wave1.active & before.wave2.active).any())
        overlap = after.positions[after.wave1.active & after.wave2.active, 2] / C
        self.assertTrue(np.all(np.abs(overlap - 1.0) <= 0.05 + 1e-9))
        self.assertTrue((later.wave1.active & later.wave2.active).all())

    def test_counterpropagating_rejects_coincident_sources(self):
        with self.assertRaises(ValueError):
            simulate_counterpropagating(
                WaveParameters(), WaveParameters(source_position=(0, 0, 0)), 0.0)


if __name__ == "__main__":
    unittest.main()
