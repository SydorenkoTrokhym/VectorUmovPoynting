import unittest

import numpy as np

from geometry import arrow_points, field_directions, wave_frame
from models import WaveParameters


class FieldDirectionTests(unittest.TestCase):
    def test_all_perpendicular_pairs_follow_right_hand_rule(self):
        cases = (
            ("+X", "+Y", "+Z"), ("+X", "-Y", "-Z"),
            ("-X", "+Y", "-Z"), ("-X", "-Y", "+Z"),
            ("+Y", "+X", "-Z"), ("+Y", "-X", "+Z"),
            ("-Y", "+X", "+Z"), ("-Y", "-X", "-Z"),
        )
        for electric, magnetic, expected in cases:
            with self.subTest(electric=electric, magnetic=magnetic):
                k_hat, e_hat, b_hat = field_directions(electric, magnetic)
                np.testing.assert_allclose(np.cross(e_hat, b_hat), k_hat)
                self.assertEqual("+Z" if k_hat[2] > 0 else "-Z", expected)

    def test_parallel_field_directions_are_rejected(self):
        with self.assertRaises(ValueError):
            field_directions("+X", "-X")

    def test_flow_must_match_configured_propagation(self):
        with self.assertRaises(ValueError):
            wave_frame(WaveParameters(
                propagation_direction="-Z",
                electric_direction="+X",
                magnetic_direction="+Y",
            ))


class ArrowPointsTests(unittest.TestCase):
    def test_head_is_a_fraction_of_the_length(self):
        points = arrow_points([0, 0, 0], [0, 0, 2])
        np.testing.assert_allclose(points[1], [0, 0, 2])
        np.testing.assert_allclose(points[3][2], 2 - 0.5)        # head = 25 % of 2
        self.assertAlmostEqual(np.linalg.norm(points[3][:2]), 0.5 * 0.32)

    def test_head_can_be_capped_for_long_arrows(self):
        points = arrow_points([1, 0, 0], [0, 3, 0], head_fraction=0.3, max_head=0.06,
                              width_ratio=0.4)
        np.testing.assert_allclose(points[3][1], 3 - 0.06)
        self.assertAlmostEqual(abs(points[3][0] - 1) + abs(points[3][2]), 0.06 * 0.4)

    def test_zero_arrow_is_rejected(self):
        with self.assertRaises(ValueError):
            arrow_points([0, 0, 0], [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
