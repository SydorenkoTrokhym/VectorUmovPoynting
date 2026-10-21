import unittest

import numpy as np

from geometry import field_directions, wave_frame
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


if __name__ == "__main__":
    unittest.main()
