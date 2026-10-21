import unittest

from models import WaveParameters
from superposition import TIME_STEP, fronts_met, place_source2, preset_periods, time_text


class SuperpositionTests(unittest.TestCase):
    def test_presets_scale_with_the_gap(self):
        self.assertEqual(preset_periods("start", 8.0), 2.0)
        self.assertEqual(preset_periods("meet", 8.0), 4.0)
        self.assertEqual(preset_periods("later", 8.0), 8.625)
        self.assertEqual(preset_periods("meet", 5.0), 2.5)

    def test_fronts_meet_halfway(self):
        self.assertFalse(fronts_met(3.99, 8.0))
        self.assertTrue(fronts_met(4.0, 8.0))

    def test_source2_keeps_its_direction_at_the_separation(self):
        placed = place_source2(WaveParameters(source_position=(0.0, 3.0, 4.0)), 10.0)
        self.assertEqual(placed.source_position, (0.0, 6.0, 8.0))

    def test_source2_at_the_origin_defaults_to_plus_z(self):
        placed = place_source2(WaveParameters(source_position=(0.0, 0.0, 0.0)), 8.0)
        self.assertEqual(placed.source_position, (0.0, 0.0, 8.0))

    def test_time_text_names_each_stage(self):
        self.assertIn("ще не випромінюють", time_text(8.0, 0.0))
        self.assertIn("ще 2.00 T", time_text(8.0, 2.0))
        self.assertIn("зустрічаються", time_text(8.0, 4.0 + TIME_STEP / 4))
        self.assertIn("перекриття 2.00λ", time_text(8.0, 5.0))
        self.assertIn("повне перекриття", time_text(8.0, 9.0))


if __name__ == "__main__":
    unittest.main()
