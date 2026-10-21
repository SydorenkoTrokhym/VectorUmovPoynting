import unittest

from models import ScenarioConfiguration
from persistence import ConfigurationError, configuration_from_dict, configuration_to_dict


class PersistenceTests(unittest.TestCase):
    def test_schema_v2_round_trip(self):
        original = ScenarioConfiguration()
        original.waves[0].amplitude = 2.5
        restored = configuration_from_dict(configuration_to_dict(original))
        self.assertEqual(restored, original)

    def test_schema_v1_is_migrated(self):
        legacy = {
            "schema_version": 1,
            "waves": {
                "0": {"pol": "linear", "phase": 0, "amplitude": 1, "mod_type": "none",
                      "mod_freq": .1, "mod_index": .5, "n_periods": 2, "prop_dir": "+Z",
                      "dir_mode": "Базова вісь", "dir_vec": [0, 0, 1]},
                "1": {"pol": "linear", "phase": 0, "amplitude": 1, "mod_type": "none",
                      "mod_freq": .1, "mod_index": .5, "n_periods": 2, "prop_dir": "-Z",
                      "dir_mode": "Базова вісь", "dir_vec": [0, 0, -1],
                      "x0": 1, "y0": 2, "z0": 3},
            },
            "visualization": {},
        }
        restored = configuration_from_dict(legacy)
        self.assertEqual(restored.waves[1].source_position, (1, 2, 3))

    def test_invalid_enum_is_rejected(self):
        data = configuration_to_dict(ScenarioConfiguration())
        data["waves"][0]["polarization"] = "diagonal"
        with self.assertRaises(ConfigurationError):
            configuration_from_dict(data)

    def test_invalid_shape_is_rejected(self):
        data = configuration_to_dict(ScenarioConfiguration())
        data["waves"][0]["direction_vector"] = [1, 2]
        with self.assertRaises(ConfigurationError):
            configuration_from_dict(data)

    def test_unknown_schema_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            configuration_from_dict({"schema_version": 99})

    def test_source_separation_round_trip_and_default(self):
        original = ScenarioConfiguration(source_separation=5.5)
        self.assertEqual(configuration_from_dict(configuration_to_dict(original)), original)
        data = configuration_to_dict(ScenarioConfiguration())
        del data["source_separation"]
        self.assertEqual(configuration_from_dict(data).source_separation, 8.0)

    def test_invalid_source_separation_is_rejected(self):
        for value in (0.0, 25.0, "8", True):
            with self.subTest(value=value):
                data = configuration_to_dict(ScenarioConfiguration())
                data["source_separation"] = value
                with self.assertRaises(ConfigurationError):
                    configuration_from_dict(data)

    def test_earlier_v2_without_field_directions_is_compatible(self):
        data = configuration_to_dict(ScenarioConfiguration())
        for wave in data["waves"]:
            wave.pop("electric_direction")
            wave.pop("magnetic_direction")
        restored = configuration_from_dict(data)
        self.assertEqual(restored.waves[0].magnetic_direction, "+Y")
        self.assertEqual(restored.waves[1].magnetic_direction, "-Y")


if __name__ == "__main__":
    unittest.main()
