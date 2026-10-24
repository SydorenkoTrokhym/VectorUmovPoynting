import unittest

from models import (Background, ComponentMode, DirectionMode, Projection, Representation,
                    ScenarioConfiguration)
from persistence import ConfigurationError, configuration_from_dict, configuration_to_dict


class PersistenceTests(unittest.TestCase):
    def test_schema_v3_round_trip(self):
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
        # v1 placed source 2 at its own position: that distance is the separation.
        self.assertAlmostEqual(restored.source_separation, 14 ** 0.5)

    def test_schema_v1_without_source_position_keeps_the_default_separation(self):
        legacy = {"schema_version": 1, "waves": {"0": {}, "1": {}}, "visualization": {}}
        self.assertEqual(configuration_from_dict(legacy).source_separation, 8.0)

    def test_files_store_stable_keys_not_ui_texts(self):
        data = configuration_to_dict(ScenarioConfiguration())
        self.assertEqual(data["schema_version"], 3)
        self.assertEqual(data["waves"][0]["direction_mode"], "axis")
        self.assertEqual(data["visualization"], {
            "component_mode": "combined", "representation_mode": "linear",
            "background": "black", "projection": "3d", "pixels_per_wavelength": 100})

    def test_schema_v2_ui_texts_are_migrated(self):
        data = configuration_to_dict(ScenarioConfiguration())
        data["schema_version"] = 2
        for wave in data["waves"]:
            wave["direction_mode"] = "Довільний вектор"
        data["visualization"] = {
            "component_mode": "Магнітна (B)", "representation_mode": "Surface (3D)",
            "background": "Білий", "projection": "XZ (Side)", "pixels_per_wavelength": 250}
        restored = configuration_from_dict(data)
        self.assertEqual(restored.waves[1].direction_mode, DirectionMode.VECTOR)
        viz = restored.visualization
        self.assertEqual((viz.component_mode, viz.representation_mode, viz.background,
                          viz.projection, viz.pixels_per_wavelength),
                         (ComponentMode.MAGNETIC, Representation.SURFACE, Background.WHITE,
                          Projection.SIDE, 250))

    def test_invalid_visualization_value_is_rejected(self):
        data = configuration_to_dict(ScenarioConfiguration())
        data["visualization"]["background"] = "Чорний"   # a v2 text is not a v3 key
        with self.assertRaises(ConfigurationError):
            configuration_from_dict(data)

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

    def test_wrongly_typed_values_are_rejected_not_crashing(self):
        cases = [("waves", "amplitude", "1"), ("waves", "phase_degrees", None),
                 ("waves", "periods", 2.5), ("waves", "periods", True),
                 ("waves", "modulation_index", [0.5]), ("waves", "source_position", 5),
                 ("waves", "source_position", [0, "1", 0]), ("waves", "direction_vector", None),
                 ("waves", "propagation_direction", ["+Z"]), ("waves", "electric_direction", 1),
                 ("waves", "polarization", ["linear"]),
                 ("visualization", "pixels_per_wavelength", "100"),
                 ("visualization", "pixels_per_wavelength", 99.5)]
        for section, key, value in cases:
            with self.subTest(key=key, value=value):
                data = configuration_to_dict(ScenarioConfiguration())
                target = data["waves"][0] if section == "waves" else data["visualization"]
                target[key] = value
                with self.assertRaises(ConfigurationError):
                    configuration_from_dict(data)

    def test_whole_number_floats_are_read_as_integers(self):
        data = configuration_to_dict(ScenarioConfiguration())
        data["waves"][0]["periods"] = 3.0
        data["visualization"]["pixels_per_wavelength"] = 150.0
        restored = configuration_from_dict(data)
        self.assertEqual((restored.waves[0].periods, restored.visualization.pixels_per_wavelength),
                         (3, 150))
        self.assertIsInstance(restored.waves[0].periods, int)

    def test_malformed_v1_sections_are_rejected(self):
        for waves in ([], {"0": "linear"}):
            with self.subTest(waves=waves):
                with self.assertRaises(ConfigurationError):
                    configuration_from_dict({"schema_version": 1, "waves": waves})

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
