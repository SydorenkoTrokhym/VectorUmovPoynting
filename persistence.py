from __future__ import annotations

from enum import StrEnum
from typing import Any

import numpy as np

from models import (DEFAULT_SOURCE_SEPARATION, Background, ComponentMode, DirectionMode,
                    Modulation, Polarization, Projection, Representation,
                    ScenarioConfiguration, VisualizationSettings, WaveParameters)
from geometry import field_directions

SCHEMA_VERSION = 3
DIRECTIONS = {"+Z", "-Z", "+X", "-X", "+Y", "-Y"}
SEPARATION_RANGE = (0.5, 20.0)   # distance between superposition sources, in λ

# Schema v1/v2 stored the Ukrainian UI texts; v3 stores the enum keys instead.
_LEGACY_VALUES = {
    "direction_mode": {"Базова вісь": DirectionMode.AXIS,
                       "Довільний вектор": DirectionMode.VECTOR},
    "component_mode": {"Узагальнений підхід": ComponentMode.COMBINED,
                       "Електрична (E)": ComponentMode.ELECTRIC,
                       "Магнітна (B)": ComponentMode.MAGNETIC},
    "representation_mode": {"Linear (3D)": Representation.LINEAR,
                            "Surface (3D)": Representation.SURFACE},
    "background": {"Чорний": Background.BLACK, "Білий": Background.WHITE},
    "projection": {"3D": Projection.PERSPECTIVE, "XY (Top)": Projection.TOP,
                   "XZ (Side)": Projection.SIDE, "YZ (Front)": Projection.FRONT},
}


class ConfigurationError(ValueError):
    pass


def configuration_to_dict(configuration: ScenarioConfiguration) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "waves": [wave.to_dict() for wave in configuration.waves],
        "visualization": configuration.visualization.to_dict(),
        "source_separation": configuration.source_separation,
    }


def _migrate_v1(data: dict[str, Any]) -> dict[str, Any]:
    old_waves = data.get("waves", {})
    if not isinstance(old_waves, dict):
        raise ConfigurationError("waves must be an object in schema version 1")
    waves = []
    for index in ("0", "1"):
        old = old_waves.get(index, {})
        if not isinstance(old, dict):
            raise ConfigurationError(f"wave {index} must be an object")
        waves.append({
            "polarization": old.get("pol", "linear"),
            "amplitude": old.get("amplitude", 1.0),
            "phase_degrees": old.get("phase", 0.0),
            "modulation_type": old.get("mod_type", "none"),
            "modulation_frequency": old.get("mod_freq", 0.1),
            "modulation_index": old.get("mod_index", 0.5),
            "periods": old.get("n_periods", 2),
            "propagation_direction": old.get("prop_dir", "+Z" if index == "0" else "-Z"),
            "direction_mode": old.get("dir_mode", "Базова вісь"),
            "direction_vector": old.get("dir_vec", [0, 0, 1 if index == "0" else -1]),
            "electric_direction": old.get("electric_direction", "+X"),
            "magnetic_direction": old.get("magnetic_direction", "+Y" if index == "0" else "-Y"),
            "source_position": [old.get("x0", 0), old.get("y0", 0), old.get("z0", 0)],
        })
    old_viz = data.get("visualization", {})
    if not isinstance(old_viz, dict):
        raise ConfigurationError("visualization must be an object")
    # v1 had no separation setting: the superposition put source 2 at its own
    # position, so that distance is the separation the file meant.
    separation = DEFAULT_SOURCE_SEPARATION
    try:
        distance = float(np.linalg.norm(np.asarray(waves[1]["source_position"], dtype=float)))
    except (TypeError, ValueError):
        distance = 0.0   # an invalid position is reported when the wave is parsed
    if SEPARATION_RANGE[0] <= distance <= SEPARATION_RANGE[1]:
        separation = distance
    return {"schema_version": 2, "waves": waves, "source_separation": separation,
            "visualization": {
        "component_mode": old_viz.get("comp_mode", "Узагальнений підхід"),
        "representation_mode": old_viz.get("repr_mode", "Linear (3D)"),
        "background": old_viz.get("background", "Чорний"),
        "projection": old_viz.get("projection", "3D"),
        "pixels_per_wavelength": old_viz.get("pixels_per_wavelength", 100),
    }}


def _migrate_v2(data: dict[str, Any]) -> dict[str, Any]:
    """Replace the Ukrainian UI texts of schema v2 with enum keys."""
    def convert(section: Any) -> Any:
        if not isinstance(section, dict):
            return section
        converted = dict(section)
        for key, aliases in _LEGACY_VALUES.items():
            if key in converted and converted[key] in aliases:
                converted[key] = str(aliases[converted[key]])
        return converted

    migrated = dict(data, schema_version=3)
    waves = data.get("waves")
    if isinstance(waves, list):
        migrated["waves"] = [convert(wave) for wave in waves]
    if "visualization" in data:
        migrated["visualization"] = convert(data["visualization"])
    return migrated


def _enum(enum_type: type[StrEnum], value: Any, what: str) -> StrEnum:
    try:
        return enum_type(value)
    except (TypeError, ValueError):
        raise ConfigurationError(f"Invalid {what}: {value!r}") from None


def _number(value: Any, what: str) -> float:
    # bool is an int subclass, but true/false in a file is never a valid number
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value):
        raise ConfigurationError(f"{what} must be a number, not {value!r}")
    return float(value)


def _integer(value: Any, what: str) -> int:
    number = _number(value, what)
    if not number.is_integer():
        raise ConfigurationError(f"{what} must be a whole number, not {value!r}")
    return int(number)


def _vector(value: Any, what: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ConfigurationError(f"{what} must have three components")
    return tuple(_number(component, what) for component in value)


def configuration_from_dict(raw: Any) -> ScenarioConfiguration:
    if not isinstance(raw, dict):
        raise ConfigurationError("Configuration must be a JSON object")
    version = raw.get("schema_version")
    if version == 1:
        raw = _migrate_v1(raw)
        version = 2
    if version == 2:
        raw = _migrate_v2(raw)
    elif version != SCHEMA_VERSION:
        raise ConfigurationError(f"Unsupported schema version: {version!r}")
    waves_data = raw.get("waves")
    if not isinstance(waves_data, list) or len(waves_data) != 2:
        raise ConfigurationError("waves must contain exactly two entries")
    waves = [_parse_wave(item, index) for index, item in enumerate(waves_data)]
    visualization = _parse_visualization(raw.get("visualization", {}))
    separation = raw.get("source_separation", DEFAULT_SOURCE_SEPARATION)
    if (not isinstance(separation, (int, float)) or isinstance(separation, bool)
            or not SEPARATION_RANGE[0] <= separation <= SEPARATION_RANGE[1]):
        raise ConfigurationError(
            f"source_separation must be a number in [{SEPARATION_RANGE[0]}, {SEPARATION_RANGE[1]}]")
    return ScenarioConfiguration(waves, visualization, float(separation))


def _parse_visualization(data: Any) -> VisualizationSettings:
    if not isinstance(data, dict):
        raise ConfigurationError("visualization must be an object")
    try:
        visualization = VisualizationSettings(**data)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"Invalid visualization settings: {exc}") from exc
    visualization.component_mode = _enum(
        ComponentMode, visualization.component_mode, "component mode")
    visualization.representation_mode = _enum(
        Representation, visualization.representation_mode, "representation")
    visualization.background = _enum(Background, visualization.background, "background")
    visualization.projection = _enum(Projection, visualization.projection, "projection")
    visualization.pixels_per_wavelength = _integer(
        visualization.pixels_per_wavelength, "pixels_per_wavelength")
    if not 10 <= visualization.pixels_per_wavelength <= 2000:
        raise ConfigurationError("pixels_per_wavelength must be in [10, 2000]")
    return visualization


def _parse_wave(data: Any, index: int) -> WaveParameters:
    if not isinstance(data, dict):
        raise ConfigurationError(f"wave {index} must be an object")
    values = dict(data)
    values["direction_vector"] = _vector(
        values.get("direction_vector", (0, 0, 1)), f"Wave {index} direction vector")
    values["source_position"] = _vector(
        values.get("source_position", (0, 0, 0)), f"Wave {index} source position")
    for key in ("amplitude", "phase_degrees", "modulation_frequency", "modulation_index"):
        if key in values:
            values[key] = _number(values[key], f"Wave {index} {key}")
    if "periods" in values:
        values["periods"] = _integer(values["periods"], f"Wave {index} periods")
    for key in ("propagation_direction", "electric_direction", "magnetic_direction"):
        if key in values and not isinstance(values[key], str):
            raise ConfigurationError(f"Wave {index} {key} must be text, not {values[key]!r}")
    try:
        values.setdefault("electric_direction", "+X")
        values.setdefault(
            "magnetic_direction",
            "-Y" if values.get("propagation_direction") == "-Z" else "+Y",
        )
        wave = WaveParameters(**values)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"Invalid wave {index}: {exc}") from exc
    wave.polarization = _enum(Polarization, wave.polarization, f"wave {index} polarization")
    wave.modulation_type = _enum(Modulation, wave.modulation_type, f"wave {index} modulation")
    wave.direction_mode = _enum(DirectionMode, wave.direction_mode, f"wave {index} direction mode")
    if wave.propagation_direction not in DIRECTIONS:
        raise ConfigurationError(f"Invalid wave {index} direction")
    try:
        flow_direction, _, _ = field_directions(wave.electric_direction, wave.magnetic_direction)
    except ValueError as exc:
        raise ConfigurationError(f"Invalid wave {index} field directions: {exc}") from exc
    if (wave.direction_mode == DirectionMode.AXIS and
            wave.propagation_direction in ("+Z", "-Z")):
        expected_z = 1.0 if wave.propagation_direction == "+Z" else -1.0
        if not np.isclose(flow_direction[2], expected_z):
            raise ConfigurationError(f"Wave {index}: E × B contradicts propagation direction")
    if not 0.1 <= wave.amplitude <= 100 or not 0 <= wave.phase_degrees <= 360:
        raise ConfigurationError(f"Wave {index} amplitude or phase is out of range")
    if not 1 <= wave.periods <= 10 or not 0 <= wave.modulation_index <= 1:
        raise ConfigurationError(f"Wave {index} periods or modulation index is out of range")
    if not 0.01 <= wave.modulation_frequency <= 1e9:
        raise ConfigurationError(f"Wave {index} modulation frequency is out of range")
    if any(abs(value) > 10 for value in wave.source_position):
        raise ConfigurationError(f"Wave {index} source position is out of range")
    return wave
