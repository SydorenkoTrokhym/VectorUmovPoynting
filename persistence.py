from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from models import ScenarioConfiguration, VisualizationSettings, WaveParameters
from geometry import field_directions

SCHEMA_VERSION = 2
POLARIZATIONS = {"linear", "circular_right", "circular_left", "elliptical"}
MODULATIONS = {"none", "AM", "FM", "AFM"}
DIRECTIONS = {"+Z", "-Z", "+X", "-X", "+Y", "-Y"}
DIRECTION_MODES = {"Базова вісь", "Довільний вектор"}


class ConfigurationError(ValueError):
    pass


def configuration_to_dict(configuration: ScenarioConfiguration) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "waves": [wave.to_dict() for wave in configuration.waves],
        "visualization": asdict(configuration.visualization),
    }


def _migrate_v1(data: dict[str, Any]) -> dict[str, Any]:
    old_waves = data.get("waves", {})
    waves = []
    for index in ("0", "1"):
        old = old_waves.get(index, {})
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
    return {"schema_version": 2, "waves": waves, "visualization": {
        "component_mode": old_viz.get("comp_mode", "Узагальнений підхід"),
        "representation_mode": old_viz.get("repr_mode", "Linear (3D)"),
        "background": old_viz.get("background", "Чорний"),
        "projection": old_viz.get("projection", "3D"),
        "pixels_per_wavelength": old_viz.get("pixels_per_wavelength", 100),
    }}


def configuration_from_dict(raw: Any) -> ScenarioConfiguration:
    if not isinstance(raw, dict):
        raise ConfigurationError("Configuration must be a JSON object")
    version = raw.get("schema_version")
    if version == 1:
        raw = _migrate_v1(raw)
    elif version != SCHEMA_VERSION:
        raise ConfigurationError(f"Unsupported schema version: {version!r}")
    waves_data = raw.get("waves")
    if not isinstance(waves_data, list) or len(waves_data) != 2:
        raise ConfigurationError("waves must contain exactly two entries")
    waves = [_parse_wave(item, index) for index, item in enumerate(waves_data)]
    viz_data = raw.get("visualization", {})
    try:
        visualization = VisualizationSettings(**viz_data)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"Invalid visualization settings: {exc}") from exc
    if not 10 <= visualization.pixels_per_wavelength <= 2000:
        raise ConfigurationError("pixels_per_wavelength must be in [10, 2000]")
    return ScenarioConfiguration(waves, visualization)


def _parse_wave(data: Any, index: int) -> WaveParameters:
    if not isinstance(data, dict):
        raise ConfigurationError(f"wave {index} must be an object")
    try:
        values = dict(data)
        values["direction_vector"] = tuple(values.get("direction_vector", (0, 0, 1)))
        values["source_position"] = tuple(values.get("source_position", (0, 0, 0)))
        values.setdefault("electric_direction", "+X")
        values.setdefault(
            "magnetic_direction",
            "-Y" if values.get("propagation_direction") == "-Z" else "+Y",
        )
        wave = WaveParameters(**values)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"Invalid wave {index}: {exc}") from exc
    if wave.polarization not in POLARIZATIONS or wave.modulation_type not in MODULATIONS:
        raise ConfigurationError(f"Invalid wave {index} polarization or modulation")
    if wave.propagation_direction not in DIRECTIONS or wave.direction_mode not in DIRECTION_MODES:
        raise ConfigurationError(f"Invalid wave {index} direction")
    try:
        flow_direction, _, _ = field_directions(wave.electric_direction, wave.magnetic_direction)
    except ValueError as exc:
        raise ConfigurationError(f"Invalid wave {index} field directions: {exc}") from exc
    if (wave.direction_mode == "Базова вісь" and
            wave.propagation_direction in ("+Z", "-Z")):
        expected_z = 1.0 if wave.propagation_direction == "+Z" else -1.0
        if not np.isclose(flow_direction[2], expected_z):
            raise ConfigurationError(f"Wave {index}: E × B contradicts propagation direction")
    if len(wave.direction_vector) != 3 or len(wave.source_position) != 3:
        raise ConfigurationError(f"Wave {index} vectors must have three components")
    if not 0.1 <= wave.amplitude <= 100 or not 0 <= wave.phase_degrees <= 360:
        raise ConfigurationError(f"Wave {index} amplitude or phase is out of range")
    if not 1 <= wave.periods <= 10 or not 0 <= wave.modulation_index <= 1:
        raise ConfigurationError(f"Wave {index} periods or modulation index is out of range")
    if not 0.01 <= wave.modulation_frequency <= 1e9:
        raise ConfigurationError(f"Wave {index} modulation frequency is out of range")
    if any(abs(value) > 10 for value in wave.source_position):
        raise ConfigurationError(f"Wave {index} source position is out of range")
    return wave
