from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class WaveParameters:
    polarization: str = "linear"
    amplitude: float = 1.0
    phase_degrees: float = 0.0
    modulation_type: str = "none"
    modulation_frequency: float = 0.1
    modulation_index: float = 0.5
    periods: int = 2
    propagation_direction: str = "+Z"
    direction_mode: str = "Базова вісь"
    direction_vector: tuple[float, float, float] = (0.0, 0.0, 1.0)
    electric_direction: str = "+X"
    magnetic_direction: str = "+Y"
    source_position: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["direction_vector"] = list(self.direction_vector)
        data["source_position"] = list(self.source_position)
        return data


@dataclass(slots=True)
class VisualizationSettings:
    component_mode: str = "Узагальнений підхід"
    representation_mode: str = "Linear (3D)"
    background: str = "Чорний"
    projection: str = "3D"
    pixels_per_wavelength: int = 100


@dataclass(slots=True)
class ScenarioConfiguration:
    waves: list[WaveParameters] = field(default_factory=lambda: [
        WaveParameters(),
        WaveParameters(
            propagation_direction="-Z",
            direction_vector=(0.0, 0.0, -1.0),
            magnetic_direction="-Y",
            source_position=(0.0, 0.0, 2.0),
        ),
    ])
    visualization: VisualizationSettings = field(default_factory=VisualizationSettings)
