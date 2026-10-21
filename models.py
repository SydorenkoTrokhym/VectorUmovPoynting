from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


DEFAULT_SOURCE_SEPARATION = 8.0   # λ; the fronts then meet after 4 T


# Stored values are stable ASCII keys. The Ukrainian texts shown in the UI live in
# controls.py, so renaming a menu item never invalidates a saved parameter file.
class Polarization(StrEnum):
    LINEAR = "linear"
    CIRCULAR_RIGHT = "circular_right"
    CIRCULAR_LEFT = "circular_left"
    ELLIPTICAL = "elliptical"


class Modulation(StrEnum):
    NONE = "none"
    AM = "AM"
    FM = "FM"
    AFM = "AFM"


class DirectionMode(StrEnum):
    AXIS = "axis"        # propagation along one of ±X, ±Y, ±Z
    VECTOR = "vector"    # propagation along an arbitrary vector


class ComponentMode(StrEnum):
    COMBINED = "combined"   # E, B and S
    ELECTRIC = "electric"
    MAGNETIC = "magnetic"


class Representation(StrEnum):
    LINEAR = "linear"     # field curves and S arrows
    SURFACE = "surface"   # field ribbons and S tube


class Background(StrEnum):
    BLACK = "black"
    WHITE = "white"


class Projection(StrEnum):
    PERSPECTIVE = "3d"
    TOP = "xy"
    SIDE = "xz"
    FRONT = "yz"


@dataclass(slots=True)
class WaveParameters:
    polarization: Polarization = Polarization.LINEAR
    amplitude: float = 1.0
    phase_degrees: float = 0.0
    modulation_type: Modulation = Modulation.NONE
    modulation_frequency: float = 0.1
    modulation_index: float = 0.5
    periods: int = 2
    propagation_direction: str = "+Z"
    direction_mode: DirectionMode = DirectionMode.AXIS
    direction_vector: tuple[float, float, float] = (0.0, 0.0, 1.0)
    electric_direction: str = "+X"
    magnetic_direction: str = "+Y"
    source_position: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("polarization", "modulation_type", "direction_mode"):
            data[key] = str(data[key])
        data["direction_vector"] = list(self.direction_vector)
        data["source_position"] = list(self.source_position)
        return data


@dataclass(slots=True)
class VisualizationSettings:
    component_mode: ComponentMode = ComponentMode.COMBINED
    representation_mode: Representation = Representation.LINEAR
    background: Background = Background.BLACK
    projection: Projection = Projection.PERSPECTIVE
    pixels_per_wavelength: int = 100

    def to_dict(self) -> dict[str, Any]:
        return {key: str(value) if isinstance(value, StrEnum) else value
                for key, value in asdict(self).items()}


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
    # Superposition tab: distance between the two sources, in wavelengths. Source 2 lies
    # in the direction of wave 2's source position, at this distance from the origin.
    source_separation: float = DEFAULT_SOURCE_SEPARATION
