from __future__ import annotations

from dataclasses import dataclass

import pyqtgraph.opengl as gl


@dataclass(slots=True)
class WaveRenderObjects:
    """All graphics primitives owned by one wave, in every representation mode."""

    poynting_vectors: gl.GLLinePlotItem
    electric_curve: gl.GLLinePlotItem
    magnetic_curve: gl.GLLinePlotItem
    electric_surface: gl.GLMeshItem
    magnetic_surface: gl.GLMeshItem
    poynting_surface: gl.GLMeshItem

    @classmethod
    def create(cls, view: gl.GLViewWidget, electric_color, magnetic_color, poynting_color,
               curve_width: float = 3):
        objects = cls(
            gl.GLLinePlotItem(width=4, color=poynting_color, mode="lines", glOptions="translucent"),
            gl.GLLinePlotItem(width=curve_width, color=electric_color, mode="line_strip", glOptions="translucent"),
            gl.GLLinePlotItem(width=curve_width, color=magnetic_color, mode="line_strip", glOptions="translucent"),
            gl.GLMeshItem(smooth=False, color=electric_color[:3] + (0.5,), shader="shaded", glOptions="translucent"),
            gl.GLMeshItem(smooth=False, color=magnetic_color[:3] + (0.5,), shader="shaded", glOptions="translucent"),
            gl.GLMeshItem(smooth=False, color=poynting_color[:3] + (0.5,), shader="shaded", glOptions="translucent"),
        )
        for item in objects.items():
            item.setVisible(False)
            view.addItem(item)
        return objects

    def items(self):
        return (
            self.poynting_vectors, self.electric_curve, self.magnetic_curve,
            self.electric_surface, self.magnetic_surface, self.poynting_surface,
        )

    def hide(self):
        for item in self.items():
            item.setVisible(False)
