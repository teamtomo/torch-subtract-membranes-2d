import pydantic

from torch_subtract_membranes_2d.path_models.path_1d import Path1D
from torch_subtract_membranes_2d.path_models.path_2d import Path2D


class Membrane2D(pydantic.BaseModel):
    profile_1d: list[float]  # (w, )
    weights_1d: list[float]  # (w, )
    path_control_points: list[tuple[float, float]]  # (b, 2)
    signal_scale_control_points: list[float]  # (b, )
    path_is_closed: bool

    @property
    def path(self) -> Path2D:
        path = Path2D(
            self.path_control_points,
            is_closed=self.path_is_closed,
            yx_coords=True
        )
        return path

    @property
    def signal_scale_spline(self) -> Path1D:
        signal_scale_spline = Path1D(
            control_points=self.signal_scale_control_points,
            is_closed=self.path_is_closed
        )
        return signal_scale_spline
