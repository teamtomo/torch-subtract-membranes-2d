import pydantic
import torch

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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        control_points = torch.as_tensor(self.path_control_points, device=device)
        path = Path2D(
            control_points=control_points,
            is_closed=self.path_is_closed,
            yx_coords=True
        )
        return path

    @property
    def signal_scale_spline(self) -> Path1D:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        control_points = torch.as_tensor(self.signal_scale_control_points, device=device)
        signal_scale_spline = Path1D(
            control_points=control_points,
            is_closed=self.path_is_closed
        )
        return signal_scale_spline
