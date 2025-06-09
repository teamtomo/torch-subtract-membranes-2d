import torch

from torch_trace_membranes_2d.path_models.path_1d import Path1D
from torch_trace_membranes_2d.path_models.path_2d import Path2D


class Membrane2D:
    profile_1d: torch.Tensor  # (w, )
    weights_1d: torch.Tensor  # (w, )
    path_control_points: torch.Tensor  # (b, 2)
    signal_scale_control_points: torch.Tensor  # (b, )
    path_is_closed: bool

    def __init__(
        self,
        profile_1d: torch.Tensor,
        weights_1d: torch.Tensor,
        path_control_points: torch.Tensor,
        signal_scale_control_points: torch.Tensor,
        path_is_closed: bool,
    ):
        self.profile_1d = profile_1d
        self.weights_1d = weights_1d
        self.path_control_points = path_control_points
        self.path_is_closed = path_is_closed
        self.signal_scale_control_points = signal_scale_control_points

        self.path = Path2D(
            self.path_control_points,
            is_closed=self.path_is_closed,
            yx_coords=True
        )
        self.signal_scale_spline = Path1D(
            control_points=signal_scale_control_points, is_closed=self.path_is_closed
        )
