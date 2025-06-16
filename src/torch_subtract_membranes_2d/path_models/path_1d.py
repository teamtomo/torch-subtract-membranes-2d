import einops
import numpy as np
import scipy
import torch
from torch_cubic_spline_grids.interpolate_grids import interpolate_grid_1d
from torch_cubic_spline_grids._constants import CUBIC_CATMULL_ROM_MATRIX


class Path1D:
    """1D path with cubic B-spline interpolation"""

    def __init__(
        self,
        control_points: torch.Tensor | np.ndarray,
        is_closed: bool,
    ):
        super().__init__()

        control_points = torch.as_tensor(control_points, dtype=torch.float32)
        self.control_points: torch.Tensor = control_points  # (b, )
        self.is_closed: bool = is_closed

    def interpolate(self, u: torch.Tensor) -> torch.Tensor:
        # expand control point set and map u onto new control points in closed case
        if self.is_closed:
            control_points, u = self._handle_closed_path_control_points_and_parameter(u=u)
        else:
            control_points = self.control_points

        # force all tensors to same device for interpolation
        u = u.to(self.control_points.device)
        interpolation_matrix = CUBIC_CATMULL_ROM_MATRIX.to(self.control_points.device)
        samples = interpolate_grid_1d(
            grid=einops.rearrange(control_points, "b -> 1 b"),
            u=einops.rearrange(u, "b -> b 1"),
            matrix=interpolation_matrix
        )
        # squeeze out the dimension that was added for grid interpolation
        samples = einops.rearrange(samples, "b 1 -> b")
        return samples

    def _handle_closed_path_control_points_and_parameter(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # add extra control points at each end to ensure continuity
        original_control_points = self.control_points
        control_points = torch.cat(
            [
                original_control_points[-3:],  # Last three points
                original_control_points,  # All original points
                original_control_points[:3],  # First three points
            ],
            dim=0
        )

        # map the input u from [0,1] to the valid range in the expanded control points
        # - u=0 should map to index 3 (after the wrapped points at beginning)
        # - u=1 should map to index n + 3 (the first wrapped point at the end)
        n = len(original_control_points)
        n_total = len(control_points)
        new_u0 = 3 / (n_total - 1)
        new_interval_width = n / (n_total - 1)  # not n-1 because we include one extra point
        u = u * new_interval_width + new_u0

        return control_points, u
