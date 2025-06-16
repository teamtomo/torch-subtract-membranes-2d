import einops
import numpy as np
import scipy
import torch
from torch_cubic_spline_grids.interpolate_grids import interpolate_grid_1d
from torch_cubic_spline_grids._constants import CUBIC_CATMULL_ROM_MATRIX


class Path2D:
    """2D path with cubic B-spline interpolation"""

    def __init__(
            self,
            control_points: torch.Tensor | np.ndarray,
            is_closed: bool,
            yx_coords: bool = False
    ):
        super().__init__()

        control_points = torch.as_tensor(control_points, dtype=torch.float32)
        self.control_points: torch.Tensor = control_points  # (b, 2)
        self.is_closed: bool = is_closed
        self.yx_coords: bool = yx_coords

    @property
    def is_clockwise(self) -> bool:
        """Check whether path is clockwise.

         Convention:
         - x coordinate increases moving right
         - y coordinate increases moving up
        """
        # sample points along the path
        n_points = 2000
        u = torch.linspace(0, 1, steps=n_points)

        with torch.no_grad():
            sampled_points = self.interpolate(u)
            sampled_points = sampled_points.detach().cpu().numpy()

        # check signed area of polygon enclosed by points
        # sum of (x2-x1)*(y2+y1) for each point pair
        if self.yx_coords is True:
            x1 = sampled_points[:-1, 1]
            x2 = sampled_points[1:, 1]
            y1 = sampled_points[:-1, 0]
            y2 = sampled_points[1:, 0]
        else:  # xy case
            x1 = sampled_points[:-1, 0]
            x2 = sampled_points[1:, 0]
            y1 = sampled_points[:-1, 1]
            y2 = sampled_points[1:, 1]

        area = float(0.5 * np.sum((x2 - x1) * (y2 + y1)))
        return area > 0

    def as_uniformly_spaced(self, spacing: float) -> "Path2D":
        """Make a new path with uniform spacing between control points."""
        # sample points along the current path
        n_points = 2000
        u_values = torch.linspace(0, 1, steps=n_points, device=self.control_points.device)

        with torch.no_grad():
            sampled_points = self.interpolate(u_values)
            sampled_points = sampled_points.detach().cpu().numpy()

        # calculate cumulative path length at each point
        diffs = np.diff(sampled_points, axis=0)
        segment_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
        cumulative_length = np.concatenate(([0], np.cumsum(segment_lengths)))
        total_length = cumulative_length[-1]

        # calculate number pf points in output
        num_points = int(np.ceil(total_length / spacing)) + 1
        num_points = max(2, num_points)  # ensure at least 2 points

        # resample evenly spaced points, one dimension at a time
        new_distances = np.linspace(0, total_length, num=num_points)
        resampled_coords = np.zeros((num_points, sampled_points.shape[-1]))

        for dim in range(sampled_points.shape[-1]):
            spline = scipy.interpolate.CubicSpline(
                x=cumulative_length, y=sampled_points[..., dim]
            )
            resampled_coords[:, dim] = spline(new_distances)

        # don't include endpoint twice if path is closed
        if self.is_closed:
            resampled_coords = resampled_coords[:-1]

        # force tensor on correct device
        resampled_coords = torch.as_tensor(
            resampled_coords, dtype=torch.float32, device=self.control_points.device
        )

        return self.__class__(control_points=resampled_coords, is_closed=self.is_closed)

    def as_reversed(self) -> "Path2D":
        path = Path2D(
            control_points=torch.flip(self.control_points, dims=(0,)),
            is_closed=self.is_closed,
            yx_coords=self.yx_coords,
        )
        return path

    def interpolate(self, u: torch.Tensor) -> torch.Tensor:
        # expand control point set and map u onto new control points in closed case
        if self.is_closed:
            control_points, u = self._handle_closed_path_control_points_and_parameter(u=u)
        else:
            control_points = self.control_points

        # force same device
        u = u.to(self.control_points.device)
        interpolation_matrix = CUBIC_CATMULL_ROM_MATRIX.to(self.control_points.device)

        # do interpolation
        samples = interpolate_grid_1d(
            grid=einops.rearrange(control_points, "b c -> c b"),
            u=einops.rearrange(u, "b -> b 1"),
            matrix=interpolation_matrix
        )
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

    def get_tangents(self, u: torch.Tensor) -> torch.Tensor:
        """
        Calculate unit tangent vectors at interpolated points along the path.

        Parameters
        ----------
        u : torch.Tensor
            `(b, )` array of fractional positions along the path.
            Values should be in the interval [0, 1].

        Returns
        -------
        results : torch.Tensor
            `(b, 2)` array of unit tangent vectors at fractional positions `u`.
        """
        u = torch.clamp(u, min=0, max=0.99999)
        with torch.no_grad():
            p1 = self.interpolate(u)
            p2 = self.interpolate(u + 0.00001)
            tangents = p2 - p1
            tangents = tangents / torch.linalg.norm(tangents, dim=-1, keepdim=True)
        return tangents

    def get_normals(self, u: torch.Tensor) -> torch.Tensor:
        """
        Calculate unit normal vectors at interpolated points along the path.

        Parameters
        ----------
        u : torch.Tensor
            `(b, )` array of fractional positions along the path.
            Values should be in the interval [0, 1].
        """
        tangents = self.get_tangents(u)

        # grab x and y components
        if self.yx_coords is True:
            y, x = einops.rearrange(tangents, "b yx -> yx b")
        else:
            x, y = einops.rearrange(tangents, "b xy -> xy b")

        # calculate new x and new y
        new_x = y
        new_y = -1 * x

        # to (b, 2) with correct ordering
        normals = einops.rearrange([new_x, new_y], "xy b -> b xy")
        normals = torch.flip(normals, dims=(-1,)) if self.yx_coords is True else normals

        return normals

    def get_closest_u(
            self,
            query_points: torch.Tensor | np.ndarray,
            n_refinement_steps: int = 50
    ):
        """
        Find u that interpolates to points on the path closest to query points.

        Parameters
        ----------
        query_points : torch.Tensor
            `(b, 2)` array of 2D points to find the closest u for
        n_initial_samples : int
            Number of initial samples to evaluate
        n_refinement_steps : int
            Number of iterative refinement steps

        Returns
        -------
        closest_u : torch.Tensor
            `(b, )` array of parameter values that gives the closest point on the path
        """
        # Convert points to tensor if needed
        query_points = torch.as_tensor(
            query_points, dtype=torch.float32, device=self.control_points.device
        )

        # Use heuristic to set initial number of samples if not set
        n_control_points = len(self.control_points)
        n_initial_samples = n_control_points * 100

        # first find closest point on oversampled path
        with torch.no_grad():
            # make initial equally spaced points
            u_initial = torch.linspace(0, 1, steps=n_initial_samples)
            path_points = self.interpolate(u_initial)

            # build KDTree for fast nearest neighbour lookup
            tree = scipy.spatial.KDTree(path_points.cpu().numpy())

            # query for nearest point on path for each query point
            _, idx = tree.query(query_points.cpu().numpy())
            closest_u_initial = u_initial[idx]

        # then iteratively refine around that closest point
        closest_u = torch.tensor(
            closest_u_initial.clone().detach().cpu().numpy(),
            dtype=torch.float32,
            device=self.control_points.device,
            requires_grad=True
        )

        optimizer = torch.optim.Adam([closest_u], lr=0.001)
        for _ in range(n_refinement_steps):
            optimizer.zero_grad()
            current_points = self.interpolate(closest_u)
            difference = current_points - query_points
            squared_distances = einops.reduce(difference ** 2, "b d -> b", reduction="sum")
            loss = torch.mean(squared_distances)
            loss.backward()
            optimizer.step()

        closest_u = torch.clamp(closest_u, min=0, max=1)
        return closest_u

    def get_signed_distance(
            self,
            query_points: torch.Tensor | np.ndarray,
            closest_u: torch.Tensor | np.ndarray | None = None,
    ):
        if closest_u is None:
            closest_u = self.get_closest_u(query_points)  # (b, )
        closest_point_on_path = self.interpolate(closest_u)  # (b, 2)
        path_to_query_point_vector = query_points - closest_point_on_path  # (b, 2)
        distance = torch.sum(path_to_query_point_vector ** 2, dim=-1) ** 0.5  # (b, )
        normal = self.get_normals(closest_u)  # (b, 2)
        dot_product = einops.einsum(path_to_query_point_vector, normal, "b yx, b yx -> b")
        sign = torch.sign(dot_product)
        return sign * distance

    def estimated_length(self, n_samples: int = 2000) -> float:
        """Estimated length of the path by sampling points along it."""
        u = torch.linspace(0, 1, steps=n_samples)

        with torch.no_grad():
            points = self.interpolate(u)
            diffs = torch.diff(points, dim=0)
            segment_lengths = torch.sqrt((diffs ** 2).sum(dim=-1))
            path_length = segment_lengths.sum().item()

        return path_length


if __name__ == "__main__":
    # open path setup
    x = np.linspace(0, 1, 50)
    y = np.sin(8 * np.pi * x)
    control_points = einops.rearrange([y * 10 + 15, x * 100 + 5], "yx b -> b yx")
    path = Path2D(control_points=control_points, is_closed=False, yx_coords=True)

    # open path usage
    u = torch.linspace(0, 1, 200)
    interpolated = path.interpolate(u)
    normals = path.get_normals(u)

    from torch_subtract_membranes_2d.utils.path_utils import find_pixels_around_path

    idx_h, idx_w = find_pixels_around_path(
        path=path,
        image_shape=(30, 110),
        maximum_distance=5,
    )
    pixel_positions = einops.rearrange([idx_h, idx_w], "yx b -> b yx")
    closest_u = path.get_closest_u(pixel_positions)
    signed_distance = path.get_signed_distance(pixel_positions)

    sdf = torch.zeros((30, 110))
    sdf[idx_h, idx_w] = signed_distance

    # viz
    import napari

    viewer = napari.Viewer()
    viewer.add_image(sdf.detach().numpy(), name="signed distance field (max 5)", colormap="PiYG")
    viewer.add_points(interpolated, name="interpolated points", face_color="orange", size=1)
    viewer.add_points(control_points, name="control_points", face_color="cornflowerblue", size=1)
    napari_vectors = einops.rearrange([interpolated, normals], "v b yx -> b v yx")
    viewer.add_vectors(napari_vectors, name="normal vectors", length=3)
    napari.run()

    # closed path setup
    theta = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    x = np.cos(theta)
    y = np.sin(theta)
    control_points = np.stack([10 * y + 15, 30 * x + 35], axis=-1)
    path = Path2D(control_points=control_points, is_closed=True, yx_coords=True)

    # closed path usage
    print(f"{path.is_clockwise=}")
    u = torch.tensor(np.linspace(0, 1, 100, endpoint=False)).float()
    interpolated = path.interpolate(u)
    normals = path.get_normals(u)

    idx_h, idx_w = find_pixels_around_path(
        path=path,
        image_shape=(30, 70),
        maximum_distance=5,
    )
    pixel_positions = einops.rearrange([idx_h, idx_w], "yx b -> b yx")
    closest_u = path.get_closest_u(pixel_positions)
    signed_distance = path.get_signed_distance(pixel_positions)

    sdf = torch.zeros((30, 70))
    sdf[idx_h, idx_w] = signed_distance

    # viz
    viewer = napari.Viewer()
    viewer.add_image(sdf.detach().numpy(), name="signed distance field (max 5)", colormap="PiYG")
    viewer.add_points(interpolated, name="interpolated points", face_color="orange", size=1)
    viewer.add_points(control_points, name="control_points", face_color="cornflowerblue", size=1)
    napari_vectors = einops.rearrange([interpolated, normals], "v b yx -> b v yx")
    viewer.add_vectors(napari_vectors, name="normal vectors", length=3)
    napari.run()
