import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from .debug_utils import IS_DEBUG

# Set matplotlib resolution once when debug mode is enabled
if IS_DEBUG:
    import matplotlib as mpl
    mpl.rcParams['figure.dpi'] = 300


def set_matplotlib_resolution():
    import matplotlib as mpl
    mpl.rcParams['figure.dpi'] = 300


def plot_preprocessed_image(image: torch.Tensor) -> Figure:
    fig, ax = plt.subplots()
    ax.set_title("preprocessed image")
    ax.imshow(image.detach().cpu().numpy(), cmap="gray")
    return fig


def plot_membrane_mask(membrane_mask: torch.Tensor) -> Figure:
    fig, ax = plt.subplots()
    ax.set_title("membrane mask")
    ax.imshow(membrane_mask.cpu().numpy())
    return fig


def plot_paths_on_membrane_mask(membrane_mask: torch.Tensor, membranes, title: str = "paths on membrane mask") -> Figure:
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.imshow(membrane_mask.detach().cpu().numpy(), cmap="gray")
    for membrane in membranes:
        if hasattr(membrane, 'path'):
            yx = membrane.path.interpolate(u=torch.linspace(0, 1, steps=100))
        else:
            yx = membrane.interpolate(u=torch.linspace(0, 1, steps=100))
        ax.plot(yx[:, -1].detach().cpu().numpy(), yx[:, -2].detach().cpu().numpy())
    return fig


def plot_membrane_refinement_comparison(
    original_membranogram,
    original_average_2d_scaled,
    refined_membranogram: torch.Tensor,
    refined_average_2d_scaled: torch.Tensor
) -> Figure:
    fig, ax = plt.subplots(ncols=4)
    ax[0].set_title("initial\ndata")
    ax[0].imshow(original_membranogram, cmap="gray")
    ax[1].set_title("initial\nreconstruction")
    ax[1].imshow(original_average_2d_scaled, cmap="gray")
    ax[2].set_title("refined\ndata")
    ax[2].imshow(refined_membranogram.detach().cpu().numpy(), cmap="gray")
    ax[3].set_title("refined\nreconstruction")
    ax[3].imshow(refined_average_2d_scaled.detach().cpu().numpy(), cmap="gray")
    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')
    ax[3].axis('off')
    fig.tight_layout()
    return fig


def plot_subtraction_results(
    image: torch.Tensor,
    membrane_image: torch.Tensor,
    subtracted: torch.Tensor,
    subtraction_factor: float
) -> Figure:
    from torch_fourier_rescale import fourier_rescale_2d
    
    image_downscaled, _ = fourier_rescale_2d(image, source_spacing=1, target_spacing=12)
    membrane_image_downscaled, _ = fourier_rescale_2d(membrane_image, source_spacing=1, target_spacing=12)
    subtracted_downscaled, _ = fourier_rescale_2d(subtracted, source_spacing=1, target_spacing=12)

    fig, axs = plt.subplots(ncols=3, figsize=(12, 4))
    for ax in axs:
        ax.set_axis_off()
    axs[0].set_title("image")
    axs[0].imshow(image_downscaled.detach().cpu().numpy(), cmap="gray")
    axs[1].set_title("2D reconstruction")
    axs[1].imshow(membrane_image_downscaled.detach().cpu().numpy(), cmap="gray")
    axs[2].set_title(f"image\n({int(100 * subtraction_factor)}% subtracted)")
    axs[2].imshow(subtracted_downscaled.detach().cpu().numpy(), cmap="gray")
    plt.tight_layout()
    return fig
