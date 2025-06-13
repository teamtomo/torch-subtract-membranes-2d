"""Model membranes in cryo-EM images"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-subtract-membranes-2d")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Alister Burt"
__email__ = "burt.alister@gene.com"

from torch_subtract_membranes_2d.model_membranes import model_membranes
from torch_subtract_membranes_2d.subtract_membranes import subtract_membranes
from torch_subtract_membranes_2d.render_membranes import render_membrane_image
from torch_subtract_membranes_2d.utils.serialization_utils import save_membranes, load_membranes

__all__ = [
    "model_membranes",
    "subtract_membranes",
    "render_membrane_image",
    "load_membranes",
    "save_membranes",
]
