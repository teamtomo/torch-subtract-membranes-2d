"""Model membranes in cryo-EM images"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-trace-membranes-2d")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Alister Burt"
__email__ = "burt.alister@gene.com"

from torch_trace_membranes_2d.trace_membranes import trace_membranes
from torch_trace_membranes_2d.subtract_membranes import subtract_membranes
from torch_trace_membranes_2d.render_membranes import render_membrane_image

__all__ = [
    "trace_membranes",
    "subtract_membranes"
]
