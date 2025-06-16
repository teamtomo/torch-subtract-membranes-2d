import os

IS_DEBUG = 'TORCH_SUBTRACT_MEMBRANES_2D_DEBUG' in os.environ


def set_matplotlib_resolution():
    import matplotlib as mpl
    mpl.rcParams['figure.dpi'] = 300
