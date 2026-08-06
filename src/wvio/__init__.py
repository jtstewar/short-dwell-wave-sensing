"""wvio — short-dwell wave sensing, measurement layer.

Platform velocity, water depth and swell orientation from the wave dispersion
surface seen in a short nadir video window. numpy + scipy only; video
extraction needs ffmpeg on PATH + opencv-python.
"""
from .wavefield import G, WaveComponent, dispersion, observe, sea_state
from .estimator import WindowFit, FusionResult, fit_window, fit_shell, fuse_windows
from .spectral import extract_peaks, dispersion_ridge

__all__ = [
    "G", "WaveComponent", "dispersion", "observe", "sea_state",
    "WindowFit", "FusionResult", "fit_window", "fit_shell", "fuse_windows",
    "extract_peaks", "dispersion_ridge",
]
