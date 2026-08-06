"""Dispersion-likelihood score: normalized spectral power along the Doppler-shifted
shell omega = sigma(k,d) + k . U. The full-spectrum-likelihood column of Table 1.

Sign convention (do not re-derive): a camera moving at +v sees the wave pattern
at -v: omega_obs = sigma(k,d) + k . U with U = (texture/current frame) - v.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wvio.spectral import _power_spectrum
from wvio.wavefield import G


def spectrum_tables(cube: np.ndarray, fps: float, gsd: float,
                    lam_band=(6.0, 30.0)) -> dict:
    """Precompute the omega>0 spectrum + k-grid tables the score probes (once per cube)."""
    P, om_ax, kx_ax, ky_ax = _power_spectrum(cube.astype(float), 1.0 / fps)
    pos = om_ax > 0
    om = om_ax[pos]
    order = np.argsort(om)
    om_s = om[order]
    Pp = P[pos][order]
    KX, KY = np.meshgrid(kx_ax, ky_ax)
    km = np.hypot(KX, KY) / gsd
    mask = (2 * np.pi / lam_band[1] <= km) & (km <= 2 * np.pi / lam_band[0])
    Pk = Pp[:, mask]
    return dict(om=om_s, P=Pk / (Pk.sum(0) + 1e-12),
                kx=-KX[mask] / gsd, ky=-KY[mask] / gsd, km=km[mask])


def score_U(tab: dict, U: np.ndarray, depth: float) -> float:
    """Normalized spectral power on the shell omega = sigma(k, d) + k . U."""
    sig = np.sqrt(G * tab["km"] * np.tanh(tab["km"] * depth))
    om_pred = sig + tab["kx"] * U[0] + tab["ky"] * U[1]
    ok = (om_pred > tab["om"][0]) & (om_pred < tab["om"][-1])
    j = np.searchsorted(tab["om"], om_pred[ok]) - 1
    f = (om_pred[ok] - tab["om"][j]) / (tab["om"][j + 1] - tab["om"][j])
    cols = np.arange(tab["P"].shape[1])[ok]
    return float((tab["P"][j, cols] * (1 - f) + tab["P"][j + 1, cols] * f).sum())


def argmax_U(tab: dict, depth: float, vmax: float = 4.0) -> tuple[np.ndarray, float]:
    """Coarse-to-fine grid argmax (classical solver; also the eval reference)."""
    best, bs = np.zeros(2), -1.0
    for span, npts in ((vmax, int(vmax * 8) + 9), (0.5, 11), (0.12, 11)):
        vs = np.linspace(-span, span, npts)
        b0 = best.copy()
        for dx in vs:
            for dy in vs:
                U = b0 + (dx, dy)
                s = score_U(tab, U, depth)
                if s > bs:
                    bs, best = s, U
    return best, bs
