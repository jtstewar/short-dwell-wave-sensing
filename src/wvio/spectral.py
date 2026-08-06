"""3-D FFT spectral peak extraction (Phase 1).

Take a rendered image stack (T, H, W) and recover the (kappa, omega) of the
dominant wave components, in the SAME convention the estimator expects:
  kappa : (M,2) image wavenumber vectors [kx, ky] in rad/pixel (propagation dir)
  omega : (M,)  observed angular frequency in rad/s (> 0)
so that omega ~= sigma(|k|) - k . U_rel  with k = kappa / scale.

Sign convention (numpy forward FFT exp(-2pi i f n)): a real wave
cos(k.r - w t) puts energy at (Omega,Kx,Ky)=(-w,+kx,+ky) and (+w,-kx,-ky).
We select the Omega>0 half and report (omega=Omega, kappa=-K) so kappa points
along propagation. (Verified by tests/test_phase1.py::test_peak_sign.)
"""
from __future__ import annotations

import numpy as np


def _power_spectrum(images: np.ndarray, dt: float):
    """3-D power spectrum of an image stack, with the standard pre-processing.

    Per-pixel temporal demean (kills static/DC), a separable Hann window in
    (t, y, x), then |FFT|^2. Returns (P, omega_ax, kx_ax, ky_ax) with omega in
    rad/s and kx/ky in rad/pixel. Shared by extract_peaks and dispersion_ridge.
    """
    T, H, W = images.shape
    imgs = images - images.mean(axis=0, keepdims=True)         # kill static/DC per pixel
    win = (np.hanning(T)[:, None, None] * np.hanning(H)[None, :, None]
           * np.hanning(W)[None, None, :])
    F = np.fft.fftn(imgs * win)
    P = np.abs(F) ** 2
    omega_ax = 2 * np.pi * np.fft.fftfreq(T, d=dt)             # rad/s (axis 0)
    kx_ax = 2 * np.pi * np.fft.fftfreq(W, d=1.0)               # rad/pixel (axis 2)
    ky_ax = 2 * np.pi * np.fft.fftfreq(H, d=1.0)               # rad/pixel (axis 1)
    return P, omega_ax, kx_ax, ky_ax


def _centroid_offset(P: np.ndarray, idx: tuple, axis: int, half: int = 2) -> float:
    """Power-weighted centroid sub-bin offset along `axis` over +-half bins (wrap-safe).

    More robust than 3-point parabolic for Hann-broadened peaks (uses the whole
    mainlobe, not just immediate neighbours)."""
    N = P.shape[axis]
    base = idx[axis]
    offs = np.arange(-half, half + 1)
    vals = np.array([P[tuple(idx[:axis]) + ((base + o) % N,) + tuple(idx[axis + 1:])]
                     for o in offs])
    return float((offs * vals).sum() / vals.sum())


def extract_peaks(images: np.ndarray, dt: float, n_peaks: int = 6,
                  dc_guard: int = 2, min_period_bins: int = 1) -> dict:
    """3-D FFT -> top-n dominant peaks (Omega>0 half), with parabolic sub-bin refine.

    Returns {kappa:(M,2) rad/pixel, omega:(M,) rad/s, power:(M,)}.
    """
    T, H, W = images.shape
    P, omega_ax, kx_ax, ky_ax = _power_spectrum(images, dt)

    # restrict to Omega>0 half; null only the SPATIAL-DC point (|kx|&|ky| both ~0)
    # -- NOT whole kx/ky slabs, which would kill near-axis-propagating waves.
    mask = np.zeros_like(P, dtype=bool)
    pos_t = np.where(omega_ax > 0)[0]
    mask[pos_t, :, :] = True
    g = dc_guard
    near0 = lambda N: np.r_[np.arange(g + 1), np.arange(N - g, N)]   # bins within g of 0
    spatial_dc = np.zeros((H, W), dtype=bool)
    spatial_dc[np.ix_(near0(H), near0(W))] = True                   # kx~0 AND ky~0
    mask[:, spatial_dc] = False
    mask[:min_period_bins + 1, :, :] = False                        # low temporal freq
    Pm = np.where(mask, P, 0.0)

    peaks = []
    used = np.zeros_like(P, dtype=bool)
    for _ in range(n_peaks):
        idx = np.unravel_index(np.argmax(np.where(used, 0.0, Pm)), P.shape)
        ti, yi, xi = idx
        if Pm[idx] <= 0:
            break
        # power-weighted centroid sub-bin refinement per axis (wrap-safe)
        dti = _centroid_offset(P, idx, axis=0)
        dyi = _centroid_offset(P, idx, axis=1)
        dxi = _centroid_offset(P, idx, axis=2)
        omega = 2 * np.pi * ((ti + dti) if (ti + dti) <= T / 2 else (ti + dti - T)) / (T * dt)
        kx = 2 * np.pi * ((xi + dxi) if (xi + dxi) <= W / 2 else (xi + dxi - W)) / W
        ky = 2 * np.pi * ((yi + dyi) if (yi + dyi) <= H / 2 else (yi + dyi - H)) / H
        # Omega>0 half: propagation-dir wavenumber is -K (see module docstring)
        peaks.append((np.array([-kx, -ky]), float(omega), float(P[idx])))
        # suppress a neighbourhood (this peak + its conjugate) before next search
        for (aa, bb, cc) in [(ti, yi, xi), ((T - ti) % T, (H - yi) % H, (W - xi) % W)]:
            used[max(aa - g, 0):aa + g + 1, max(bb - g, 0):bb + g + 1,
                 max(cc - g, 0):cc + g + 1] = True

    peaks.sort(key=lambda p: -p[2])
    return {"kappa": np.array([p[0] for p in peaks]),
            "omega": np.array([p[1] for p in peaks]),
            "power": np.array([p[2] for p in peaks])}


def dispersion_ridge(images: np.ndarray, dt: float, *, dc_guard: int = 2,
                     min_period_bins: int = 1, kappa_lo: float | None = None,
                     kappa_hi: float | None = None, min_rel_power: float = 5e-3,
                     ridge_half: int = 2) -> dict:
    """Energy-weighted dispersion ridge over the WHOLE spectrum (Phase 1b).

    Where extract_peaks() takes the top-N *global* maxima, this returns one
    (kappa, omega) sample for EVERY in-band spatial wavenumber that carries
    energy, weighted by that energy -- the cBathy / X-band-radar "fit the shell"
    view. Robust on a broadband sea, where no handful of discrete peaks captures
    the directional + scale spread (the top-N peak count then becomes a fragile
    knob; see kill_checks.py / RESULTS_phase1b.md).

    For each spatial wavenumber bin K=(kx,ky) on the Omega>0 half we read the
    energy ridge in frequency -- argmax over Omega, power-weighted-centroid
    refined over +-ridge_half bins -- as omega_obs(K), and weight = total in-band
    energy at K. The propagation-direction wavenumber is kappa = -K (same
    Omega>0 convention as extract_peaks; see module docstring), so the result
    drops straight into fit_window(..., weights=...).

    Band: |kappa| in [kappa_lo, kappa_hi] rad/pixel. Defaults exclude the
    spatial-DC smear (dc_guard bins) and waves shorter than ~6 px. A bin is kept
    only if its energy exceeds min_rel_power * (max in-band energy).

    Returns {kappa:(M,2) rad/pixel, omega:(M,) rad/s, weight:(M,)}.
    """
    empty = {"kappa": np.empty((0, 2)), "omega": np.empty(0), "weight": np.empty(0)}
    P, omega_ax, kx_ax, ky_ax = _power_spectrum(images, dt)
    T, H, W = images.shape

    pos = np.where(omega_ax > 0)[0]
    pos = pos[pos >= min_period_bins + 1]                  # drop slow-drift/DC temporal bins
    if pos.size == 0:
        return empty
    Ppos = P[pos]                                          # (npos, H, W); pos is contiguous
    npos = pos.size
    d_omega = 2 * np.pi / (T * dt)

    # per spatial bin: total in-band energy (weight) + frequency ridge (argmax + centroid)
    weight = Ppos.sum(axis=0)                              # (H, W)
    ridge = Ppos.argmax(axis=0)                            # (H, W) index into pos
    num = np.zeros((H, W)); den = np.zeros((H, W))
    for o in range(-ridge_half, ridge_half + 1):
        j = ridge + o
        valid = (j >= 0) & (j <= npos - 1)                # SKIP out-of-range neighbours
        val = np.where(valid, np.take_along_axis(Ppos, np.clip(j, 0, npos - 1)[None],
                                                  axis=0)[0], 0.0)
        num += o * val; den += val                        # (clipping then masking, not counting)
    ridge_f = ridge + num / np.maximum(den, 1e-300)        # sub-bin ridge (pos-index units)
    omega_obs = (pos[0] + ridge_f) * d_omega               # rad/s (pos contiguous from pos[0])

    KX, KY = np.meshgrid(kx_ax, ky_ax)                     # (H, W) rad/pixel
    kmag = np.hypot(KX, KY)
    klo = kappa_lo if kappa_lo is not None else dc_guard * 2 * np.pi / max(W, H)
    khi = kappa_hi if kappa_hi is not None else np.pi / 3.0     # >= ~6 px wavelength
    band = (kmag >= klo) & (kmag <= khi)
    if not band.any():
        return empty
    keep = band & (weight > min_rel_power * weight[band].max())
    if not keep.any():
        return empty
    # propagation-dir wavenumber is -K on the Omega>0 half (see module docstring)
    kappa = np.stack([-KX[keep], -KY[keep]], axis=1)
    return {"kappa": kappa, "omega": omega_obs[keep], "weight": weight[keep]}
