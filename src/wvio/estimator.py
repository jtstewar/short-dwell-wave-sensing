"""Dispersion-Doppler estimator (Phase 0).

Two stages:

1. fit_window(): from one window's (kappa, omega) measurements recover the
   metric scale s (m/pixel) and the water-relative velocity U_rel = U_p - U_c,
   by nonlinear least squares on  omega = sqrt(g k tanh(k depth)) - k.U_rel
   with k = kappa / s. Scale is observable from the sqrt(k) curvature provided
   the components span a real range of |k|.

2. fuse_windows(): combine per-window U_rel with IMU velocity increments to try
   to recover absolute platform velocity U_p[w] and the (constant) current U_c.
   This is where the over-water identifiability structure lives:
     - wave gives   U_p[w] - U_c
     - IMU gives    U_p[w] - U_p[w-1]
   The common-mode shift (U_p += c, U_c += c) is invisible to BOTH => a rank-2
   nullspace (one per horizontal axis). It is removed only by an absolute
   velocity reference (a known-hover/GNSS epoch) supplied via `anchor`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .wavefield import G, dispersion


@dataclass
class WindowFit:
    scale: float           # m/pixel
    U_rel: np.ndarray      # (2,) m/s
    U_rel_cov: np.ndarray  # (2,2)
    scale_cond: float      # conditioning of the scale/velocity Jacobian
    residual_rms: float    # rad/s
    depth: float = np.inf  # m (estimated if fit_depth=True, else the input)


def fit_window(kappa: np.ndarray, omega: np.ndarray, depth: float = np.inf,
               g: float = G, scale0: float | None = None,
               fit_depth: bool = False, depth0: float = 20.0,
               scale_fixed: float | None = None,
               weights: np.ndarray | None = None,
               loss: str = "linear", f_scale: float = 1.0) -> WindowFit:
    """Fit metric scale s and water-relative velocity U_rel to (kappa, omega).

    fit_depth   : also estimate water depth d (Tier-2 joint scale<->depth; needs
                  wavenumbers spanning the tanh(kd) knee -- intermediate/shallow).
    scale_fixed : if given, scale is treated as a KNOWN input (Tier-0 oracle, or
                  a value bootstrapped from a stationary hover) and not estimated.
    weights     : per-measurement weights (e.g. spectral energy from a full-shell
                  fit, dispersion_ridge); residuals are scaled by sqrt(weight).
    loss,f_scale: passed to least_squares. The full-shell fit uses a robust loss
                  ('soft_l1') so off-shell spectral-leakage samples are
                  down-weighted; f_scale is the soft threshold in rad/s.
    """
    kappa = np.asarray(kappa, float)          # (M,2) rad/pixel
    omega = np.asarray(omega, float)          # (M,)
    kap_mag = np.linalg.norm(kappa, axis=1)   # (M,)

    if weights is None:
        sw = np.ones(omega.shape)
    else:
        w = np.asarray(weights, float)
        sw = np.sqrt(w / w.mean())            # normalise to mean 1 (keeps cost/cov scale sane)

    if scale0 is None:
        # deep-water, zero-current guess: omega^2 = g k = g kappa/s -> s = g kappa/omega^2
        with np.errstate(divide="ignore", invalid="ignore"):
            s_est = g * kap_mag / np.maximum(omega ** 2, 1e-9)
        scale0 = float(np.median(s_est[np.isfinite(s_est) & (s_est > 0)]))

    def unpack(p):
        i = 0
        s = scale_fixed if scale_fixed is not None else p[i]
        if scale_fixed is None:
            i += 1
        urx, ury = p[i], p[i + 1]; i += 2
        d = p[i] if fit_depth else depth
        return s, urx, ury, d

    def model_resid(p):                        # unweighted physical residual (rad/s)
        s, urx, ury, d = unpack(p)
        s = max(s, 1e-9)
        kvec = kappa / s                       # (M,2) rad/m
        kmag = kap_mag / s                      # (M,)
        sig = dispersion(kmag, d, g)
        model = sig - kvec @ np.array([urx, ury])
        return model - omega

    def resid(p):                              # weighted residual seen by the optimiser
        return model_resid(p) * sw

    p0, lo, hi = [], [], []
    if scale_fixed is None:
        p0.append(scale0); lo.append(1e-6); hi.append(np.inf)
    p0 += [0.0, 0.0]; lo += [-np.inf, -np.inf]; hi += [np.inf, np.inf]
    if fit_depth:
        p0.append(depth0); lo.append(0.3); hi.append(1e4)
    # x_scale='jac' auto-scales the very different parameter magnitudes
    # (scale ~0.08 m/px vs depth ~30 m vs velocity ~1 m/s) -- critical for the
    # joint scale<->depth fit's trust region to converge.
    sol = least_squares(resid, np.array(p0), method="trf", bounds=(lo, hi),
                        x_scale="jac", loss=loss, f_scale=f_scale, max_nfev=6000)
    s_hat, urx, ury, d_hat = unpack(sol.x)

    # covariance ~ sigma_res^2 * (J^T J)^-1 ; U_rel occupies the 2 params after scale
    ur0 = 0 if scale_fixed is not None else 1
    J = sol.jac
    dof = max(len(omega) - len(sol.x), 1)
    sig2 = float(2 * sol.cost / dof)
    JtJ = J.T @ J
    cond = float(np.linalg.cond(JtJ))
    try:
        cov = sig2 * np.linalg.inv(JtJ)
        urel_cov = cov[ur0:ur0 + 2, ur0:ur0 + 2]
    except np.linalg.LinAlgError:
        urel_cov = np.full((2, 2), np.inf)
    rms = float(np.sqrt(np.mean(model_resid(sol.x) ** 2)))   # unweighted, physical (rad/s)
    return WindowFit(float(s_hat), np.array([urx, ury]), urel_cov, cond, rms, float(d_hat))


def fit_shell(images: np.ndarray, dt: float, *, depth: float = np.inf, g: float = G,
              scale0: float | None = None, fit_depth: bool = False, depth0: float = 20.0,
              scale_fixed: float | None = None, f_scale: float = 0.08,
              **ridge_kw) -> WindowFit:
    """Phase 1b: full dispersion-shell fit straight from an image stack.

    Energy-weighted ridge over the WHOLE 3-D spectrum (spectral.dispersion_ridge)
    -> the weighted, robust single-window fit. Unlike extract_peaks + fit_window
    this has no top-N peak-count knob, so it stays stable on a broadband sea.
    `ridge_kw` is forwarded to dispersion_ridge (dc_guard, kappa_lo/hi, ...).
    """
    from .spectral import dispersion_ridge
    r = dispersion_ridge(images, dt, **ridge_kw)
    return fit_window(r["kappa"], r["omega"], depth=depth, g=g, scale0=scale0,
                      fit_depth=fit_depth, depth0=depth0, scale_fixed=scale_fixed,
                      weights=r["weight"], loss="soft_l1", f_scale=f_scale)


@dataclass
class FusionResult:
    U_p: np.ndarray            # (W,2) estimated platform velocity
    U_c: np.ndarray            # (2,)  estimated current
    U_rel_in: np.ndarray       # (W,2) the per-window U_rel measurements used
    nullspace_dim: int         # 0 if fully observable
    smallest_singular: float   # of the (normalised) design matrix
    rank_deficient: bool


def fuse_windows(U_rel_meas: np.ndarray, dV_meas: np.ndarray,
                 anchor: tuple[int, np.ndarray] | None = None,
                 w_wave: float = 1.0, w_imu: float = 5.0,
                 w_anchor: float = 1e3) -> FusionResult:
    """Least-squares fuse wave U_rel + IMU increments (+ optional anchor).

    U_rel_meas : (W,2)   per-window water-relative velocity (wave-Doppler).
    dV_meas    : (W,2)   IMU increments, row 0 ignored (=0 reference).
    anchor     : (w*, value) a KNOWN absolute platform velocity at window w*
                 (e.g. (0, [0,0]) for a verified hover, or a GNSS velocity).
                 This is the only thing that removes the common-mode nullspace.

    Unknowns x = [U_p[0]..U_p[W-1], U_c]  (length 2W+2).
    """
    U_rel_meas = np.asarray(U_rel_meas, float)
    dV_meas = np.asarray(dV_meas, float)
    W = U_rel_meas.shape[0]
    nx = 2 * W + 2

    def up(w):  # index slice for U_p[w]
        return slice(2 * w, 2 * w + 2)
    uc = slice(2 * W, 2 * W + 2)

    rows, bvals = [], []

    def add(coef_map, rhs, weight):
        for axis in range(2):
            r = np.zeros(nx)
            for idx, val in coef_map.items():
                r[idx + axis if isinstance(idx, int) else (idx.start + axis)] = val
            rows.append(weight * r)
            bvals.append(weight * rhs[axis])

    # wave: U_p[w] - U_c = U_rel[w]
    for w in range(W):
        add({up(w): 1.0, uc: -1.0}, U_rel_meas[w], w_wave)
    # imu: U_p[w] - U_p[w-1] = dV[w]
    for w in range(1, W):
        add({up(w): 1.0, up(w - 1): -1.0}, dV_meas[w], w_imu)
    # anchor: U_p[w*] = value
    if anchor is not None:
        wstar, val = anchor
        add({up(wstar): 1.0}, np.asarray(val, float), w_anchor)

    A = np.array(rows)
    b = np.array(bvals)

    # observability from the UNWEIGHTED structural matrix (wave+imu only, no anchor)
    struct = []
    for w in range(W):
        for axis in range(2):
            r = np.zeros(nx); r[up(w).start + axis] = 1; r[uc.start + axis] = -1; struct.append(r)
    for w in range(1, W):
        for axis in range(2):
            r = np.zeros(nx); r[up(w).start + axis] = 1; r[up(w - 1).start + axis] = -1; struct.append(r)
    sv = np.linalg.svd(np.array(struct), compute_uv=False)
    tol = 1e-9 * sv.max()
    null_dim = int(np.sum(sv < tol))
    smallest = float(sv.min())

    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    return FusionResult(
        U_p=x[: 2 * W].reshape(W, 2),
        U_c=x[2 * W:],
        U_rel_in=U_rel_meas,
        nullspace_dim=null_dim,
        smallest_singular=smallest,
        rank_deficient=(anchor is None and null_dim > 0),
    )
