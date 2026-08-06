"""Analytic linear gravity-wave field + camera-frame measurement model (Phase 0).

No rendering. We define the truth (wavenumbers, frequencies, amplitudes,
directions) and compute exactly what a nadir camera registered to a moving
platform observes in the (wavenumber, frequency) domain.

Conventions (pin these down once; sign bugs are the classic silent failure):
- Earth-fixed horizontal frame, x east, y north, metres.
- A wave component i propagates along unit vector  d_i = [cos th_i, sin th_i].
  Its wavenumber vector is  k_i = k_i * d_i  (rad/m), k_i = |k_i| > 0.
- Intrinsic angular frequency (still water, no current):
      sigma_i = sqrt(g * k_i * tanh(k_i * depth))           (rad/s, > 0)
- A *uniform surface current* U_c advects the pattern, so in the earth frame
  the component's apparent frequency is  sigma_i + k_i . U_c.
- A camera whose image is registered to the platform (horizontal velocity U_p)
  observes, in the image/time domain, an angular frequency
      omega_obs_i = sigma_i  -  k_i . (U_p - U_c)
                  = sigma_i  -  k_i . U_rel,     U_rel := U_p - U_c.
  Derivation: earth-frame crest speed along d_i is c_i + (U_c.d_i); the platform
  closes on it at U_p.d_i; observed temporal freq = k_i * (relative crest speed)
  = sigma_i + k_i.(U_c - U_p). (Verified against the ship "encounter frequency"
  omega_e = sigma - k.U and against X-band-radar current retrieval, which is the
  same relation solved for U_c from a *fixed* platform.)
- Monocular scale: a wave measured in the image has spatial wavenumber
      kappa_i = k_i * s        (rad/pixel),     s = ground sampling distance (m/pixel).
  Time is absolute (frame rate known) so omega_obs is in true rad/s. The estimator
  recovers s by forcing all components onto a single valid dispersion curve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

G = 9.80665  # m/s^2


def dispersion(k: np.ndarray, depth: float, g: float = G) -> np.ndarray:
    """Intrinsic angular frequency sigma = sqrt(g k tanh(k depth)).

    depth = np.inf gives the deep-water limit sigma = sqrt(g k).
    """
    k = np.asarray(k, dtype=float)
    if np.isinf(depth):
        return np.sqrt(g * k)
    return np.sqrt(g * k * np.tanh(k * depth))


@dataclass(frozen=True)
class WaveComponent:
    amplitude: float       # m (unused in Phase-0 measurement model; kept for Layer B render)
    k: float               # wavenumber magnitude, rad/m  (>0)
    theta: float           # propagation direction, rad (earth frame)
    phase: float = 0.0     # rad

    def k_vec(self) -> np.ndarray:
        return self.k * np.array([np.cos(self.theta), np.sin(self.theta)])


def sea_state(kind: str, *, depth: float = np.inf, rng: np.random.Generator | None = None,
              n: int | None = None) -> list[WaveComponent]:
    """A few canonical sea states for Phase-0 observability tests.

    - 'single_swell' : one narrow train (1-D Doppler observability only).
    - 'crossing'     : two trains ~70 deg apart (full 2-D velocity observability).
    - 'spread'       : n components spanning a range of k AND directions
                       (good for scale + 2-D velocity; default n=6).
    Wavenumbers are chosen so phase speed c = sigma/k varies across components
    (needed for scale to be observable -- avoid the deep-water all-same-c trap by
    spanning a real k range).
    """
    if kind == "single_swell":
        # one dominant swell, lambda ~ 80 m  -> k ~ 0.0785
        return [WaveComponent(1.0, 2 * np.pi / 80.0, np.deg2rad(20.0), 0.3)]
    if kind == "crossing":
        return [
            WaveComponent(1.0, 2 * np.pi / 80.0, np.deg2rad(20.0), 0.3),
            WaveComponent(0.7, 2 * np.pi / 55.0, np.deg2rad(95.0), 1.1),
        ]
    if kind == "spread":
        rng = rng or np.random.default_rng(0)
        n = n or 6
        lambdas = np.linspace(40.0, 110.0, n)          # 40..110 m -> real k spread
        thetas = np.deg2rad(np.linspace(-40.0, 60.0, n))
        return [WaveComponent(float(rng.uniform(0.5, 1.0)), float(2 * np.pi / L),
                              float(th), float(rng.uniform(0, 2 * np.pi)))
                for L, th in zip(lambdas, thetas)]
    raise ValueError(f"unknown sea state {kind!r}")


def observe(components: list[WaveComponent], U_rel: np.ndarray, scale: float,
            depth: float = np.inf, g: float = G,
            noise: dict | None = None,
            rng: np.random.Generator | None = None) -> dict:
    """Forward measurement model: what the platform-registered image spectrum yields.

    Returns dict with:
      kappa : (M, 2) image wavenumber vectors, rad/pixel   ( = k_vec * scale )
      omega : (M,)   observed angular frequencies, rad/s    ( = sigma - k.U_rel )
      k_mag : (M,)   true physical |k| (for diagnostics only)
      sigma : (M,)   true intrinsic freq (diagnostics only)
    `noise`, if given, may contain 'omega' (std, rad/s) and 'kappa_rel' (relative
    std on each kappa component) to emulate finite spectral-peak precision.
    """
    U_rel = np.asarray(U_rel, dtype=float)
    kvecs = np.array([c.k_vec() for c in components])          # (M,2) rad/m
    kmag = np.array([c.k for c in components])                 # (M,)
    sigma = dispersion(kmag, depth, g)                         # (M,)
    omega = sigma - kvecs @ U_rel                              # (M,)
    kappa = kvecs * scale                                      # (M,2) rad/pixel

    if noise:
        rng = rng or np.random.default_rng(0)
        if noise.get("omega"):
            omega = omega + rng.normal(0.0, noise["omega"], size=omega.shape)
        if noise.get("kappa_rel"):
            kappa = kappa * (1.0 + rng.normal(0.0, noise["kappa_rel"], size=kappa.shape))
    return {"kappa": kappa, "omega": omega, "k_mag": kmag, "sigma": sigma}
