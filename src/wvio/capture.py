"""wvio.capture — consolidated capture pipeline (video/cube -> depth, velocity).

The perception layer used by the real-capture analyses and the field tools, in ONE tested place
instead of re-implemented per script. Pure-numpy/scipy functions (deglint, hover_depth, fuse_velocity)
need only the core deps; the video/optical-flow helpers lazily import opencv (`pip install -e ".[capture]"`).

All results are in the IMAGE frame (camera x,y). World-frame rotation + integration is the caller's job
(it needs telemetry: gimbal yaw + the one-time mount extrinsic).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math, subprocess, tempfile
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from .spectral import dispersion_ridge
from .wavefield import G

# validated defaults (see ASSUMPTIONS_AND_DATA.md): swell band, spatial band, deglint percentile
HOVER_T_BAND = (2.3, 4.5)      # s — narrow swell band for a stationary hover
CRAWL_T_BAND = (1.6, 8.0)      # s — wider, to keep the Doppler-shifted shell
LAMBDA_BAND = (6.0, 30.0)      # m — reject sub-metre glint sparkle and long-lambda gradients
DEGLINT_PCT = 98.0
DJI_FOCAL_4K = 2560.0          # px focal at 3840 px width (DJI Mini 5 Pro); focal = DJI_FOCAL_4K*W/3840


def gsd_for(alt_m: float, width_px: int) -> float:
    """Ground sample distance (m/px) for a nadir flat-water view at this altitude + extraction width."""
    return alt_m / (DJI_FOCAL_4K * width_px / 3840.0)


def extract_cube(video, t0, dur, width=640, fps=4.0):
    """ffmpeg -> (T, H, W, 3) float RGB cube. Needs ffmpeg on PATH + opencv-python."""
    try:
        import cv2
    except ImportError as e:
        raise ImportError("opencv-python needed for video I/O: pip install -e '.[capture]'") from e
    d = Path(tempfile.mkdtemp())
    subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t0), "-i", str(video), "-t", str(dur),
                    "-vf", f"fps={fps},scale={width}:-1", str(d / "f%05d.png")], check=True)
    fr = sorted(d.glob("f*.png"))
    try:
        cube = np.stack([cv2.imread(str(f))[:, :, ::-1].astype(float) for f in fr])
    finally:
        for f in fr: f.unlink()
        d.rmdir()
    return cube


def _gray(cube):
    """(T,H,W,3) RGB -> min-of-RGB gray; (T,H,W) already-gray -> as float. Accepts real captures + sim."""
    c = np.asarray(cube, float)
    return c.min(3) if c.ndim == 4 else c


def deglint(cube):
    """min-of-RGB + per-frame transient-bright (top (100-DEGLINT_PCT)%) -> temporal median. -> (T,H,W)."""
    g = _gray(cube)
    tm = np.median(g, 0)
    for t in range(g.shape[0]):
        m = g[t] > np.percentile(g[t], DEGLINT_PCT)
        g[t][m] = tm[m]
    return g


def _bandpass(g, fps, t_band):
    T = g.shape[0]
    F = np.fft.fft(g - g.mean(0), axis=0)
    om = np.abs(2 * np.pi * np.fft.fftfreq(T, 1.0 / fps))
    F[(om < 2 * np.pi / t_band[1]) | (om > 2 * np.pi / t_band[0])] = 0
    return np.fft.ifft(F, axis=0).real


def _ridge(bp, gsd, fps, lam_band):
    return dispersion_ridge(bp, 1.0 / fps,
                            kappa_lo=2 * np.pi / lam_band[1] * gsd,
                            kappa_hi=2 * np.pi / lam_band[0] * gsd)


def _swell_axis(k, w):
    ang = np.arctan2(k[:, 1], k[:, 0])
    return 0.5 * np.arctan2(np.average(np.sin(2 * ang), weights=w),
                            np.average(np.cos(2 * ang), weights=w))


def _fit_d_u(k, km, omg, w, u_prior=0.5, u_bound=2.0, d0=1.5):
    """Robust joint fit of (depth, U) from a dispersion ridge:  omega = sqrt(g k tanh(kd)) + k.U.
    u_prior > 0 favours small |U| (use on HOVERS, where U is a small current; set 0 on crawls, where
    U absorbs the platform Doppler and is large). Returns (depth, Ux, Uy)."""
    swt = np.sqrt(w / w.mean())
    def resid(p):
        d, ux, uy = p
        data = (np.sqrt(G * km * np.tanh(km * d)) + k[:, 0] * ux + k[:, 1] * uy - omg) * swt
        return np.concatenate([data, u_prior * np.array([ux, uy])])
    sol = least_squares(resid, [d0, 0.0, 0.0], bounds=([0.3, -u_bound, -u_bound], [20.0, u_bound, u_bound]),
                        loss="soft_l1", f_scale=0.1)
    return sol.x


def _fit_u_at_depth(k, km, omg, w, d):
    """Linear LS for U at a KNOWN depth: omega - sigma(k,d) = k.U. Clean (no depth<->U trade)."""
    rhs = (omg - np.sqrt(G * km * np.tanh(km * d)))
    sw = np.sqrt(w)
    U, *_ = np.linalg.lstsq(k * sw[:, None], rhs * sw, rcond=None)
    return U


@dataclass
class DepthResult:
    depth: float            # m, celerity inversion (primary)
    off_shell: float        # fit residual / shell magnitude (lower = cleaner)
    kh: float               # observability: ~0.3-1.5 good, >1.8 near-deep
    period: float           # s
    wavelength: float       # m
    swell_axis: float       # rad, image frame (axis mod pi)
    current: np.ndarray     # U_rel image-frame vector (= -current at a hover)
    n_ridge: int

    @property
    def observable(self) -> bool:
        return self.off_shell < 0.35 and 0.25 <= self.kh <= 1.8

    def verdict(self) -> str:
        if self.off_shell > 0.35:
            return "WEAK/GLINTY shell"
        if self.kh > 1.8:
            return f"NEAR-DEEP (kh={self.kh:.1f}) — depth not observable"
        if self.kh < 0.25:
            return f"VERY SHALLOW (kh={self.kh:.2f}) — watch breaking"
        return f"OBSERVABLE -> {self.depth:.1f} m (kh={self.kh:.2f})"


def hover_depth(cube, gsd, fps=4.0, deglint_on=True, t_band=HOVER_T_BAND, lam_band=LAMBDA_BAND) -> DepthResult:
    """Dispersion depth from a hover cube. Deglint -> 3-D FFT band-pass -> ridge -> shell fit -> celerity inv."""
    g = deglint(cube) if deglint_on else _gray(cube)
    bp = _bandpass(g, fps, t_band)
    r = _ridge(bp, gsd, fps, lam_band)
    w = r["weight"]; k = r["kappa"] / gsd; km = np.hypot(*k.T); omg = r["omega"]
    depth, ux, uy = _fit_d_u(k, km, omg, w, u_prior=0.5, u_bound=2.0)   # hover: current is small -> prior on
    pred = np.sqrt(G * km * np.tanh(km * depth)) + k[:, 0] * ux + k[:, 1] * uy
    off = float(np.average(np.abs(omg - pred), weights=w) / np.average(np.abs(omg), weights=w))
    kbar = float(np.average(km, weights=w))
    return DepthResult(depth=float(depth), off_shell=off, kh=kbar * float(depth),
                       period=float(2 * np.pi / np.average(omg, weights=w)),
                       wavelength=float(2 * np.pi / kbar), swell_axis=_swell_axis(k, w),
                       current=np.array([ux, uy]), n_ridge=len(w))


def optical_flow(cube, gsd, fps, max_features=400):
    """KLT median flow -> m/s image-frame vector (surface motion as seen by the camera). Needs opencv."""
    import cv2
    c = np.asarray(cube, float)
    if c.ndim == 4:                                   # real RGB (0-255) -> luminance
        gray = [cv2.cvtColor(fr.astype(np.uint8), cv2.COLOR_RGB2GRAY) for fr in c]
    else:                                             # sim/grayscale -> normalise to 0-255
        n = (c - c.min()) / (np.ptp(c) + 1e-9) * 255
        gray = [fr.astype(np.uint8) for fr in n]
    fl = []
    for i in range(len(gray) - 1):
        p0 = cv2.goodFeaturesToTrack(gray[i], max_features, 0.01, 9)
        if p0 is None:
            continue
        p1, stt, _ = cv2.calcOpticalFlowPyrLK(gray[i], gray[i + 1], p0, None, winSize=(21, 21), maxLevel=3)
        good = stt.ravel() == 1
        if good.sum() > 20:
            fl.append(np.median((p1 - p0).reshape(-1, 2)[good], 0))
    return (np.median(np.array(fl), 0) * gsd * fps) if fl else np.zeros(2)


def of_ego(crawl_cube, gsd, fps, wave_flow=None, max_features=400):
    """Platform ego-velocity from optical flow (image frame): ego = -(OF_crawl - wave_flow).
    Pass wave_flow = optical_flow(before-hover cube) to remove the persistent wave-phase motion."""
    of = optical_flow(crawl_cube, gsd, fps, max_features)
    wf = np.zeros(2) if wave_flow is None else np.asarray(wave_flow)
    return -(of - wf)


def doppler_ego(crawl_cube, hover_cube, gsd, fps=5.0, t_band=CRAWL_T_BAND, lam_band=LAMBDA_BAND, depth=None):
    """Wave-Doppler platform ego-velocity (image frame). From omega = sqrt(gk tanh kd) + k.U:
    a hover gives U = U_current; a crawl gives U = U_current - U_platform; so ego = U_hover - U_crawl.
    Returns (ego, swell_axis). Reliable along-swell; weak cross-swell (fuse with optical flow).
    depth: bathy depth (m) at the crawl's location; pass it for crawls that moved across the slope (a single
    crawl can't resolve depth vs the platform Doppler). None -> use the hover's depth (short/same-depth crawls)."""
    def ridge(cube, band):
        g = deglint(cube); bp = _bandpass(g, fps, band); r = _ridge(bp, gsd, fps, lam_band)
        k = r["kappa"] / gsd
        return k, np.hypot(*k.T), r["omega"], r["weight"]
    kh, kmh, omgh, wh = ridge(hover_cube, HOVER_T_BAND)
    d_h, uxh, uyh = _fit_d_u(kh, kmh, omgh, wh, u_prior=0.5, u_bound=2.0)   # hover: depth + current
    kc, kmc, omgc, wc = ridge(crawl_cube, t_band)
    # A single crawl CANNOT jointly resolve depth vs a large platform Doppler (the depth<->current degeneracy:
    # the joint fit trades the platform velocity for a wrong depth). So fix the depth and do a clean linear U
    # fit. Pass `depth` = the bathy depth at the crawl's location when the crawl has moved across the slope
    # (that's what fixes the blowup); otherwise fall back to the hover's depth (valid only when the crawl stays
    # near the hover, e.g. a short crawl or the synthetic same-depth case).
    U_c = _fit_u_at_depth(kc, kmc, omgc, wc, d_h if depth is None else depth)
    return (np.array([uxh, uyh]) - U_c), _swell_axis(kc, wc)


def fuse_velocity(of_ego_vec, doppler_ego_vec, swell_axis):
    """Best fusion (image or world frame, consistent): along-swell component from wave-Doppler (always
    observable), cross-swell from optical flow (geometry-free).

    FRAME WARNING (2026-07-02, measured on real Long Point crawls): on REAL video cubes the
    dispersion-ridge frame (doppler_ego, swell_axis) is the optical-flow PIXEL frame rotated +90 deg
    -- pass doppler_ego and swell_axis through ridge_to_pixel()/(swell_axis - pi/2) before fusing
    with of_ego. On wvio.sim cubes the two frames coincide (why the synthetic tests can't catch the
    mix). Root-cause alignment of sim vs extract_cube axes is an open refactor; see
    captures/2026-07-02_frame_audit.md."""
    sh = np.array([np.cos(swell_axis), np.sin(swell_axis)]); sh = sh / np.linalg.norm(sh)
    sp = np.array([-sh[1], sh[0]])
    return (doppler_ego_vec @ sh) * sh + (of_ego_vec @ sp) * sp


def ridge_to_world(v, osd_yaw_rad):
    """Rotate a ridge-frame vector (doppler_ego output, _fit_d_u U) into the WORLD (ENU) frame for
    REAL video cubes: v_world = -R(4deg - psi_osd) @ v_ridge (equivalently world->ridge is the pure
    rotation -R(psi_osd - 4deg)). UNIFIED 2026-07-02 late: brute-forced under the FINAL yaw
    convention on F3 (rescur 0.12) and F1 (0.47). SUPERSEDES the earlier rot(-90) 'ridge_to_pixel'
    (that map was co-calibrated with a retracted yaw convention -- a reflection and a rotation can
    agree on a single flight when v lies near the reflection axis). On sim cubes frames coincide --
    do NOT apply to sim-rendered cubes. See captures/2026-07-02_crawl_transect_v2.md late addenda."""
    a = math.radians(4.0) - osd_yaw_rad
    c, s = math.cos(a), math.sin(a)
    return -np.array([[c, -s], [s, c]]) @ np.asarray(v, float)


def ridge_to_pixel(v):
    """DEPRECATED (2026-07-02 late): the rot(-90) ridge->pixel map was co-calibrated with a
    RETRACTED yaw convention and does not generalize across flights. Use ridge_to_world() with the
    aircraft OSD yaw instead."""
    import warnings
    warnings.warn("ridge_to_pixel is deprecated; use ridge_to_world(v, osd_yaw_rad)", DeprecationWarning)
    return np.array([[0.0, 1.0], [-1.0, 0.0]]) @ np.asarray(v, float)
