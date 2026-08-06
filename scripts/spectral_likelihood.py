"""Full-spectrum likelihood prototype v5 (task #11) — the statistical parent model.

Rebuild recipe (NEXT_STEPS / imu-era addenda): build DIRECTLY on the known-good
_power_spectrum machinery (spectral.py — axes and sign convention verified by unit test),
add a normalized-scalar-product (NSP) style score along the Doppler-shifted shell, and
validate stepwise: (0) unit equivalence vs the ridge-LS fit on the SAME window,
(1) hover (truth ~ current ~ 0), (2) crawl windows vs GPS velocity.

Score (single component, NSP family — cf. Smeltzer & Ellingsen 2022):
    L(v) = sum_K  P( omega_pred(K; v, d), K ) / sum_K sum_omega P(omega, K)
with omega_pred = sigma(|k|, d) + k . v, k = -K/gsd (propagation vector, rad/m),
probed on the omega>0 half by linear interpolation along the omega axis.

Usage: spectral_likelihood.py KEY TELG SYNC T0 DUR DEPTH [VGRID]
  e.g.  spectral_likelihood.py 0004 "*16-40-11*" 15 410 20 2.6      (F3 hover, d from hover fit)
"""
import csv, glob, math, os, re, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wvio import capture as cap
from wvio.spectral import _power_spectrum
from wvio.wavefield import G

D = Path(os.environ.get("WVIO_FLIGHTS", "data/flights"))
KEY, TELG, OFF = sys.argv[1], sys.argv[2], float(sys.argv[3])
T0, DUR, DEPTH = float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])
VMAX = float(sys.argv[7]) if len(sys.argv) > 7 else 3.0
C = math.radians(4.0)
def rot(a): return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])

mp4 = next(D.glob(f"*_{KEY}_D.MP4"))
srt = next(D.glob(f"*_{KEY}_D.SRT")).read_text(errors="replace")
tc = re.findall(r"(\d\d):(\d\d):(\d\d),(\d\d\d) -->", srt)
ra = np.array(re.findall(r"\[rel_alt: ([\d.\-]+)", srt), float)
sts = np.array([int(a) * 3600 + int(b) * 60 + int(c) + int(dd) / 1000 for a, b, c, dd in tc[:len(ra)]])
ALT = float(np.interp(T0 + DUR / 2, sts, ra[:len(sts)]))
GSD = cap.gsd_for(ALT, 640)

tel = list(csv.DictReader(open(glob.glob(f"captures/telemetry_0701/{TELG}.csv")[0])))
def col(k):
    o = []
    for x in tel:
        try: o.append(float(x[k]))
        except Exception: o.append(np.nan)
    return np.array(o)
ft, oy = col("OSD.flyTime"), col("OSD.yaw")
tla, tlo = col("OSD.latitude"), col("OSD.longitude")
mo = np.isfinite(oy); mv = (tla != 0) & np.isfinite(tla)
psi = float(np.interp(T0 + DUR / 2 + OFF, ft[mo], np.unwrap(np.radians(oy[mo]))))
# GPS truth velocity (world), central difference over the window, -> ridge frame (unified map)
tq = np.array([T0 + OFF, T0 + DUR + OFF])
la2 = np.interp(tq, ft[mv], tla[mv]); lo2 = np.interp(tq, ft[mv], tlo[mv])
vN = (la2[1] - la2[0]) * 111320.0 / DUR
vE = (lo2[1] - lo2[0]) * 111320.0 * math.cos(math.radians(la2.mean())) / DUR
v_true_w = np.array([vE, vN])
# the Doppler-measured U is the WATER-PATTERN velocity in the ridge frame = c - v (the camera
# moving at +v sees the pattern move at -v): expected U = -(v_ridge) + c_ridge. The tracker's
# world map A = -rot(C-psi) carries this negation (A@U = v_w - c_w). Truth here: U_exp = -v_r.
v_true_r = rot(psi - C) @ v_true_w

print(f"window {KEY} t={T0:.0f}+{DUR:.0f}s alt={ALT:.1f} gsd={GSD:.3f} d={DEPTH} "
      f"| GPS v_world=({vE:+.2f},{vN:+.2f}) v_ridge=({v_true_r[0]:+.2f},{v_true_r[1]:+.2f})")

FPS = 5.0
cube = cap.extract_cube(mp4, T0, DUR, width=640, fps=FPS)
g = cap.deglint(cube)
# NO temporal band-pass here: the likelihood probes the raw spectrum; band selection is done
# in k (wavelength gate) and the shell prediction handles the omega placement.
P, om_ax, kx_ax, ky_ax = _power_spectrum(g, 1.0 / FPS)

# omega>0 half, positive-frequency axis for interpolation
Tn = P.shape[0]
pos = om_ax > 0
om_pos = om_ax[pos]
o_order = np.argsort(om_pos)
om_sorted = om_pos[o_order]
P_pos = P[pos][o_order]                      # (n_om, H, W)

# spatial-k gate: wavelength 6-30 m
KX, KY = np.meshgrid(kx_ax, ky_ax)           # (H,W) rad/px, axis order: P[omega, ky, kx]
km_px = np.hypot(KX, KY)
km_m = km_px / GSD
kmask = (2 * np.pi / 30.0 <= km_m) & (km_m <= 2 * np.pi / 6.0)
# per-k total power (for the normalization: bright k must not dominate)
Ptot_k = P_pos.sum(0) + 1e-12

kprop_x = -KX[kmask] / GSD                   # propagation vector, rad/m (kappa = -K)
kprop_y = -KY[kmask] / GSD
km_sel = km_m[kmask]
sig = np.sqrt(G * km_sel * np.tanh(km_sel * DEPTH))
Pk = P_pos[:, kmask]                          # (n_om, n_k)
Pn = Pk / Ptot_k[kmask]                       # omega-normalized per k
idx_col = np.arange(Pk.shape[1])

def score(v):
    om_pred = sig + kprop_x * v[0] + kprop_y * v[1]
    ok = (om_pred > om_sorted[0]) & (om_pred < om_sorted[-1])
    j = np.searchsorted(om_sorted, om_pred[ok]) - 1
    f = (om_pred[ok] - om_sorted[j]) / (om_sorted[j + 1] - om_sorted[j])
    cols = idx_col[ok]
    vals = Pn[j, cols] * (1 - f) + Pn[j + 1, cols] * f
    return float(vals.sum())

# --- coarse-to-fine argmax over v (ridge frame, m/s) ---
best = (0.0, 0.0); bs = -1
for span, npts in ((VMAX, 25), (0.6, 21), (0.12, 21)):
    vs = np.linspace(-span, span, npts)
    b0 = best
    for dx in vs:
        for dy in vs:
            v = (b0[0] + dx, b0[1] + dy)
            s_ = score(v)
            if s_ > bs: bs, best = s_, v
v_hat_r = np.array(best)
v_hat_w = -rot(C - psi) @ v_hat_r
print(f"LIKELIHOOD argmax: v_ridge=({v_hat_r[0]:+.2f},{v_hat_r[1]:+.2f})  "
      f"v_world=({v_hat_w[0]:+.2f},{v_hat_w[1]:+.2f})  score={bs:.3f}  score(0)={score((0,0)):.3f}")
print(f"error vs GPS: ridge ({v_hat_r[0]-v_true_r[0]:+.2f},{v_hat_r[1]-v_true_r[1]:+.2f})  "
      f"|err|={np.hypot(*(v_hat_r-v_true_r)):.2f} m/s")

# --- unit-equivalence: ridge-LS on the SAME window (band-passed path) for comparison ---
bp = cap._bandpass(g, FPS, (1.6, 14.0))
r = cap._ridge(bp, GSD, FPS, cap.LAMBDA_BAND)
kk, omg, wt = r["kappa"] / GSD, r["omega"], r["weight"]
km = np.hypot(*kk.T)
U_ls = cap._fit_u_at_depth(kk, km, omg, wt, DEPTH)
print(f"ridge-LS same window:  v_ridge=({U_ls[0]:+.2f},{U_ls[1]:+.2f})  "
      f"|err|={np.hypot(*(U_ls-v_true_r)):.2f} m/s   (n_ridge={len(wt)})")
