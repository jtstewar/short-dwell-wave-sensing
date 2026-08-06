"""Crawl-as-transect v2 — SURVEY MODE (fix the shallow-depth bias found 2026-07-02).

v1 (crawl_transect_0701.py) ran hover_depth's JOINT depth+current fit on each moving window with
u_prior=0.5, u_bound=2.0. The ~2.05 m/s platform Doppler RAILS at the bound (cur_mag pegs at
2.00-2.01) and the clipped residual leaks into depth -> systematically SHALLOW vs the hover
profile at the same x (0.2 m nearshore -> ~2.8 m at x=+88). The #12 degeneracy, shallow direction.

Fix (the dual of #12): in survey mode the platform velocity is KNOWN (GPS). Project it into the
image frame with the resolved convention (2026-07-02: v_img = diag(1,-1) @ R(-(GIMBAL.yaw-180deg)) @ v_ENU),
shift the ridge omega by +k.v_plat (since omega = sigma(k,d) + k.(U_cur - v_plat)), then run the
standard small-current depth fit. Depth is then well-posed per window.

Outputs: captures/crawl_transect_0701_v2.csv, output/crawl_transect_v2_0701.png
(both v1 and v2 depths per window + the hover profile for validation).
"""
import csv, math, os, re, glob, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wvio import capture as cap

D = Path(os.environ.get("WVIO_FLIGHTS", "data/flights"))
CT0, CT1, ALT = 502.5, 609.1, 67.1
KEY, TELG = "0004", "*16-40-11*"
C_MOUNT = math.radians(-180.0)          # resolved 2026-07-02 (yaw_convention_0701.py)
FPS, W_PX = 4.0, 640

# ---- hover profile (validation reference; same frame as v1) ----
rows = [r for r in csv.DictReader(open("captures/hovers_0701.csv")) if r["depth_m"]]
dep = np.array([float(r["depth_m"]) for r in rows]); railed = dep >= 15
hla = np.array([float(r["lat"]) for r in rows]); hlo = np.array([float(r["lon"]) for r in rows])
lat0, lon0 = hla.mean(), hlo.mean()
def EN(la, lo): return (lo - lon0) * 111320 * math.cos(math.radians(lat0)), (la - lat0) * 111320
hE, hN = EN(hla, hlo); m = ~railed
co, *_ = np.linalg.lstsq(np.c_[hE[m], hN[m], np.ones(m.sum())], dep[m], rcond=None)
n = np.array([co[0], co[1]]) / math.hypot(co[0], co[1])
hx = hE * n[0] + hN * n[1]
q = np.polyfit(hx[m], dep[m], 2)

# ---- SRT GPS + telemetry gimbal yaw, synced ----
txt = next(D.glob(f"*_{KEY}_D.SRT")).read_text(errors="replace")
tc = re.findall(r"(\d\d):(\d\d):(\d\d),(\d\d\d) -->", txt)
sla = np.array(re.findall(r"\[latitude: ([\d.\-]+)\]", txt), float)
slo = np.array(re.findall(r"\[longitude: ([\d.\-]+)\]", txt), float)
sts = np.array([int(a)*3600 + int(b)*60 + int(c) + int(d)/1000 for a, b, c, d in tc[:len(sla)]])

tel = list(csv.DictReader(open(glob.glob(f"captures/telemetry_0701/{TELG}.csv")[0])))
def tcol(k):
    o = []
    for x in tel:
        try: o.append(float(x[k]))
        except Exception: o.append(np.nan)
    return np.array(o)
ft, tla, tlo, tgy = tcol("OSD.flyTime"), tcol("OSD.latitude"), tcol("OSD.longitude"), tcol("GIMBAL.yaw")
mv = (tla != 0) & np.isfinite(tla)
best = None
for off in np.arange(ft[mv].min() - 30, ft[mv].min() + 150, 1.0):
    vt = sts[(sts + off >= ft[mv].min()) & (sts + off <= ft[mv].max())]
    if len(vt) < 50: continue
    te, tn = EN(np.interp(vt + off, ft[mv], tla[mv]), np.interp(vt + off, ft[mv], tlo[mv]))
    se, sn = EN(np.interp(vt, sts, sla), np.interp(vt, sts, slo))
    rms = np.sqrt(((te - se) ** 2 + (tn - sn) ** 2).mean())
    if best is None or rms < best[1]: best = (off, rms)
OFF = best[0]; print(f"sync offset {OFF:.0f}s (GPS match {best[1]:.1f} m)", flush=True)
mg = np.isfinite(tgy)
gyaw_u = np.unwrap(np.radians(tgy[mg])); gyt = ft[mg]

def rot(a): return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
FLIP = np.diag([1.0, -1.0])
R90 = rot(math.pi / 2)   # ridge-k frame = OF pixel frame rotated +90deg (FFT row/col vs pixel x/y);
                         # measured 2026-07-02: rot90 -> rescur 0.37 (=real current), off 0.03; all
                         # other orthogonal variants rail rescur at ~2 m/s (platform leaks into fit)

def v_plat_img(t, dt=2.0):
    """GPS ENU velocity at video time t -> RIDGE-frame platform velocity (convention + R90)."""
    la1, lo1 = np.interp(t - dt, sts, sla), np.interp(t - dt, sts, slo)
    la2, lo2 = np.interp(t + dt, sts, sla), np.interp(t + dt, sts, slo)
    vE = (lo2 - lo1) * 111320 * math.cos(math.radians(lat0)) / (2 * dt)
    vN = (la2 - la1) * 111320 / (2 * dt)
    psi = float(np.interp(t + OFF, gyt, gyaw_u))
    return R90 @ FLIP @ rot(-(psi + C_MOUNT)) @ np.array([vE, vN]), math.hypot(vE, vN)

gsd = cap.gsd_for(ALT, W_PX)
print("extracting full crawl cube...", flush=True)
cube = cap.extract_cube(str(next(D.glob(f"*_{KEY}_D.MP4"))), CT0, CT1 - CT0, W_PX, FPS)
print(f"cube {cube.shape}", flush=True)

v1 = {float(r["t_center"]): r for r in csv.DictReader(open("captures/crawl_transect_0701.csv"))}
Wn, step = int(20 * FPS), int(6 * FPS)
out = open("captures/crawl_transect_0701_v2.csv", "w"); w = csv.writer(out)
w.writerow(["t_center", "x_gps_m", "depth_v2", "depth_v1", "depth_hover_profile",
            "kh", "off_shell", "resid_cur", "gps_speed", "verdict"]); out.flush()
res = []
for s in range(0, len(cube) - Wn + 1, step):
    t = CT0 + (s + Wn / 2) / FPS
    try:
        g = cap.deglint(cube[s:s + Wn]); bp = cap._bandpass(g, FPS, cap.CRAWL_T_BAND)
        r = cap._ridge(bp, gsd, FPS, cap.LAMBDA_BAND)
        k = r["kappa"] / gsd; km = np.hypot(*k.T); omg = r["omega"]; wt = r["weight"]
        vp, gsp = v_plat_img(t)
        omg_c = omg + k @ vp                                   # remove the platform Doppler
        d2, ux, uy = cap._fit_d_u(k, km, omg_c, wt, u_prior=0.5, u_bound=2.0)
        pred = np.sqrt(cap.G * km * np.tanh(km * d2)) + k[:, 0] * ux + k[:, 1] * uy
        offsh = float(np.average(np.abs(omg_c - pred), weights=wt) / np.average(np.abs(omg_c), weights=wt))
        kh = float(np.average(km, weights=wt)) * d2
        la_, lo_ = np.interp(t, sts, sla), np.interp(t, sts, slo)
        E_, N_ = EN(la_, lo_); xg = E_ * n[0] + N_ * n[1]
        dh = float(np.polyval(q, np.clip(xg, hx[m].min(), hx[m].max())))
        d1 = float(v1[t]["depth_m"]) if t in v1 else np.nan
        v = "obs" if (offsh < 0.35 and kh < 1.8 and d2 < 15) else "near-deep/weak"
        w.writerow([f"{t:.1f}", f"{xg:.1f}", f"{d2:.2f}", f"{d1:.2f}", f"{dh:.2f}",
                    f"{kh:.2f}", f"{offsh:.2f}", f"{math.hypot(ux, uy):.2f}", f"{gsp:.2f}", v]); out.flush()
        res.append((xg, d2, d1, dh, v == "obs"))
        print(f"t={t:.0f} x={xg:+5.0f}  v2={d2:5.2f}  v1={d1:5.2f}  hover={dh:5.2f}  "
              f"kh={kh:.2f} off={offsh:.2f} rescur={math.hypot(ux, uy):.2f} {v}", flush=True)
    except Exception as e:
        print("win", t, "ERR", e, flush=True)
out.close()

res = np.array([(a, b, c, d) for a, b, c, d, _ in res])
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(9, 5.5))
xs = np.linspace(res[:, 0].min() - 10, res[:, 0].max() + 10, 200)
ax.plot(xs, np.polyval(q, np.clip(xs, hx[m].min(), hx[m].max())), "-", c="gray", lw=2,
        label="hover concave profile (independent)")
ax.plot(hx[m], dep[m], "o", c="k", ms=7, label="hovers")
ax.plot(res[:, 0], res[:, 2], "s--", c="tab:red", ms=5, label="crawl v1 (joint fit — biased)")
ax.plot(res[:, 0], res[:, 1], "^-", c="tab:blue", ms=6, label="crawl v2 (survey mode)")
ax.invert_yaxis(); ax.grid(alpha=.3); ax.legend()
ax.set(xlabel="cross-shore x (m, +offshore)", ylabel="depth (m)",
       title="Crawl transect v2: platform Doppler removed via GPS + resolved yaw convention")
fig.tight_layout(); fig.savefig("output/crawl_transect_v2_0701.png", dpi=120)
mfin = np.isfinite(res[:, 1]) & np.isfinite(res[:, 3])
rms2 = float(np.sqrt(np.mean((res[mfin, 1] - res[mfin, 3]) ** 2)))
rms1 = float(np.sqrt(np.nanmean((res[:, 2] - res[:, 3]) ** 2)))
print(f"\nvs hover profile: v1 RMS {rms1:.2f} m -> v2 RMS {rms2:.2f} m")
print("saved output/crawl_transect_v2_0701.png\nCRAWL TRANSECT V2 DONE", flush=True)
