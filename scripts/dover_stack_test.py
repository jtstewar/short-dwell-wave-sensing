"""Score-surface STACKING test (user's buoy-prior insight, 2026-07-04): in a lull sea each
6-s window's score(U) surface is too flat to trust — but over a steady crawl the surfaces
ADD (signal stacks in U-space, noise averages down ~1/sqrt(N)). With sigma(k,d) fixed by
chart+buoy (a known template), can stacking cross the detection floor June-28 sits under?
Per crawl: (a) single-window argmax err (the failed baseline), (b) STACKED-surface argmax err,
(c) peak contrast before/after.
"""
import csv, math, os
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wvio import capture as cap
from wvio.wavefield import G

CACHE = Path(os.environ.get("WVIO_CACHE_0628", "data/flights/dji_fly_20260628_140048_cache.mp4"))
C = math.radians(4.0)
def rot(a): return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
FPS, DUR, DEPTH = 5.0, 6.0, 2.5
UG = np.arange(-3.0, 3.01, 0.15)               # common U grid for stacking

tel = list(csv.DictReader(open("captures/flight_2026-06-28_main.csv")))
def col(k):
    o = []
    for x in tel:
        try: o.append(float(x[k]))
        except Exception: o.append(np.nan)
    return np.array(o)
ft, oy = col("OSD.flyTime"), col("OSD.yaw")
tla, tlo = col("OSD.latitude"), col("OSD.longitude")
mo = np.isfinite(oy); mv = (tla != 0) & np.isfinite(tla)
SYNC = -17.0
rows = [r for r in csv.DictReader(open("captures/clips_0628_manifest.csv")) if r["type"] == "crawl"]
for r in rows:
    v0, v1, alt = float(r["v0"]), float(r["v1"]), float(r["alt"])
    gsd = cap.gsd_for(alt, 640)
    S_stack = np.zeros((len(UG), len(UG)))
    singles, vrs = [], []
    n = 0
    for T0 in np.arange(v0 + 1, v1 - DUR - 1, DUR):
        try:
            cube = cap.extract_cube(CACHE, float(T0), DUR, width=640, fps=FPS)
        except Exception:
            continue
        if len(cube) < int(DUR * FPS * 0.9): continue
        g = cap.deglint(cube)
        tm = T0 + DUR / 2 + SYNC
        psi_ = float(np.interp(tm, ft[mo], np.unwrap(np.radians(oy[mo]))))
        tq = np.array([T0 + SYNC, T0 + DUR + SYNC])
        la2 = np.interp(tq, ft[mv], tla[mv]); lo2 = np.interp(tq, ft[mv], tlo[mv])
        v_w = np.array([(lo2[1]-lo2[0])*111320*math.cos(math.radians(la2.mean()))/DUR,
                        (la2[1]-la2[0])*111320/DUR])
        vrs.append(rot(psi_ - C) @ v_w)
        T = len(g)
        gm = g - g.mean(0)
        F2 = np.fft.fft2(gm, axes=(1, 2))
        kx_ax = 2*np.pi*np.fft.fftfreq(g.shape[2], 1.0)/gsd
        ky_ax = 2*np.pi*np.fft.fftfreq(g.shape[1], 1.0)/gsd
        KX, KY = np.meshgrid(kx_ax, ky_ax)
        km2 = np.hypot(KX, KY)
        mask = (2*np.pi/30 <= km2) & (km2 <= 2*np.pi/6)
        ridx = np.where(mask.ravel())[0]
        E = (np.abs(F2)**2).sum(0).ravel()[ridx]
        keep = ridx[np.argsort(E)[::-1][:400]]
        ts_ = F2.reshape(T, -1)[:, keep]
        kk = np.c_[-KX.ravel()[keep], -KY.ravel()[keep]]
        kms = km2.ravel()[keep]
        sig = np.sqrt(G*kms*np.tanh(kms*DEPTH))
        Ff = np.fft.fft(ts_ * np.hanning(T)[:, None], axis=0)
        Pf = np.abs(Ff)**2
        om_all = 2*np.pi*np.fft.fftfreq(T, 1.0/FPS)
        pm = om_all > 0
        om_f = om_all[pm]; o_ = np.argsort(om_f)
        Pf = Pf[pm][o_]; om_f = om_f[o_]
        Pf = Pf/(Pf.sum(0, keepdims=True)+1e-12)
        cols_ = np.arange(len(keep))
        Sw = np.zeros((len(UG), len(UG)))
        for i, ux in enumerate(UG):
            op_base = sig + kk[:, 0]*ux
            for j, uy in enumerate(UG):
                op = op_base + kk[:, 1]*uy
                ok = (op > om_f[0]) & (op < om_f[-1])
                jj = np.clip(np.searchsorted(om_f, op[ok])-1, 0, len(om_f)-2)
                f = (op[ok]-om_f[jj])/(om_f[jj+1]-om_f[jj])
                cc = cols_[ok]
                Sw[i, j] = (Pf[jj, cc]*(1-f)+Pf[jj+1, cc]*f).sum()
        S_stack += Sw
        iu = np.unravel_index(np.argmax(Sw), Sw.shape)
        singles.append(np.array([UG[iu[0]], UG[iu[1]]]))
        n += 1
    if n < 3: continue
    v_r = np.mean(vrs, 0)
    e_single = float(np.median([np.hypot(*(u - vr)) for u, vr in zip(singles, vrs)]))
    iu = np.unravel_index(np.argmax(S_stack), S_stack.shape)
    U_st = np.array([UG[iu[0]], UG[iu[1]]])
    e_stack = float(np.hypot(*(U_st - v_r)))
    contrast = float(S_stack.max() / np.median(S_stack))
    print(f"{r['name']:>14} (|v|~{np.hypot(*v_w):.1f}, n={n}): single med {e_single:.2f}"
          f" -> STACKED {e_stack:.2f}  (peak/med contrast {contrast:.2f})", flush=True)
