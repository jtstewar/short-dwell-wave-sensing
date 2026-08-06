"""Per-window MVDR error dump for the supplement (all 4 durations, d=3.0 m).
Identical inventory + estimator to scripts/mvdr_baseline.py; adds a CSV dump.
Run from wave-vio root."""
import csv, glob, math, os, re, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wvio import capture as cap
from wvio.wavefield import G

D = Path(os.environ.get("WVIO_FLIGHTS", "data/flights"))
C = math.radians(4.0)
def rot(a): return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
FPS = 5.0
SECTIONS = [("0004", "*16-40-11*", 15, 410, 600), ("0005", "*16-52-59*", 23, 25, 110),
            ("0003", "*16-30-58*", 1339, 20, 330)]
M = 10
OMG = np.linspace(0.3, 12.0, 120)

def mvdr_tables(g, gsd):
    T = len(g); gm = g - g.mean(0)
    F2 = np.fft.fft2(gm, axes=(1, 2))
    kx_ax = 2*np.pi*np.fft.fftfreq(g.shape[2], d=1.0)/gsd
    ky_ax = 2*np.pi*np.fft.fftfreq(g.shape[1], d=1.0)/gsd
    KX, KY = np.meshgrid(kx_ax, ky_ax)
    km = np.hypot(KX, KY)
    mask = (2*np.pi/30 <= km) & (km <= 2*np.pi/6)
    ridx = np.where(mask.ravel())[0]
    E = (np.abs(F2)**2).sum(0).ravel()[ridx]
    keep = ridx[np.argsort(E)[::-1][:400]]
    ts = F2.reshape(T, -1)[:, keep]
    nsnap = T - M + 1
    P = np.zeros((len(OMG), ts.shape[1]))
    steer = np.exp(1j*np.outer(np.arange(M)/FPS, OMG))
    for kbin in range(ts.shape[1]):
        X = np.stack([ts[i:i+M, kbin] for i in range(nsnap)], 1)
        R = X @ X.conj().T / nsnap
        R += np.trace(R).real/M*0.03*np.eye(M)
        Ri = np.linalg.inv(R)
        den = np.einsum("mo,mn,no->o", steer.conj(), Ri, steer).real
        P[:, kbin] = 1.0/np.maximum(den, 1e-12)
    P = P/(P.sum(0, keepdims=True)+1e-12)
    kk = np.c_[-KX.ravel()[keep], -KY.ravel()[keep]]
    return P, kk, km.ravel()[keep]

def argmax_U(P, kk, km, depth, vmax=6.0):
    sig = np.sqrt(G*km*np.tanh(km*depth))
    best, bs = np.zeros(2), -1.0
    for span, npts in ((vmax, 25), (0.5, 11), (0.15, 9)):
        vs = np.linspace(-span, span, npts); b0 = best.copy()
        for dx in vs:
            for dy in vs:
                U = b0 + (dx, dy)
                om_pred = sig + kk[:, 0]*U[0] + kk[:, 1]*U[1]
                ok = (om_pred > OMG[0]) & (om_pred < OMG[-1])
                j = np.clip(np.searchsorted(OMG, om_pred[ok])-1, 0, len(OMG)-2)
                f = (om_pred[ok]-OMG[j])/(OMG[j+1]-OMG[j])
                cc = np.arange(P.shape[1])[ok]
                s_ = float((P[j, cc]*(1-f)+P[j+1, cc]*f).sum())
                if s_ > bs: bs, best = s_, U
    return best

out = open("mvdr_perwindow.csv", "w", newline="")   # compare against supp/mvdr_perwindow.csv
wr = csv.writer(out)
wr.writerow(["flight", "t0_s", "dur_s", "alt_m", "truth_vx", "truth_vy",
             "est_vx", "est_vy", "err_ms"])
for KEY, TELG, OFF, T0S, T1S in SECTIONS:
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
    mp4 = next(D.glob(f"*_{KEY}_D.MP4"))
    srt = next(D.glob(f"*_{KEY}_D.SRT")).read_text(errors="replace")
    tc = re.findall(r"(\d\d):(\d\d):(\d\d),(\d\d\d) -->", srt)
    ra = np.array(re.findall(r"\[rel_alt: ([\d.\-]+)", srt), float)
    sts = np.array([int(a)*3600+int(b)*60+int(c)+int(dd)/1000 for a, b, c, dd in tc[:len(ra)]])
    for T0 in np.arange(T0S, T1S-8, 30.0):
        ALT = float(np.interp(T0+4, sts, ra[:len(sts)]))
        if ALT < 35: continue
        gsd = cap.gsd_for(ALT, 640)
        try:
            cube8 = cap.extract_cube(mp4, float(T0), 8.0, width=640, fps=FPS)
        except Exception:
            continue
        if len(cube8) < 36: continue
        g8 = cap.deglint(cube8)
        for DUR in (8.0, 6.0, 4.0, 3.0):
            n = int(DUR*FPS)
            if n <= M: continue
            s0 = (len(g8)-n)//2
            g = g8[s0:s0+n]
            tm = T0 + s0/FPS + DUR/2
            psi_ = float(np.interp(tm+OFF, ft[mo], np.unwrap(np.radians(oy[mo]))))
            tq = np.array([tm-DUR/2+OFF, tm+DUR/2+OFF])
            la2 = np.interp(tq, ft[mv], tla[mv]); lo2 = np.interp(tq, ft[mv], tlo[mv])
            v_w = np.array([(lo2[1]-lo2[0])*111320*math.cos(math.radians(la2.mean()))/DUR,
                            (la2[1]-la2[0])*111320/DUR])
            v_r = rot(psi_-C) @ v_w
            P, kk, km = mvdr_tables(g, gsd)
            U = argmax_U(P, kk, km, 3.0)
            wr.writerow([KEY, f"{T0:.0f}", f"{DUR:.0f}", f"{ALT:.1f}",
                         f"{v_r[0]:.3f}", f"{v_r[1]:.3f}", f"{U[0]:.3f}",
                         f"{U[1]:.3f}", f"{np.hypot(*(U-v_r)):.3f}"])
        print(f"{KEY}@{T0:.0f} done", flush=True)
out.close()
print("CSV written")
