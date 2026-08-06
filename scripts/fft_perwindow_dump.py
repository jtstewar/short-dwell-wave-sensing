"""Per-window full-spectrum-likelihood errors (the FFT column of the paired Table-1
statistic). Writes fft_perwindow.csv to CWD; compare against supp/fft_perwindow.csv.
Run from repo root; WVIO_FLIGHTS = raw flight dir."""
import csv, glob, math, os, re, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wvio import capture as cap
from fft_likelihood import spectrum_tables, argmax_U

D = Path(os.environ.get("WVIO_FLIGHTS", "data/flights"))
WV = Path(__file__).resolve().parents[1]
C = math.radians(4.0)
def rot(a): return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
FPS = 5.0
SECTIONS = [("0004", "*16-40-11*", 15, 410, 600), ("0005", "*16-52-59*", 23, 25, 110),
            ("0003", "*16-30-58*", 1339, 20, 330)]
out = open("fft_perwindow.csv", "w", newline="")   # compare against supp/fft_perwindow.csv
wr = csv.writer(out)
wr.writerow(["flight", "t0_s", "dur_s", "err_ms"])
for KEY, TELG, OFF, T0S, T1S in SECTIONS:
    tel = list(csv.DictReader(open(glob.glob(str(WV / "captures/telemetry_0701" / TELG) + ".csv")[0])))
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
    for T0 in np.arange(T0S, T1S - 8, 30.0):
        ALT = float(np.interp(T0 + 4, sts, ra[:len(sts)]))
        if ALT < 35: continue
        gsd = cap.gsd_for(ALT, 640)
        try:
            cube8 = cap.extract_cube(mp4, float(T0), 8.0, width=640, fps=FPS)
        except Exception:
            continue
        g8 = cap.deglint(cube8)
        for DUR in (8.0, 6.0, 4.0, 3.0):
            n = int(DUR * FPS)
            s0 = (len(g8) - n) // 2
            g = g8[s0:s0 + n]
            tm = T0 + s0 / FPS + DUR / 2
            psi_ = float(np.interp(tm + OFF, ft[mo], np.unwrap(np.radians(oy[mo]))))
            tq = np.array([tm - DUR/2 + OFF, tm + DUR/2 + OFF])
            la2 = np.interp(tq, ft[mv], tla[mv]); lo2 = np.interp(tq, ft[mv], tlo[mv])
            v_w = np.array([(lo2[1]-lo2[0])*111320*math.cos(math.radians(la2.mean()))/DUR,
                            (la2[1]-la2[0])*111320/DUR])
            v_r = rot(psi_ - C) @ v_w
            tab = spectrum_tables(g, FPS, gsd)
            U, _ = argmax_U(tab, 3.0)
            wr.writerow([KEY, f"{T0:.0f}", f"{DUR:.0f}", f"{float(np.hypot(*(U - v_r))):.3f}"])
        print(f"{KEY}@{T0:.0f} done", flush=True)
out.close()
print("FFT CSV written")
