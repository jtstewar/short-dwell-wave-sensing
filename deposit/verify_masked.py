"""Verify the masked deposit videos against the originals and the committed CSVs.

Per file: duration and frame count unchanged; a mid-envelope analysis cube is
byte-identical to the original; masked head/tail frames decode as black.
For 0003, the land-band mask overlaps the first Table-1 anchor (t0=20), so that
anchor's MVDR estimate is recomputed from the masked file and compared against
both the original file and the committed supp/mvdr_perwindow.csv rows.

Usage: python3 deposit/verify_masked.py [key ...]   (default: all built)
"""
import csv, glob, json, math, os, re, subprocess, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from wvio import capture as cap
import mvdr_perwindow_dump as mv

D = Path(os.environ.get("WVIO_FLIGHTS", "/mnt/c/Users/james/DJI July 1s"))
CACHE = Path(os.environ.get("WVIO_CACHE_0628",
             "/mnt/c/Users/james/Downloads/dji_fly_20260628_140048_cache.mp4"))
OUT = Path(os.environ.get("DEPOSIT_OUT", str(Path.home() / "deposit_staging")))
MIDT = {"0001": 600.0, "0003": 100.0, "0004": 500.0, "0005": 60.0,
        "0006": 100.0, "cache": 400.0}


def probe(src, what):
    p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", what, "-of", "csv=p=0", str(src)],
                       capture_output=True, text=True, check=True)
    return p.stdout.strip().splitlines()[0]


def frame_at(src, t):
    return cap.extract_cube(src, t, 0.6, width=640, fps=2.0)[0]


def check(key):
    meta = json.loads((OUT / f"mask_{key}.json").read_text())
    src, out = Path(meta["src"]), Path(meta["out"])
    if not src.exists():  # build may have run from a scratch copy
        src = CACHE if key == "cache" else next(D.glob(f"*_{key}_D.MP4"))
    ok = True

    d0, d1 = float(probe(src, "format=duration")), float(probe(out, "format=duration"))
    n0, n1 = probe(src, "stream=nb_frames"), probe(out, "stream=nb_frames")
    if abs(d0 - d1) > 0.05 or n0 != n1:
        print(f"{key}: FAIL duration/frames orig {d0:.3f}/{n0} masked {d1:.3f}/{n1}")
        ok = False
    else:
        print(f"{key}: duration {d1:.3f} s, {n1} frames (match)")

    a = cap.extract_cube(src, MIDT[key], 4.0, width=640, fps=5.0)
    b = cap.extract_cube(out, MIDT[key], 4.0, width=640, fps=5.0)
    same = a.shape == b.shape and np.array_equal(a, b)
    print(f"{key}: mid-envelope cube @{MIDT[key]:.0f}s byte-identical: {same}")
    ok &= same

    if meta["kA"] > 1.0:
        mx = frame_at(out, meta["kA"] / 2).max()
        print(f"{key}: head frame max px {mx:.1f} (black<=2: {mx <= 2})"); ok &= mx <= 2
    if meta["kB"] < meta["dur"] - 1.0:
        mx = frame_at(out, (meta["kB"] + meta["dur"]) / 2).max()
        print(f"{key}: tail frame max px {mx:.1f} (black<=2: {mx <= 2})"); ok &= mx <= 2
    if meta.get("band"):
        b0, b1, x0, x1, y0 = meta["band"]
        fo, fm = frame_at(src, 25.0), frame_at(out, 25.0)
        h, w = fo.shape[:2]
        ys, xs = slice(int(h * y0), h), slice(int(w * x0), int(w * x1))
        din = float(np.abs(fo[ys, xs].astype(int) - fm[ys, xs].astype(int)).mean())
        outside = np.abs(fo[:int(h * y0)].astype(int) - fm[:int(h * y0)].astype(int))
        print(f"{key}: band @25s blurred-box mean diff {din:.1f} (masked>3: {din > 3}), "
              f"outside-box mean diff {float(outside.mean()):.2f}")
        ok &= din > 3
    return ok


def anchor_0003():
    """Recompute the 0003/t0=20 anchor on original vs masked; compare to committed CSV."""
    KEY, TELG, OFF, T0 = "0003", "*16-30-58*", 1339, 20.0
    tel = list(csv.DictReader(open(glob.glob(str(ROOT / f"captures/telemetry_0701/{TELG}.csv"))[0])))
    def col(k):
        o = []
        for x in tel:
            try: o.append(float(x[k]))
            except Exception: o.append(np.nan)
        return np.array(o)
    ft, oy = col("OSD.flyTime"), col("OSD.yaw")
    tla, tlo = col("OSD.latitude"), col("OSD.longitude")
    mo = np.isfinite(oy); mv_ = (tla != 0) & np.isfinite(tla)
    srt = next(D.glob(f"*_{KEY}_D.SRT")).read_text(errors="replace")
    tc = re.findall(r"(\d\d):(\d\d):(\d\d),(\d\d\d) -->", srt)
    ra = np.array(re.findall(r"\[rel_alt: ([\d.\-]+)", srt), float)
    sts = np.array([int(a)*3600+int(b)*60+int(c)+int(dd)/1000 for a, b, c, dd in tc[:len(ra)]])
    ALT = float(np.interp(T0 + 4, sts, ra[:len(sts)]))
    gsd = cap.gsd_for(ALT, 640)
    committed = {r["dur_s"]: r for r in csv.DictReader(open(ROOT / "supp/mvdr_perwindow.csv"))
                 if r["flight"] == KEY and r["t0_s"] == "20"}
    meta = json.loads((OUT / "mask_0003.json").read_text())
    for name, mp4 in (("original", next(D.glob(f"*_{KEY}_D.MP4"))), ("masked", Path(meta["out"]))):
        g8 = cap.deglint(cap.extract_cube(mp4, T0, 8.0, width=640, fps=mv.FPS))
        for DUR in (8.0, 6.0, 4.0, 3.0):
            n = int(DUR * mv.FPS)
            s0 = (len(g8) - n) // 2
            g = g8[s0:s0 + n]
            tm = T0 + s0 / mv.FPS + DUR / 2
            psi_ = float(np.interp(tm + OFF, ft[mo], np.unwrap(np.radians(oy[mo]))))
            tq = np.array([tm - DUR/2 + OFF, tm + DUR/2 + OFF])
            la2 = np.interp(tq, ft[mv_], tla[mv_]); lo2 = np.interp(tq, ft[mv_], tlo[mv_])
            v_w = np.array([(lo2[1]-lo2[0])*111320*math.cos(math.radians(la2.mean()))/DUR,
                            (la2[1]-la2[0])*111320/DUR])
            v_r = mv.rot(psi_ - mv.C) @ v_w
            P, kk, km = mv.mvdr_tables(g, gsd)
            U = mv.argmax_U(P, kk, km, 3.0)
            err = float(np.hypot(*(U - v_r)))
            ref = committed.get(f"{DUR:.0f}")
            refs = f" committed {float(ref['err_ms']):.3f}" if ref else ""
            print(f"0003/t0=20 {DUR:.0f}s {name}: err {err:.3f} m/s{refs}")


if __name__ == "__main__":
    keys = sys.argv[1:] or sorted(p.stem[5:] for p in OUT.glob("mask_*.json"))
    allok = all([check(k) for k in keys])
    if "0003" in keys:
        anchor_0003()
    print("VERIFY", "PASS" if allok else "FAIL (see above)")
