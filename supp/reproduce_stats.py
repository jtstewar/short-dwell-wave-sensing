#!/usr/bin/env python3
"""Regenerates the uncertainty numbers of supplement Sec. 6 from the
per-window CSVs in this archive. Three analyses:
  (a) paired bootstrap over the 20 anchors (conditional on this inventory),
  (b) by-flight cluster resampling (3 clusters; robustness check only),
  (c) leave-one-flight-out sensitivity at the 4 s and 3 s cells.
Deterministic: NumPy default_rng(20260714), B = 20000, percentile
(2.5/97.5) intervals. Run in the directory containing
mvdr_perwindow.csv and fft_perwindow.csv. Requires numpy.
"""
import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
B = 20000
rng = np.random.default_rng(20260714)


def load(f):
    with open(HERE / f) as fh:
        return {(r["flight"], r["t0_s"], r["dur_s"]): float(r["err_ms"])
                for r in csv.DictReader(fh)}


mv, ff = load("mvdr_perwindow.csv"), load("fft_perwindow.csv")
assert set(mv) == set(ff)
durs = ["8", "6", "4", "3"]


def ci(x):
    return np.percentile(x, [2.5, 97.5])


print("(a) paired bootstrap over anchors, B=%d" % B)
for d in durs:
    keys = sorted(k for k in mv if k[2] == d)
    m = np.array([mv[k] for k in keys])
    f = np.array([ff[k] for k in keys])
    idx = rng.integers(0, len(keys), (B, len(keys)))
    dmed = np.median(f[idx], axis=1) - np.median(m[idx], axis=1)
    sub1 = (m[idx] < 1).mean(1) - (f[idx] < 1).mean(1)
    print(f"  {d}s: dmed(FFT-MVDR) 95% [{ci(dmed)[0]:+.2f},{ci(dmed)[1]:+.2f}] "
          f"P(MVDR better)={np.mean(dmed > 0):.3f}   "
          f"d(<1m/s) [{ci(sub1)[0]:+.2f},{ci(sub1)[1]:+.2f}] "
          f"P={np.mean(sub1 > 0):.3f}")

print("(b) by-flight cluster resampling (3 clusters)")
for d in ["4", "3"]:
    keys = sorted(k for k in mv if k[2] == d)
    fls = sorted(set(k[0] for k in keys))
    bykey = {fl: [k for k in keys if k[0] == fl] for fl in fls}
    dmeds = []
    for _ in range(B // 4):
        pick = rng.choice(fls, len(fls))
        ks = [k for fl in pick for k in bykey[fl]]
        m = np.array([mv[k] for k in ks])
        f = np.array([ff[k] for k in ks])
        dmeds.append(np.median(f) - np.median(m))
    dmeds = np.array(dmeds)
    print(f"  {d}s: dmed 95% [{ci(dmeds)[0]:+.2f},{ci(dmeds)[1]:+.2f}] "
          f"P={np.mean(dmeds > 0):.3f}  min={dmeds.min():+.2f}")

print("(c) leave-one-flight-out")
for d in ["4", "3"]:
    keys = sorted(k for k in mv if k[2] == d)
    for fl in sorted(set(k[0] for k in keys)):
        sub = [k for k in keys if k[0] != fl]
        m = np.array([mv[k] for k in sub])
        f = np.array([ff[k] for k in sub])
        print(f"  {d}s drop {fl} (n={len(sub)}): med FFT {np.median(f):.2f} "
              f"MVDR {np.median(m):.2f}  wins {(m < f).sum()}/{len(sub)}  "
              f"<1m/s {np.sum(f < 1)} vs {np.sum(m < 1)}")
