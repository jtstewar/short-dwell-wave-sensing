"""Supplement figure: sun-glint suppression before/after (deglint A/B exhibit).

Window: flight 0004 crawl, t0=490 s, 20 s @ 5 Hz, 640 px — the SAME window as the
raw-vs-deglinted on-shell score A/B in captures/2026-07-03_rigor_sprint.md §8
(shell @ +v_r: 60 deglinted / 52 raw, hypothesis-point diagnostic,
scripts/two_component_likelihood.py). Deterministic: frame choice is the argmax
of the per-frame 99.9th-percentile brightness of the min-of-RGB projection
(the glintiest frame in the window).

Run from wave-vio root:  python3 scripts/supp_glint_fig.py
Writes figs/supp_glint.png
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wvio import capture as cap

D = Path(os.environ.get("WVIO_FLIGHTS", "data/flights"))
mp4 = next(D.glob("*_0004_D.MP4"))
cube = cap.extract_cube(mp4, 490.0, 20.0, width=640, fps=5.0)

g_raw = cap._gray(cube)                       # min-of-RGB projection, no deglint
g_dg = cap.deglint(cube)                      # + brightest-2% -> temporal median

t = int(np.argmax([np.percentile(f, 99.9) for f in g_raw]))
frac = float((g_raw[t] != g_dg[t]).mean()) * 100.0
print(f"glintiest frame: t={t} (video {490 + t/5.0:.1f} s); "
      f"{frac:.1f}% of pixels replaced")

vmax = np.percentile(g_raw[t], 99.9)          # shared scale so sparkle saturates
fig, ax = plt.subplots(1, 3, figsize=(9.0, 2.35), constrained_layout=True)
for a, img, title in ((ax[0], g_raw[t], "min-of-RGB projection (raw)"),
                      (ax[1], g_dg[t], "after glint suppression"),
                      (ax[2], np.abs(g_raw[t] - g_dg[t]), "|difference| (replaced pixels)")):
    a.imshow(img, cmap="gray", vmin=0, vmax=vmax, interpolation="nearest")
    a.set_title(title, fontsize=9)
    a.set_xticks([]); a.set_yticks([])
out = Path("figs/supp_glint.png")
fig.savefig(out, dpi=200)
print(f"wrote {out}")
