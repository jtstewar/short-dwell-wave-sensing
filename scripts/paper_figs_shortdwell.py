"""Figures for the ECCV-MV 2026 short-dwell sensing paper.

All inputs are committed artifacts or raw flight video referenced by committed
command lines; no registered artifact is modified. Outputs land in
figs/.

  fig1_spectrum.pdf   real (omega, k_parallel) spectrum slices: hover (unshifted
                      ridge) vs moving window (Doppler-tilted), shell overlays
  fig2_sweep.pdf      velocity error vs window length (med + p90); sources:
                      captures/2026-07-04_shortwindow_channel.md + verified
                      re-run of scripts/mvdr_baseline.py (2026-07-12); per-window
                      evidence: supp/{mvdr,fft}_perwindow.csv
                      (regenerate via scripts/mvdr_perwindow_dump.py and
                      WLF scripts/fft_perwindow_dump.py)
  fig3_bathy.pdf      wave-derived profile + crawl soundings + cross-shore
                      position information content sigma_x(x) with kh band
  fig4_stack.pdf      Dover lull-sea stacking recovery (capture table)
  fig5_ood.png        copy of data/exhibits/dl_compare_f4.png (SegFormer OOD exhibit)

Usage: python3 scripts/paper_figs_shortdwell.py [fig1 fig2 ...]  (default: all)
"""
import csv, glob, math, os, re, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
FIGS = ROOT / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
D = Path(os.environ.get("WVIO_FLIGHTS", "data/flights"))
G = 9.81
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7})

def _tel(glob_pat):
    tel = list(csv.DictReader(open(glob.glob(str(ROOT / f"captures/telemetry_0701/{glob_pat}.csv"))[0])))
    def col(k):
        o = []
        for x in tel:
            try: o.append(float(x[k]))
            except Exception: o.append(np.nan)
        return np.array(o)
    return col

def _alt(key):
    srt = next(D.glob(f"*_{key}_D.SRT")).read_text(errors="replace")
    tc = re.findall(r"(\d\d):(\d\d):(\d\d),(\d\d\d) -->", srt)
    ra = np.array(re.findall(r"\[rel_alt: ([\d.\-]+)", srt), float)
    sts = np.array([int(a)*3600 + int(b)*60 + int(c) + int(dd)/1000 for a, b, c, dd in tc[:len(ra)]])
    return sts, ra[:len(sts)]

def _spec_slice(key, telg, off, t0, dur, depth, fps=5.0):
    """(omega, k_par) power slice along the dominant azimuth + shell overlays."""
    from wvio import capture as cap
    mp4 = next(D.glob(f"*_{key}_D.MP4"))
    sts, ra = _alt(key)
    alt = float(np.interp(t0 + dur/2, sts, ra))
    gsd = cap.gsd_for(alt, 640)
    g = cap.deglint(cap.extract_cube(mp4, float(t0), dur, width=640, fps=fps))
    g = g - g.mean(0)
    w_t = np.hanning(len(g))[:, None, None]
    w_y = np.hanning(g.shape[1])[None, :, None]
    w_x = np.hanning(g.shape[2])[None, None, :]
    F = np.fft.fftn(g * w_t * w_y * w_x)
    P = np.abs(F) ** 2
    om = 2 * np.pi * np.fft.fftfreq(len(g), d=1/fps)
    kx = 2 * np.pi * np.fft.fftfreq(g.shape[2], d=gsd)
    ky = 2 * np.pi * np.fft.fftfreq(g.shape[1], d=gsd)
    KX, KY = np.meshgrid(kx, ky)
    km = np.hypot(KX, KY)
    band = (km >= 2*np.pi/30) & (km <= 2*np.pi/6)
    # dominant azimuth (axial mean over positive-omega half-volume)
    Ppos = P[om > 0].sum(0)
    th = np.arctan2(KY, KX)
    z = (Ppos * band * np.exp(2j*th)).sum()
    th0 = 0.5 * np.angle(z)
    u = np.array([np.cos(th0), np.sin(th0)])
    kpar = KX * u[0] + KY * u[1]
    kperp = -KX * u[1] + KY * u[0]
    sel = band & (np.abs(kperp) < 0.25 * np.abs(kpar) + 0.08)
    # bin (omega>0) power onto (om, kpar) grid
    kbins = np.linspace(-2*np.pi/6, 2*np.pi/6, 81)
    obins = om[om >= 0]
    H = np.zeros((len(obins), len(kbins) - 1))
    oi = np.where(om >= 0)[0]
    for r, i in enumerate(oi):
        Pi = P[i][sel]; kv = kpar[sel]
        idx = np.clip(np.digitize(kv, kbins) - 1, 0, len(kbins) - 2)
        np.add.at(H[r], idx, Pi)
    kc = 0.5 * (kbins[1:] + kbins[:-1])
    return H, obins, kc, th0, gsd, alt

def fig1():
    from wvio import capture as cap  # noqa: F401  (import check before video IO)
    # hover: flight 0001 station F1 (hovers_0701.csv: t 35-165, depth 1.66 m)
    # moving: f3sec crawl (0004), t_center 512.5 s row of crawl_transect_0701_v2.csv
    #         depth_v2 2.30-3.6 band; pick the committed row at t=452.5 if cleaner
    rows = list(csv.DictReader(open(ROOT / "captures/crawl_transect_0701_v2.csv")))
    rows = [r for r in rows if float(r["off_shell"]) < 0.15 and 0.4 < float(r["kh"]) < 1.8
            and float(r["gps_speed"]) > 1.0]
    r0 = rows[len(rows)//2]
    t_c, d_mov = float(r0["t_center"]), float(r0["depth_v2"])
    v_mov = float(r0["gps_speed"])
    fig, axs = plt.subplots(1, 3, figsize=(4.9, 1.8), constrained_layout=True)
    # panel A (added v9.5): raw nadir frame from the same hover station, square
    # center crop, metre axes — grounds the scene before its spectrum.
    mp4 = next(D.glob("*_0001_D.MP4"))
    sts, ra = _alt("0001")
    alt0 = float(np.interp(90.0, sts, ra))
    gsd0 = cap.gsd_for(alt0, 640)
    cube0 = cap.extract_cube(mp4, 89.5, 1.0, width=640, fps=2.0)
    fr = np.clip(cube0[0] / max(cube0.max(), 1e-9), 0, 1)
    sq = fr[:, 140:500]
    L = sq.shape[0] * gsd0
    axs[0].imshow(sq, extent=[0, L, 0, L])
    axs[0].set(title=f"nadir frame ({alt0:.0f} m alt)", xlabel="m")
    axs[0].tick_params(labelsize=6)
    for ax, (key, telg, off, t0, dur, dep, ttl, vmag) in zip(axs[1:], [
        ("0001", "*15-58-55*", 34, 80.0, 20.0, 1.66, "hover, 20 s ($U\\approx0$)", 0.0),
        ("0004", "*16-40-11*", 15, t_c - 4.0, 8.0, d_mov, f"transect, 8 s ($|v|={v_mov:.1f}$ m/s)", v_mov)]):
        H, ob, kc, th0, gsd, alt = _spec_slice(key, telg, off, t0, dur, dep)
        Hs = H / np.maximum(H.max(0, keepdims=True), 1e-30)  # per-k normalization
        ax.pcolormesh(kc, ob, np.log10(Hs + 3e-3), cmap="magma", rasterized=True)
        kk = np.linspace(0.02, 2*np.pi/6, 200)
        for sgn in (+1, -1):
            sig = np.sqrt(G * kk * np.tanh(kk * dep))
            ax.plot(sgn * kk, sig, "c--", lw=0.8)
            if vmag > 0:
                ax.plot(sgn * kk, sig + sgn * kk * vmag, "w:", lw=0.8)
        ax.set(xlim=(-2*np.pi/6, 2*np.pi/6), ylim=(0, 4.2), title=ttl,
               xlabel="$k_\\parallel$ (rad/m)")
    axs[1].set_ylabel("$\\omega$ (rad/s)")
    axs[2].set_yticklabels([])
    fig.savefig(FIGS / "fig1_spectrum.pdf", dpi=300)
    if os.environ.get("FIG_PREVIEW"):
        fig.savefig(os.environ["FIG_PREVIEW"], dpi=200)
    print("fig1 done  (frame 0001@89.5s %.0fm; hover 0001@80s d=1.66; moving 0004@%.0fs d=%.2f |v|=%.2f)"
          % (alt0, t_c, d_mov, v_mov))

def fig2():
    # committed sources: shortwindow capture (FFT/NSP, n=20) + verified MVDR
    # re-run 2026-07-12 (identical inventory): 8s .50/3.01  6s .49/2.53
    # 4s .57/2.19  3s .70/1.99
    # FFT/NSP full row (med/p90 incl. 3 s) from the committed WLF crossover
    # table (wave-learned-frontend RESULTS.md, iteration 3, same inventory)
    dur = [8, 6, 4, 3]
    fft_med = [0.46, 0.47, 0.93, 1.58]; fft_p90 = [3.30, 3.14, 3.09, 2.48]
    mv_med = [0.50, 0.49, 0.57, 0.70];  mv_p90 = [3.01, 2.53, 2.19, 1.99]
    # drawn at 2.9 in for a 0.49\textwidth slot (was 3.2 in): fonts print
    # ~10% larger, same aspect ratio -> identical printed footprint
    fig, ax = plt.subplots(figsize=(2.9, 2.1), constrained_layout=True)
    ax.plot(dur, fft_med, "o-", color="#c44", label="FFT/NSP median")
    ax.plot(dur, fft_p90, "o--", color="#c44", alpha=0.45, label="FFT/NSP $p_{90}$")
    ax.plot(dur, mv_med, "s-", color="#268", label="Capon/MVDR median")
    ax.plot(dur, mv_p90, "s--", color="#268", alpha=0.45, label="MVDR $p_{90}$")
    ax.axhline(0.33, color="gray", lw=0.7, ls=":",)
    ax.text(5.5, 0.18, "20 s likelihood mean", color="0.35", fontsize=7,
            ha="center", va="center")
    ax.axvspan(2.8, 4.4, color="0.92", zorder=0)
    ax.text(3.55, 2.9, "resolution\nbreak", ha="center", fontsize=7, color="0.35")
    ax.set(xlabel="window length (s)", ylabel="velocity error (m/s)",
           xlim=(2.7, 8.3), ylim=(0, 3.4))
    ax.invert_xaxis()
    # center-left: the only zone no curve crosses (medians hug 0.5, p90s
    # stay above 2.5 there); upper-left sat on the FFT p90 curve
    ax.legend(loc="center left", framealpha=0.85, fontsize=6.5,
              handlelength=1.5, labelspacing=0.3, borderpad=0.35)
    fig.savefig(FIGS / "fig2_sweep.pdf"); print("fig2 done")

def fig3():
    z = np.load(ROOT / "captures/chart_0701.npz")
    x, dep = z["xgrid"], z["depth"]
    rows = list(csv.DictReader(open(ROOT / "captures/crawl_transect_0701_v2.csv")))
    cx = np.array([float(r["x_gps_m"]) for r in rows])
    cd = np.array([float(r["depth_v2"]) for r in rows])
    bad = ("weak", "rail", "near-deep")
    ok = np.array([float(r["off_shell"]) < 0.2 and
                   not any(b in r["verdict"] for b in bad) for r in rows])
    khv = np.array([float(r["kh"]) for r in rows])
    inband = ok & (khv >= 0.25) & (khv <= 2.0)
    outband = ok & ~((khv >= 0.25) & (khv <= 2.0))
    fig, axs = plt.subplots(2, 1, figsize=(3.35, 3.1), sharex=True, constrained_layout=True)
    axs[0].plot(x, dep, "-", color="#268", lw=1.4, label="hover-derived profile")
    axs[0].plot(cx[inband], cd[inband], ".", color="#c44", ms=4, label="transect-derived estimates")
    axs[0].plot(cx[outband], cd[outband], "o", mfc="none", mec="0.55", ms=4,
                mew=0.8, label="outside $kh$ band (rejected)")
    axs[0].invert_yaxis()
    axs[0].set_ylabel("depth (m)"); axs[0].legend(framealpha=0.9)
    # NOTE: an earlier version shaded "outside kh window" from a T=3.3 s
    # free-wave model; that model gives kh 0.78-1.41 over this whole reach
    # (never outside (0.25, 2)), so the shading never rendered and its
    # in-plot annotation was misleading — both removed 2026-07-14. The kh
    # ceiling is carried by the top panel's rejected markers and Sec. 6.2.
    grad = np.gradient(dep, x)
    sx = 0.3 / np.maximum(np.abs(grad), 1e-4)
    # cap the axis and shade sigma_x > cap instead of drawing divergent
    # spikes at extrema/terrace: same information, readable 10-25 m band
    cap = 60.0
    axs[1].fill_between(x, 0, cap, where=sx > cap, color="0.92", zorder=0)
    axs[1].plot(x, np.where(sx <= cap, sx, np.nan), "-", color="#268", lw=1.4)
    axs[1].text(-122, 31, "no locally unique\ncross-shore fix",
                fontsize=6.5, color="0.35", ha="center")
    # x=0 is the hover-station centroid (chart builder's LAT0/LON0), NOT
    # the shoreline (waterline is shoreward of the plotted reach); label
    # the origin so "cross-shore" isn't read as distance-from-shore
    axs[1].set(xlabel="cross-shore position (m; 0 = hover centroid)",
               ylabel="conditional $\\sigma_x$ (m)",
               ylim=(0, cap))
    axs[0].text(0.985, 0.93, "offshore $\\rightarrow$",
                transform=axs[0].transAxes, ha="right", va="top",
                fontsize=7, color="0.35")
    fig.savefig(FIGS / "fig3_bathy.pdf"); print("fig3 done")

def fig4():
    # captures/2026-07-04_dover_revisit.md addendum table (committed)
    crawls = ["C4\n57 m", "C5\n57 m", "C6\n57 m", "C3\n85 m", "C2\n85 m", "C7\n116 m"]
    single = [2.40, 2.63, 3.11, 0.94, 1.69, 0.95]
    stacked = [0.79, 0.68, 3.16, 0.89, 1.66, 0.92]
    xpos = np.arange(len(crawls))
    fig, ax = plt.subplots(figsize=(3.2, 2.0), constrained_layout=True)
    ax.bar(xpos - 0.19, single, 0.38, color="#bbb", label="per-window median")
    ax.bar(xpos + 0.19, stacked, 0.38, color="#268", label="stacked")
    ax.set_xticks(xpos, crawls, fontsize=6)
    ax.set_ylabel("velocity error (m/s)")
    ax.legend(framealpha=0.9)
    ax.set_title("lull seas ($H_s$ 0.1--0.2 m): stacking vs altitude", fontsize=7)
    fig.savefig(FIGS / "fig4_stack.pdf"); print("fig4 done")

def fig5():
    import shutil
    shutil.copy(ROOT / "data/exhibits/dl_compare_f4.png", FIGS / "fig5_ood.png")
    print("fig5 done (copied)")

def figobs():
    """fig5_obs.pdf: the measurement end-to-end on one 6-s transect window.
    Panels: raw nadir frame | (omega,k_par) slice + shells | velocity score
    surface + N-derived observability ellipse. Supplied depth = 3.0 m (the
    protocol's fixed nominal; Sec 3.6)."""
    from wvio import capture as cap
    DEP = 3.0
    rows = list(csv.DictReader(open(ROOT / "captures/crawl_transect_0701_v2.csv")))
    rows = [r for r in rows if float(r["off_shell"]) < 0.15 and 0.4 < float(r["kh"]) < 1.8
            and float(r["gps_speed"]) > 1.0]
    r0 = rows[len(rows)//2]
    t_c, vmag = float(r0["t_center"]), float(r0["gps_speed"])
    key, telg, OFF, dur = "0004", "*16-40-11*", 15, 6.0
    t0 = t_c - dur/2
    mp4 = next(D.glob(f"*_{key}_D.MP4"))
    sts, ra = _alt(key)
    alt = float(np.interp(t_c, sts, ra))
    gsd = cap.gsd_for(alt, 640)
    cube = cap.extract_cube(mp4, float(t0), dur, width=640, fps=5.0)
    frame = np.clip(cube[len(cube)//2] / max(cube.max(), 1e-9), 0, 1)
    # GNSS reference velocity in the camera-aligned frame (mvdr_baseline conv.)
    col = _tel(telg)
    ft, oy = col("OSD.flyTime"), col("OSD.yaw")
    tla, tlo = col("OSD.latitude"), col("OSD.longitude")
    mo = np.isfinite(oy); mv_ = (tla != 0) & np.isfinite(tla)
    psi_ = float(np.interp(t_c + OFF, ft[mo], np.unwrap(np.radians(oy[mo]))))
    tq = np.array([t_c - dur/2 + OFF, t_c + dur/2 + OFF])
    la2 = np.interp(tq, ft[mv_], tla[mv_]); lo2 = np.interp(tq, ft[mv_], tlo[mv_])
    v_w = np.array([(lo2[1]-lo2[0])*111320*math.cos(math.radians(la2.mean()))/dur,
                    (la2[1]-la2[0])*111320/dur])
    a = psi_ - math.radians(4.0)
    v_r = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]) @ v_w
    # 3-D spectrum
    g = cap.deglint(cube); g = g - g.mean(0)
    w3 = (np.hanning(len(g))[:, None, None] * np.hanning(g.shape[1])[None, :, None]
          * np.hanning(g.shape[2])[None, None, :])
    F = np.fft.fftn(g * w3); P = np.abs(F) ** 2
    om = 2 * np.pi * np.fft.fftfreq(len(g), d=0.2)
    kx = 2 * np.pi * np.fft.fftfreq(g.shape[2], d=gsd)
    ky = 2 * np.pi * np.fft.fftfreq(g.shape[1], d=gsd)
    KX, KY = np.meshgrid(kx, ky)
    km = np.hypot(KX, KY).ravel()
    band = (km >= 2*np.pi/30) & (km <= 2*np.pi/6)
    oi = np.where(om > 0)[0]; omp = om[oi]
    Pk = P[oi].reshape(len(oi), -1)[:, band]
    Pk = Pk / np.maximum(Pk.sum(0, keepdims=True), 1e-30)   # per-k normalize
    E = Pk.sum(0) * 0 + P[oi].reshape(len(oi), -1)[:, band].sum(0)
    keep = np.argsort(E)[::-1][:400]
    Pk = Pk[:, keep]
    kk = np.c_[-KX.ravel()[band][keep], -KY.ravel()[band][keep]]
    kmk = km[band][keep]
    sig = np.sqrt(G * kmk * np.tanh(kmk * DEP))
    vg = np.linspace(-3.5, 3.5, 71)
    S = np.zeros((71, 71))
    for iy, uy in enumerate(vg):
        for ix, ux in enumerate(vg):
            op = sig + kk[:, 0] * ux + kk[:, 1] * uy
            ok = (op > omp[0]) & (op < omp[-1])
            j = np.clip(np.searchsorted(omp, op[ok]) - 1, 0, len(omp) - 2)
            f = (op[ok] - omp[j]) / (omp[j+1] - omp[j])
            cc = np.arange(Pk.shape[1])[ok]
            S[iy, ix] = (Pk[j, cc] * (1-f) + Pk[j+1, cc] * f).sum()
    am = np.unravel_index(np.argmax(S), S.shape)
    U_hat = np.array([vg[am[1]], vg[am[0]]])
    # N-derived observability ellipse (orientation/aspect; display scale)
    wN = P[oi].reshape(len(oi), -1)[:, band][:, keep].sum(0)
    N = (wN[:, None, None] * (kk[:, :, None] * kk[:, None, :])).sum(0)
    evals, evecs = np.linalg.eigh(np.linalg.inv(N))
    ax_len = np.sqrt(evals) / np.sqrt(evals).max() * 1.2   # major axis 1.2 m/s
    th = np.linspace(0, 2*np.pi, 100)
    ell = (evecs @ np.diag(ax_len) @ np.vstack([np.cos(th), np.sin(th)])).T + U_hat
    # panels
    H, ob, kc, th0, _, _ = _spec_slice(key, telg, OFF, t0, dur, DEP)
    fig, axs = plt.subplots(1, 3, figsize=(4.8, 1.78), constrained_layout=True)
    axs[0].imshow(frame); axs[0].set_xticks([]); axs[0].set_yticks([])
    axs[0].set_title(f"nadir frame, {alt:.0f} m")
    Hs = H / max(H.max(), 1e-30)
    axs[1].pcolormesh(kc, ob, np.log10(Hs + 1e-4), cmap="magma", rasterized=True)
    kl = np.linspace(0.02, 2*np.pi/6, 200)
    sg = np.sqrt(G * kl * np.tanh(kl * DEP))
    for sgn in (+1, -1):
        axs[1].plot(sgn*kl, sg, "c--", lw=0.8)
        axs[1].plot(sgn*kl, sg + sgn*kl*vmag, "w:", lw=0.9)
    axs[1].set(xlim=(-2*np.pi/6, 2*np.pi/6), ylim=(0, 4.2),
               xlabel="$k_\\parallel$ (rad/m)", ylabel="$\\omega$ (rad/s)",
               title="power on the shifted shell")
    axs[2].pcolormesh(vg, vg, S, cmap="viridis", rasterized=True)
    axs[2].plot(*ell.T, "w-", lw=0.8, alpha=0.9)
    axs[2].plot(*U_hat, "wx", ms=6, mew=1.6)
    axs[2].plot(*v_r, "o", ms=5, mfc="none", mec="#19c24a", mew=1.6)
    axs[2].set(xlabel="$v_x$ (m/s)", ylabel="$v_y$ (m/s)",
               title="velocity score, $\\boldsymbol{N}$ ellipse")
    axs[2].set_aspect("equal")
    fig.savefig(FIGS / "fig5_obs.pdf", dpi=300)
    print(f"figobs done (0004 @ {t_c:.1f}s, alt {alt:.0f} m, |v_gnss| {np.hypot(*v_r):.2f},"
          f" argmax err {np.hypot(*(U_hat - v_r)):.2f} m/s)")

if __name__ == "__main__":
    want = sys.argv[1:] or ["fig1", "fig2", "fig3", "fig4", "figobs"]
    for w in want:
        globals()[w]()
