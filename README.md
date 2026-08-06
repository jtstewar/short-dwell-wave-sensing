# Short-Dwell Aerial Wave Sensing over Coastal Waters

Code and data release for:

> **Short-Dwell Aerial Wave Sensing over Coastal Waters: Feasibility, Observability,
> and Failure Modes**
> James Stewart and John Zelek — Systems Design Engineering, University of Waterloo
> *ECCV 2026, 2nd Workshop on Marine Vision (archival proceedings).*

A few seconds of nadir drone video over coastal water is enough to measure platform
velocity, water depth, and swell orientation — because the wave field obeys a known
physical law (the dispersion relation), and platform motion Doppler-shifts it. This
repository contains the measurement-layer code, the per-window evaluation data behind
every number in the paper, and deterministic scripts that regenerate the paper's
statistics and figures.

**Scope.** This release is the paper's measurement layer only: single-window and
stacked-window estimators evaluated against GNSS truth. The navigation system built on
these measurements (fusion filter, terrain-relative charts, online deployment) is
separate, ongoing work.

## Layout

```
src/wvio/            measurement-layer package (numpy + scipy)
  wavefield.py         dispersion relation, wave-field model
  spectral.py          3-D power spectrum, dispersion-ridge extraction (ridge-LS family)
  capture.py           video -> space-time cube, deglint, hover joint depth-current fit
  estimator.py         windowed shell fits and fusion of window measurements
scripts/             evaluation drivers (per-window dumps, figures; see ladder below)
supp/                the paper supplement's evidence pack: per-window CSVs +
                     reproduce_stats.py (regenerates every uncertainty number)
captures/            committed field-data derivatives: DJI telemetry CSVs, hover and
                     crawl-transect tables, hover worklist
data/rasters/        clipped public bathymetry rasters (consistency checks only)
data/exhibits/       committed exhibit images used by the figure script
data/flights/        EMPTY - place raw flight video here (see Data availability)
figs/                output directory for regenerated figures
```

## Reproduction ladder

Everything deterministic; seeds are fixed in the scripts.

**Tier 0 — statistics, no video needed:**

```bash
cd supp && python3 reproduce_stats.py
```

Regenerates the supplement's uncertainty analyses from the committed per-window CSVs:
paired bootstrap over the 20 short-window anchors, by-flight cluster resampling, and
leave-one-flight-out sensitivity (B = 20000, seed 20260714).

**Tier 1 — figures from committed inputs:**

```bash
python3 scripts/paper_figs_shortdwell.py fig2 fig3 fig4 fig5
```

Window-length sweep (fig 2), bathymetric profile + information content (fig 3),
lull-sea stacking recovery (fig 4), segmentation OOD exhibit (fig 5). Fig 1 (real
spectrum slices) needs raw video — see Tier 2.

**Tier 2 — full pipeline from raw video:**

Place the raw flights in `data/flights/` (or set `WVIO_FLIGHTS=/path/to/flights`), then:

```bash
# per-window velocity errors vs GNSS truth (the paired Table-1 evidence)
python3 scripts/mvdr_perwindow_dump.py     # Capon/MVDR column
python3 scripts/fft_perwindow_dump.py      # full-spectrum-likelihood column
# measurement models
python3 scripts/process_hovers_0701.py     # hover joint depth-current fits (needs rasterio+pyproj)
python3 scripts/crawl_transect_0701_v2.py  # depth while moving (survey mode)
python3 scripts/dover_stack_test.py        # dwell-as-a-resource stacking (June-28 cache clip)
# supplement exhibits
python3 scripts/depth_sensitivity.py       # MVDR depth-conditioning sensitivity
python3 scripts/supp_glint_fig.py          # sun-glint suppression A/B
python3 scripts/paper_figs_shortdwell.py   # all five figures
```

Run all scripts from the repo root. The per-window dumps write CSVs to the current
directory for comparison against the committed copies in `supp/`.

## Environment

Python ≥ 3.12 with the pinned versions in `requirements.txt`, plus `ffmpeg` on PATH
for video extraction. `rasterio`/`pyproj` are needed only by `process_hovers_0701.py`.
The paper's numbers were produced under the pinned manifest; spectral estimators
involve FFT/eigen routines whose last digits can move across major numpy/BLAS
versions, so use the pins when comparing against the committed CSVs.

## Data

- **Field data:** three flights on 2026-07-01 at Long Point, Ontario (Lake Erie) and
  hover/crawl sessions on 2026-06-21/28 at Port Dover, Ontario. DJI Mini series,
  nadir gimbal, 4K HEVC + SRT; GNSS/telemetry truth from the flight logs
  (`captures/telemetry_0701/`, `captures/flight_2026-06-28_main.csv`).
- **Committed derivatives** (everything the statistics and most figures need): the
  per-window error CSVs (`supp/`), hover fit tables, crawl-transect depth tables, and
  the hover worklist with window boundaries and GNSS positions (`captures/`).
- **Raw video** (~36 GB) is not hosted here — available from the authors on request
  (an archived deposit is planned).
- **Bathymetry rasters** in `data/rasters/` are small clips of public datasets used
  as consistency checks only (the sites have no independent surf-zone depth truth;
  that gap is part of the paper's point): NONNA-10 (Canadian Hydrographic Service,
  Open Government Licence – Canada) and NCEI CUDEM (NOAA, public domain).

## Citation

Until the proceedings entry is out, please cite via `CITATION.cff` (GitHub's
"Cite this repository" button), or:

```bibtex
@inproceedings{stewart2026shortdwell,
  title     = {Short-Dwell Aerial Wave Sensing over Coastal Waters:
               Feasibility, Observability, and Failure Modes},
  author    = {Stewart, James and Zelek, John},
  booktitle = {ECCV Workshops (2nd Workshop on Marine Vision)},
  year      = {2026},
}
```

## License

MIT (see `LICENSE`). The clipped public bathymetry rasters retain their upstream
terms (OGL-Canada for NONNA-10; US public domain for NCEI CUDEM).
