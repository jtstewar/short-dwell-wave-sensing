# Raw flight video deposit — provenance and masking

The archived deposit contains the raw drone video behind every number in the paper,
with a privacy mask applied outside the analysis envelope. This file documents
exactly what was changed, how, and what was verified, so the deposit can be treated
as evidence rather than trusted blindly.

## Files

| deposit file | source | consumed envelope (video s) | masked |
|---|---|---|---|
| `DJI_20260701155928_0001_D.MP4` | flight 0001 (F1) | 35.5 – 1180.2 | head + tail black |
| `DJI_20260701163111_0003_D.MP4` | flight 0003 | 20.0 – 298.0 | head + tail black; blur box 17.5–36 s |
| `DJI_20260701164025_0004_D.MP4` | flight 0004 (F3) | 39.5 – 609.1 | head + tail black |
| `DJI_20260701165321_0005_D.MP4` | flight 0005 (F4) | 25.0 – 93.0 | head + tail black |
| `DJI_20260701170749_0006_D.MP4` | flight 0006 (F5) | 23.0 – 315.8 | head black |
| `dji_fly_20260628_140048_cache.mp4` | June-28 Port Dover app cache | 113.0 – 1445.4 | head + tail black |
| `*.SRT` | DJI subtitle sidecars | — | unmodified |

The consumed envelope is the union of every window any release script reads
(hover/crawl worklist, the Table-1 anchor sections including their altitude-gate
skip rule, the glint and figure windows, and the June-28 stacking clips manifest).

## Masking method

Frames outside the envelope are replaced with black; the envelope itself is
**stream-copied** — bit-identical to the camera originals — so every committed
`t0` lands on identical pixels, and durations and frame counts match the originals
exactly. Cut points snap to keyframes (~0.5 s) just outside the envelope.

The masked segments are re-encoded (libx265) and joined to the copied middle at
the container level; the copied segment is passed through Annex B so its parameter
sets travel in-band (the concatenated file's extradata comes from the re-encoded
head). Players that require out-of-band-only HEVC headers may glitch across the
boundary; ffmpeg-based decoding (which all release scripts use) is verified clean.
DJI auxiliary data streams are dropped; the sources carry no audio.

**Flight 0003 exception.** The drone operator is faintly visible at the beach
treeline inside the first Table-1 anchor window (t0=20 s), which is committed
evidence and cannot be blacked. That region (x 25–85 %, lower 32 % of frame) is
heavily blurred during 17.5–36 s and the segment re-encoded near-lossless
(crf 8). Water pixels used by the estimator are otherwise untouched, but this
one anchor is *not* bit-identical in the deposit.

**June-28 cache exception.** That file is H.264 1080p60 — the DJI app's own
transcode — not HEVC like the flight originals, so its masked segments are
x264-encoded and its Annex B round-trip uses the H.264 filter. Its timing is
also mildly variable: it declares 60/1 but averages 60.0024 fps, and the copied
middle is re-stamped at that average. The deposit's frame count matches the
original exactly (88 511); its duration runs 29 ms — 1.7 frames — long, and the
timing offset inside the consumed envelope stays under one frame. Pixels in the
envelope are still stream-copied and bit-identical.

## Verification (`verify_masked.py`)

For every file: frame count equals the original, and duration equals it to
within one frame (exactly, except the June-28 cache noted above); a mid-envelope
analysis cube extracted from the deposit is byte-identical to the original;
masked head/tail frames decode as pure black.

For the 0003 anchor: re-running the MVDR estimator on the **original** file
reproduces the committed `supp/mvdr_perwindow.csv` rows exactly (2.774 / 2.442 /
2.065 / 1.483 m/s at 8/6/4/3 s). On the **masked** file the same windows give
2.762 / 2.470 / 2.021 / 1.418 m/s — within 0.07 m/s of the committed values;
the difference is the blur box plus re-encode. All other per-window rows
regenerate from bit-identical pixels.

The deposit ships a `SHA256SUMS` manifest covering the videos, the per-build
mask records, and the SRT sidecars.

## Using the deposit

Point the release scripts at the deposit directory:

```bash
WVIO_FLIGHTS=/path/to/deposit python3 scripts/mvdr_perwindow_dump.py
WVIO_CACHE_0628=/path/to/deposit/dji_fly_20260628_140048_cache.mp4 \
  python3 scripts/dover_stack_test.py
```

`make_masked_videos.py` (the mask builder) and `verify_masked.py` are in this
directory; the per-file mask parameters are in `make_masked_videos.py::SPEC` and
recorded per build in the deposit's `mask_*.json`.
