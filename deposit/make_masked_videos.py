"""Build the privacy-masked deposit copies of the raw flight videos.

Frames outside each flight's consumed analysis envelope (the [first, last]
window any release script reads) are blacked; the envelope itself is stream-
copied bit-for-bit, so every committed t0 lands on identical pixels. Flight
0003 additionally gets a land-band mask during 17.5-36 s, where the operator
is visible at the treeline inside the first anchor window; the effect of that
mask on the committed numbers is measured by deposit/verify_masked.py.
Cut points snap to keyframes (~0.5 s apart) just outside the envelope.
Audio: none in the sources. DJI aux data streams are dropped; the .SRT
sidecars are published unmodified.

Usage: python3 deposit/make_masked_videos.py [key ...]   (default: all)
Env: WVIO_FLIGHTS (raw July-1 dir), WVIO_CACHE_0628, DEPOSIT_OUT (staging dir).
"""
import json, os, subprocess, sys
from pathlib import Path

D = Path(os.environ.get("WVIO_FLIGHTS", "/mnt/c/Users/james/DJI July 1s"))
CACHE = Path(os.environ.get("WVIO_CACHE_0628",
             "/mnt/c/Users/james/Downloads/dji_fly_20260628_140048_cache.mp4"))
OUT = Path(os.environ.get("DEPOSIT_OUT", str(Path.home() / "deposit_staging")))

# consumed envelope [lo, hi] in video seconds (see deposit/README.md for provenance)
SPEC = {
    "0001": dict(lo=35.5, hi=1180.2),
    "0003": dict(lo=20.0, hi=298.0,
                 band=(17.5, 36.0, 0.25, 0.85, 0.68)),  # blur t0,t1, box x0,x1,y0 (fracs)
    "0004": dict(lo=39.5, hi=609.1),
    "0005": dict(lo=25.0, hi=93.0),
    "0006": dict(lo=23.0, hi=315.8),
    "cache": dict(lo=113.0, hi=1445.4),
}

# The July-1 flights are HEVC; the June-28 app cache is H.264. Masked segments
# must be encoded in the source's codec or they cannot concat with the copied
# middle, so encoder, stream tag and Annex B filter all follow the probed codec.
CODECS = {
    "hevc": dict(ext="h265", tag="hvc1",
                 enc=["-c:v", "libx265", "-pix_fmt", "yuv420p10le",
                      "-x265-params", "repeat-headers=1:log-level=error"]),
    "h264": dict(ext="h264", tag="avc1",
                 enc=["-c:v", "libx264", "-pix_fmt", "yuv420p",
                      "-x264-params", "repeat-headers=1:log-level=error"]),
}


def run(*cmd):
    subprocess.run(list(cmd), check=True)


def vinfo(src):
    """Codec and the rate the copied middle is re-timestamped at.

    That rate must be the average, not r_frame_rate: the June-28 cache reports
    60/1 but truly runs at 60.0024, and stamping it at 60 drifts the tail of
    the file ~5 frames late. The DJI sources report both as 60000/1001.
    """
    p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=codec_name,r_frame_rate,avg_frame_rate",
                        "-of", "default=noprint_wrappers=1", str(src)],
                       capture_output=True, text=True, check=True)
    d = dict(l.split("=", 1) for l in p.stdout.strip().splitlines())
    codec = d["codec_name"]
    if codec not in CODECS:
        raise SystemExit(f"{src.name}: unsupported codec {codec}")
    avg = d.get("avg_frame_rate", "")
    return codec, avg if avg not in ("", "0/0", "0/1") else d["r_frame_rate"]


def probe_keyframes(src, around, span=6.0):
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-skip_frame", "nokey", "-select_streams", "v:0",
         "-show_entries", "frame=pts_time", "-of", "csv=p=0",
         "-read_intervals", f"{max(0.0, around - span)}%+{2 * span}", str(src)],
        capture_output=True, text=True, check=True)
    return [float(x) for x in p.stdout.split() if x]


def kf_before(src, t):
    ks = [k for k in probe_keyframes(src, t) if k <= t]
    if not ks:
        return 0.0
    return ks[-1]


def kf_after(src, t):
    ks = [k for k in probe_keyframes(src, t) if k >= t]
    return ks[0] if ks else None


def duration(src):
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", str(src)], capture_output=True, text=True, check=True)
    return float(p.stdout.strip())


def build(key):
    src = CACHE if key == "cache" else next(D.glob(f"*_{key}_D.MP4"))
    spec = SPEC[key]
    codec, rate = vinfo(src)
    cdc = CODECS[codec]
    # Every segment is pinned to one timescale: the concat demuxer takes the
    # output timebase from the first input, and a copied middle on a different
    # one has its packet durations misread (x264 defaults to 1/15360, not
    # 1/60000), which silently stretches the joined file.
    ENC = [*cdc["enc"], "-preset", "fast", "-tag:v", cdc["tag"],
           "-video_track_timescale", "60000"]
    dur = duration(src)
    kA = kf_before(src, spec["lo"] - 0.3)
    kB = kf_after(src, spec["hi"] + 0.3) or dur
    seg_dir = OUT / f"segs_{key}"
    seg_dir.mkdir(parents=True, exist_ok=True)
    segs = []
    crf = ["-crf", "18"]
    black = "drawbox=x=0:y=0:w=iw:h=ih:color=black:t=fill"

    if kA > 0:
        head = seg_dir / "head.mp4"
        run("ffmpeg", "-v", "error", "-y", "-i", src, "-t", f"{kA:.6f}",
            "-map", "0:v:0", "-vf", black, *ENC, *crf, "-an", head)
        segs.append(head)

    copy_from = kA
    if "band" in spec:              # 0003: blur the operator region, near-lossless
        b0, b1, x0, x1, y0 = spec["band"]
        kM = kf_after(src, b1)
        band = seg_dir / "band.mp4"
        w, h = f"trunc(iw*{x1 - x0})", f"trunc(ih*{1 - y0})"
        x, y = f"trunc(iw*{x0})", f"trunc(ih*{y0})"
        blur = (f"split[a][b];[b]crop={w}:{h}:{x}:{y},boxblur=24:3[c];"
                f"[a][c]overlay=trunc(W*{x0}):trunc(H*{y0})")
        run("ffmpeg", "-v", "error", "-y", "-ss", f"{kA:.6f}", "-i", src,
            "-t", f"{kM - kA:.6f}", "-map", "0:v:0", "-vf", blur,
            *ENC, "-crf", "8", "-an", band)
        segs.append(band)
        copy_from = kM

    # copied middle: round-trip through Annex B so every IDR carries SPS/PPS
    # in-band (the concat output's extradata comes from the x265 head segment)
    mid_es = seg_dir / f"mid.{cdc['ext']}"
    run("ffmpeg", "-v", "error", "-y", "-ss", f"{copy_from:.6f}", "-i", src,
        "-t", f"{kB - copy_from:.6f}", "-map", "0:v:0", "-c", "copy",
        "-bsf:v", f"{codec}_mp4toannexb", "-f", codec, mid_es)
    mid = seg_dir / "mid_ab.mp4"
    run("ffmpeg", "-v", "error", "-y", "-fflags", "+genpts", "-r", rate,
        "-i", mid_es, "-c", "copy", "-tag:v", cdc["tag"],
        "-video_track_timescale", "60000", mid)
    segs.append(mid)

    if kB < dur - 0.05:
        tail = seg_dir / "tail.mp4"
        run("ffmpeg", "-v", "error", "-y", "-ss", f"{kB:.6f}", "-i", src,
            "-map", "0:v:0", "-vf", black, *ENC, *crf, "-an", tail)
        segs.append(tail)

    lst = seg_dir / "concat.txt"
    lst.write_text("".join(f"file '{s.resolve()}'\n" for s in segs))
    out = OUT / (src.name if key != "cache" else "dji_fly_20260628_140048_cache.mp4")
    run("ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst,
        "-c", "copy", "-movflags", "+faststart", out)
    meta = dict(key=key, src=str(src), out=str(out), dur=dur, kA=kA, kB=kB,
                band=spec.get("band"), segs=[str(s) for s in segs])
    (OUT / f"mask_{key}.json").write_text(json.dumps(meta, indent=1))
    print(f"{key}: masked [0,{kA:.3f}) + [{kB:.3f},{dur:.3f}) -> {out}", flush=True)


if __name__ == "__main__":
    keys = sys.argv[1:] or list(SPEC)
    OUT.mkdir(parents=True, exist_ok=True)
    for k in keys:
        build(k)
    print("ALL DONE")
