"""Hover depth-current fits, July-1 inventory -> captures/hovers_0701.csv (resumable).
Chart columns from the clipped rasters in data/rasters: consistency check, not truth.
Run from repo root; WVIO_FLIGHTS = raw flight dir."""
import json, csv, os, sys, time, numpy as np
from pathlib import Path
import rasterio
from pyproj import Transformer
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wvio import capture as cap
FLIGHTS=Path(os.environ.get("WVIO_FLIGHTS", "data/flights"))
hovers=json.load(open("captures/hovers_0701_worklist.json"))["hovers"]
for h in hovers: h["mp4"]=str(FLIGHTS/Path(h["mp4"]).name)
L=0.99
def opentif(p):
    try: ds=rasterio.open(p); return ds, ds.read(1).astype(float)
    except: return None,None
n10,v10=opentif("data/rasters/_lp_n10.tif"); ncei,vne=opentif("data/rasters/_lp_ncei.tif")
tfn=Transformer.from_crs(4326,n10.crs,always_xy=True) if n10 else None
tfe=Transformer.from_crs(4326,ncei.crs,always_xy=True) if ncei else None
def chart_depth(lat,lon):
    if n10:
        try:
            x,y=tfn.transform(lon,lat); r,c=n10.index(x,y)
            if 0<=r<n10.height and 0<=c<n10.width:
                val=v10[r,c]
                if np.isfinite(val) and val!=n10.nodata and abs(val)<1e4 and 0.1<L-val<60: return L-val,"NONNA10"
        except: pass
    if ncei:
        try:
            x,y=tfe.transform(lon,lat); r,c=ncei.index(x,y)
            if 0<=r<ncei.height and 0<=c<ncei.width and 0.1<-vne[r,c]<60: return float(-vne[r,c]),"NCEI"
        except: pass
    return None,"none"
out=Path("captures/hovers_0701.csv")
done=set()
if out.exists():
    for row in csv.reader(open(out)):
        if row and row[0]!="flight" and len(row)>2: done.add((row[0],row[1]))   # (flight, t0) -> resume
mode="a" if done else "w"; f=open(out,mode,newline=""); w=csv.writer(f)
if not done: w.writerow(["flight","t0","t1","alt_m","lat","lon","depth_m","kh","period_s","off_shell","current_ms","swell_deg","chart_depth_m","chart_src","period_match","verdict"]); f.flush()
print(f"processing {len(hovers)} hovers ({len(done)} already done -> resuming)...")
for i,h in enumerate(hovers):
    if (h["flight"],str(h["t0"])) in done:
        print(f"[{i+1}/{len(hovers)}] {h['flight']} {h['t0']:.0f}s already done, skip",flush=True); continue
    try:
        dur=min(h["t1"]-h["t0"],90); gsd=cap.gsd_for(h["alt"],640)
        cube=cap.extract_cube(h["mp4"],h["t0"],dur,640,4.0)
        r=cap.hover_depth(cube,gsd,fps=4.0,t_band=cap.HOVER_T_BAND)
        cd,src=chart_depth(h["lat"],h["lon"]); pm=bool(abs(r.period-3.5)<0.7)
        v=("DEGENERATE-weak" if r.off_shell>0.35 else "DEGENERATE-neardeep" if r.kh>1.8 else "borderline-shallow" if r.kh<0.25 else "USABLE")
        w.writerow([h["flight"],h["t0"],h["t1"],h["alt"],h["lat"],h["lon"],f"{r.depth:.2f}",f"{r.kh:.2f}",f"{r.period:.1f}",f"{r.off_shell:.2f}",f"{np.hypot(*r.current):.2f}",f"{np.degrees(r.swell_axis):.0f}",(f"{cd:.1f}" if cd else ""),src,pm,v]); f.flush()
        print(f"[{i+1}/{len(hovers)}] {h['flight']} {h['t0']:.0f}s {h['alt']:.0f}m: depth {r.depth:.2f} kh {r.kh:.2f} T {r.period:.1f} off {r.off_shell:.2f} chart {cd if cd else '-'} {src} -> {v}",flush=True)
    except Exception as e:
        w.writerow([h["flight"],h["t0"],h["t1"],h["alt"],h["lat"],h["lon"]]+[""]*8+["FAIL"]); f.flush()
        print(f"[{i+1}/{len(hovers)}] {h['flight']} {h['t0']:.0f}s FAIL: {e}",flush=True)
f.close(); print("DONE")
