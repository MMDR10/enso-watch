#!/usr/bin/env python3
"""Control: 係 ENSO 振盪本身定真係非洲→WWV 傳導？
partial corr(AS_MJJ(Y), WWV(Y+10m) | DJF ONI(Y→Y+1)) —— 控制 ENSO 相位後睇個 +10m lead 仲有冇。"""
import json
import os
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# ── ONI：CPC detrend.nino3.4 月度（真數據）──
url = "https://www.cpc.ncep.noaa.gov/data/indices/detrend.nino34.ascii.txt"
raw = urllib.request.urlopen(url, timeout=30).read().decode()
oni = {}
for line in raw.splitlines():
    p = line.split()
    if len(p) == 5 and p[0].isdigit():
        oni[f"{int(p[0])}-{int(p[1]):02d}"] = float(p[4])
print(f"ONI months: {len(oni)} ({min(oni)}→{max(oni)})")

wwv = {}
for line in open(os.path.join(HERE, "wwv.dat")):
    p = line.split()
    if len(p) == 3 and p[0].isdigit() and len(p[0]) == 6:
        wwv[f"{p[0][:4]}-{p[0][4:]}"] = float(p[2])

d = np.load(os.path.join(HERE, "..", "africa_features_528.npz"), allow_pickle=True)
afr = {str(m): (float(a), float(b)) for m, a, b in zip(d["dates"], d["AS"], d["RHO"])}

years = range(1982, 2026)
as_mjj, wwv_l10, oni_djf, oni_mjj = [], [], [], []
for y in years:
    if not all(f"{y}-{m:02d}" in afr for m in (5, 6, 7)):
        continue
    as_mjj.append(np.mean([afr[f"{y}-{m:02d}"][0] for m in (5, 6, 7)]))
    oni_mjj.append(np.mean([oni[f"{y}-{m:02d}"] for m in (5, 6, 7)]))
    mo = 7 + 10
    yy, mm = y + (mo - 1) // 12, (mo - 1) % 12 + 1
    wwv_l10.append(wwv.get(f"{yy}-{mm:02d}", np.nan))
    djf = [oni.get(f"{y}-12"), oni.get(f"{y+1}-01"), oni.get(f"{y+1}-02")]
    oni_djf.append(np.mean([v for v in djf if v is not None]) if all(v is not None for v in djf) else np.nan)

as_mjj = np.array(as_mjj); wwv_l10 = np.array(wwv_l10)
oni_djf = np.array(oni_djf); oni_mjj = np.array(oni_mjj)
ok = ~(np.isnan(wwv_l10) | np.isnan(oni_djf))
as_mjj, wwv_l10, oni_djf, oni_mjj = as_mjj[ok], wwv_l10[ok], oni_djf[ok], oni_mjj[ok]
n = len(as_mjj)


def pcorr(x, y, z):
    """partial corr(x,y | z)"""
    zx = x - np.polyval(np.polyfit(z, x, 1), z)
    zy = y - np.polyval(np.polyfit(z, y, 1), z)
    return float(np.corrcoef(zx, zy)[0, 1])


r_raw = float(np.corrcoef(as_mjj, wwv_l10)[0, 1])
r_pdjf = pcorr(as_mjj, wwv_l10, oni_djf)          # 控制 DJF ONI
r_pmjj = pcorr(as_mjj, wwv_l10, oni_mjj)          # 控制同年 MJJ ONI
r_pdjf2 = pcorr(as_mjj, wwv_l10, oni_djf ** 2)    # 控制 |ENSO| 二次項

print(f"n={n}")
print(f"raw corr(AS_MJJ, WWV_Y+10m)          = {r_raw:+.3f}")
print(f"partial | DJF ONI(Y→Y+1)             = {r_pdjf:+.3f}")
print(f"partial | MJJ ONI(Y)                 = {r_pmjj:+.3f}")
print(f"partial | DJF ONI²                   = {r_pdjf2:+.3f}")

out = {"n": int(n), "raw_r": round(r_raw, 3), "partial_given_djf_oni": round(r_pdjf, 3),
       "partial_given_mjj_oni": round(r_pmjj, 3), "partial_given_djf_oni_sq": round(r_pdjf2, 3)}
json.dump(out, open(os.path.join(HERE, "africa_wwv_partial_control.json"), "w"), indent=1)
print("saved africa_wwv_partial_control.json")
