#!/usr/bin/env python3
"""
ENSO Watch — 計算 OISST 月氣候 (climatology) 並存檔
====================================================
同 Phase 30/31 方法完全一致：
  clim[m] = nanmean(sst[月份==m], axis=0)   (m = 1..12)
  anom = sst - clim[month-1]

輸出: oisst_climatology.npy (12, 240, 680) float32
      oisst_clim_lat.npy / oisst_clim_lon.npy (座標)
      oisst_clim_nmonths.npy (每個月用咗幾多個月)

用 memmap 逐月累加，峰值 RAM ~ 300MB（唔會 OOM）。
"""

import numpy as np
from datetime import date

SRC = "/app/working/workspaces/tygtDc/data/sst/oisst_mon_1982_2025.npy"
LAT = "/app/working/workspaces/tygtDc/data/sst/oisst_mon_lat.npy"
LON = "/app/working/workspaces/tygtDc/data/sst/oisst_mon_lon.npy"
OUT_CLIM = "oisst_climatology.npy"
OUT_LAT = "oisst_clim_lat.npy"
OUT_LON = "oisst_clim_lon.npy"
OUT_N = "oisst_clim_nmonths.npy"

def main():
    lat = np.load(LAT)
    lon = np.load(LON)
    sst = np.load(SRC, mmap_mode="r")  # (528, 240, 680)
    n_t, n_lat, n_lon = sst.shape
    print(f"OISST: {n_t} months, {n_lat}x{n_lon} grid")

    # 月份陣列: 1982-01 開始
    y0, m0 = 1982, 1
    months = []
    for i in range(n_t):
        y = y0 + (m0 - 1 + i) // 12
        m = (m0 - 1 + i) % 12 + 1
        months.append(m)
    months = np.array(months)

    # 逐月累加 nanmean（用 float64 累加避免精度問題）
    clim = np.full((12, n_lat, n_lon), np.nan, dtype=np.float32)
    counts = np.zeros(12, dtype=np.int32)
    for m in range(1, 13):
        idx = np.where(months == m)[0]
        counts[m - 1] = len(idx)
        # 逐 chunk 載入（每次 12 個月左右）減少 RAM
        sums = None
        ns = None
        for start in range(0, len(idx), 12):
            chunk = np.asarray(sst[idx[start:start + 12]], dtype=np.float64)
            with np.errstate(invalid="ignore"):
                chunk_sum = np.nansum(chunk, axis=0)
                chunk_n = np.sum(~np.isnan(chunk), axis=0)
            sums = chunk_sum if sums is None else sums + chunk_sum
            ns = chunk_n if ns is None else ns + chunk_n
        with np.errstate(invalid="ignore"):
            mean = np.where(ns > 0, sums / np.maximum(ns, 1), np.nan)
        clim[m - 1] = mean.astype(np.float32)
        print(f"  month {m:2d}: n={counts[m-1]:3d}  valid%={np.mean(~np.isnan(mean))*100:.1f}")

    np.save(OUT_CLIM, clim)
    np.save(OUT_LAT, lat)
    np.save(OUT_LON, lon)
    np.save(OUT_N, counts)
    print(f"\n✅ 存檔: {OUT_CLIM} ({clim.shape}, {clim.nbytes/1e6:.1f} MB)")
    print(f"        {OUT_LAT} {OUT_LON} {OUT_N}")
    print(f"        每月樣本數: {counts.tolist()}")

if __name__ == "__main__":
    main()
