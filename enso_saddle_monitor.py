#!/usr/bin/env python3
"""
ENSO Watch — Saddle Monitor（鞍點結構監測）
============================================
將鞍點結構測量（Phase 30-35 方法）加入 ENSO Watch 自動監測：
每月下載最新 OISST SST → 減月氣候 (climatology) → 2D Hessian/Morse 鞍點分類
→ 輸出鞍點結構指標。

監測指標（基於已驗證發現）：
  1. 全場鞍點比例（Phase 30：ENSO 場級 ≈56%，相位無敏感 — 做基線）
  2. 鞍點密度 × 雜訊密度 ρ 場相關（Phase 31：格點 mask r≈0.42，null z=85）
     → 鞍點聚集喺鋒面帶（ITCZ/冷舌）嘅強度
  3. 鞍點密度 × |SST 梯度| 相關（Phase 31 補充：鋒面依附）
  4. 東端 (270-290E) vs Nino3.4 中心鞍點密度（Phase 32：兩端記憶強、中心弱）

重要：Morse 分類用 SST anomaly（減 climatology），唔係 raw SST —
  Phase 31 驗證：raw SST 會俾出 saddle 51%/ρ corr 0.14（失真），
  anomaly 先係 56%/0.42（Phase 30/31 baseline）。
  Climatology 用 1982-2025 全期 528 月月氣候（compute_climatology.py 計出）。

數據：NOAA OISST v2.1 0.25° 月度（OPeNDAP，免 key）
輸出：saddle_data.json（供 dashboard）+ saddle_history.json（時間序列收集）

Dependencies: numpy, scipy, netCDF4（GitHub Actions: pip install numpy scipy netCDF4）
"""

import json, sys, os
from datetime import datetime, timezone
import numpy as np

# ─── Config ───
OISST_URL = "https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres/sst.mon.mean.nc"
CLIM_FILE = "oisst_climatology.npy"   # (12, 240, 680) 月氣候（同 repo 一齊 commit）
CLIM_LAT_FILE = "oisst_clim_lat.npy"
CLIM_LON_FILE = "oisst_clim_lon.npy"
OUTPUT_JSON = "saddle_data.json"
HISTORY_JSON = "saddle_history.json"
# 熱帶太平洋（同 Phase 30-35 一致）
LAT_S, LAT_N = -30.0, 30.0
LON_W, LON_E = 120.0, 290.0
# 東端（週期記憶最強區）vs Nino3.4（中心）
EAST_LON = (270.0, 290.0)
NINO34_LON = (190.0, 240.0)
EQ_LAT = (-5.0, 5.0)


def load_climatology():
    """載入月氣候 (12, nlat, nlon) + 座標；檢查同 monitor 範圍一致"""
    if not os.path.exists(CLIM_FILE):
        print(f"  ❌ 搵唔到 {CLIM_FILE} — 請先跑 compute_climatology.py 或確認 repo 有呢個檔")
        sys.exit(1)
    clim = np.load(CLIM_FILE)
    clim_lat = np.load(CLIM_LAT_FILE)
    clim_lon = np.load(CLIM_LON_FILE)
    return clim, clim_lat, clim_lon


def fetch_latest_sst():
    """從 OPeNDAP 攞最新月 OISST SST（熱帶太平洋區域）"""
    import netCDF4
    ds = netCDF4.Dataset(OISST_URL)
    lat_all = ds.variables['lat'][:]
    lon_all = ds.variables['lon'][:]
    time_v = ds.variables['time']
    nt = len(time_v)
    
    # 攞最新完整月（最後一個月）
    t_idx = nt - 1
    
    # 區域索引
    lat_idx = np.where((lat_all >= LAT_S) & (lat_all <= LAT_N))[0]
    lon_idx = np.where((lon_all >= LON_W) & (lon_all <= LON_E))[0]
    
    # 讀數據（sst[time, lat, lon]）
    sst = ds.variables['sst'][t_idx, lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1]
    lat = lat_all[lat_idx]
    lon = lon_all[lon_idx]
    
    # 時間戳（days since 1800-01-01）
    t0 = datetime(1800, 1, 1)
    from datetime import timedelta
    date = t0 + timedelta(days=float(time_v[t_idx]))
    ds.close()
    return np.ma.filled(sst, np.nan), lat, lon, date


def hessian2d(f, dy, dx):
    fy = np.gradient(f, dy, axis=0)
    fx = np.gradient(f, dx, axis=1)
    fyy = np.gradient(fy, dy, axis=0)
    fxx = np.gradient(fx, dx, axis=1)
    fxy = np.gradient(fx, dy, axis=0)
    return fxx, fyy, fxy


def morse_classify_2d(field, dy, dx, parabolic_frac=0.10):
    """2D Morse 分類（同 Phase 30-35 完全一致）"""
    valid = ~np.isnan(field)
    fxx, fyy, fxy = hessian2d(field, dy, dx)
    det = fxx * fyy - fxy**2
    trace = fxx + fyy
    adet = np.abs(det)
    thr = np.nanpercentile(adet[valid], parabolic_frac * 100) if valid.any() else 0.0
    parabolic = (~valid) | (adet < thr)
    saddle = (~parabolic) & (det < 0)
    peak = (~parabolic) & (det > 0) & (trace < 0)
    trough = (~parabolic) & (det > 0) & (trace > 0)
    return {'saddle': saddle, 'peak': peak, 'trough': trough,
            'parabolic': parabolic, 'valid': valid}


def saddle_density_map(saddle, valid, box=9):
    from scipy.ndimage import uniform_filter
    d = uniform_filter(saddle.astype(float), size=box, mode='constant')
    v = uniform_filter(valid.astype(float), size=box, mode='constant')
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(v > 0, d / np.maximum(v, 1e-9), np.nan)


def noise_density(field, sigma=2.0):
    """雜訊密度 ρ = |f - smooth(f)|（高斯平滑殘差，Phase 31 定義）"""
    from scipy.ndimage import gaussian_filter
    smooth = gaussian_filter(np.where(np.isnan(field), np.nanmean(field), field), sigma)
    rho = np.abs(field - smooth)
    rho[np.isnan(field)] = np.nan
    return rho


def region_mean(field, valid, lat, lon, lat_r, lon_r):
    m = valid & (lat[:, None] >= lat_r[0]) & (lat[:, None] <= lat_r[1]) & \
        (lon[None, :] >= lon_r[0]) & (lon[None, :] <= lon_r[1])
    if m.sum() == 0:
        return None
    return float(np.nanmean(field[m]))


def main():
    print("=" * 60)
    print("ENSO Watch — Saddle Monitor")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    # 1. Fetch
    print("\n[1/4] Fetching OISST latest month...")
    try:
        sst, lat, lon, date = fetch_latest_sst()
    except Exception as e:
        print(f"  ❌ Fetch failed: {e}")
        sys.exit(1)
    print(f"  OK: {date.strftime('%Y-%m')}, grid {len(lat)}x{len(lon)}")

    # 1b. 減月氣候 → SST anomaly（Phase 30/31 方法）
    print("\n[1b] Computing SST anomaly (minus monthly climatology)...")
    clim, clim_lat, clim_lon = load_climatology()
    # 驗證座標一致
    if len(clim_lat) != len(lat) or len(clim_lon) != len(lon):
        print(f"  ❌ climatology 座標 ({len(clim_lat)}x{len(clim_lon)}) 同 OISST 當月 ({len(lat)}x{len(lon)}) 唔一致")
        sys.exit(1)
    month = date.month
    anom = sst - clim[month - 1]
    anom[np.isnan(sst)] = np.nan   # 陸地 mask 保留
    # 季節性診斷：climatology 覆蓋率
    clim_valid = np.mean(~np.isnan(clim[month - 1])) * 100
    print(f"  month={month}, clim valid%={clim_valid:.1f}, anom range=[{np.nanmin(anom):.2f}, {np.nanmax(anom):.2f}]")

    # 2. Morse 分類（用 anomaly 場）
    print("\n[2/4] Morse classification (anomaly field)...")
    dlat = abs(lat[1] - lat[0]) if len(lat) > 1 else 1.0
    dlon = abs(lon[1] - lon[0]) if len(lon) > 1 else 1.0
    # 轉成以度為單位（梯度唔理單位，Morse 分類只睇符號）
    M = morse_classify_2d(anom, dlat, dlon)
    # 分母 = 非 parabolic 嘅 valid 點（Phase 30 慣例：saddle / (valid & ~parabolic)）
    denom = M['valid'] & ~M['parabolic']
    tot = denom.sum()
    saddle_frac = M['saddle'][denom].sum() / tot
    peak_frac = M['peak'][denom].sum() / tot
    trough_frac = M['trough'][denom].sum() / tot
    print(f"  denom(valid&~para)={tot:,.0f}")
    print(f"  saddle={saddle_frac*100:.2f}% peak={peak_frac*100:.2f}% trough={trough_frac*100:.2f}%")

    # 3. 鞍點密度 × ρ / |∇T| 相關（anomaly 場）
    print("\n[3/4] Saddle density correlations...")
    rho = noise_density(anom)
    # SST 梯度幅值
    gy, gx = np.gradient(np.where(np.isnan(anom), np.nan, anom), dlat, dlon)
    grad = np.hypot(gx, gy)
    grad[np.isnan(anom)] = np.nan
    rho[np.isnan(anom)] = np.nan

    density = saddle_density_map(M['saddle'], M['valid'])
    valid2 = M['valid'] & ~np.isnan(density) & ~np.isnan(rho) & ~np.isnan(grad)

    if valid2.sum() > 1000:
        from scipy.stats import spearmanr
        r_rho, _ = spearmanr(rho[valid2], density[valid2])
        r_grad, _ = spearmanr(grad[valid2], density[valid2])
        print(f"  spearman(density, ρ)    = {r_rho:.4f} (n={valid2.sum():,})")
        print(f"  spearman(density, |∇T|) = {r_grad:.4f}")
    else:
        r_rho = r_grad = None
        print("  ⚠️ too few valid points")

    # 4. 區域鞍點密度（東端 vs Nino3.4）
    print("\n[4/4] Regional saddle density...")
    east_dens = region_mean(density, M['valid'], lat, lon, EQ_LAT, EAST_LON)
    nino34_dens = region_mean(density, M['valid'], lat, lon, EQ_LAT, NINO34_LON)
    print(f"  east_end (270-290E)  : {east_dens if east_dens is None else round(east_dens, 4)}")
    print(f"  nino34  (190-240E)   : {nino34_dens if nino34_dens is None else round(nino34_dens, 4)}")

    # 5. Output
    print("\n[5/5] Writing JSON...")
    data = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "source": "NOAA OISST v2.1 (0.25°) via OPeNDAP",
        "field": "SST anomaly (minus 1982-2025 monthly climatology)",
        "month": date.strftime("%Y-%m"),
        "grid": {"nlat": len(lat), "nlon": len(lon),
                 "lat_range": [LAT_S, LAT_N], "lon_range": [LON_W, LON_E]},
        "morse": {
            "saddle_frac": round(float(saddle_frac), 5),
            "peak_frac": round(float(peak_frac), 5),
            "trough_frac": round(float(trough_frac), 5),
            "n_valid": int(tot),
        },
        "correlations": {
            "spearman_density_rho": round(float(r_rho), 4) if r_rho is not None else None,
            "spearman_density_grad": round(float(r_grad), 4) if r_grad is not None else None,
        },
        "regional_density": {
            "east_end": round(float(east_dens), 5) if east_dens is not None else None,
            "nino34": round(float(nino34_dens), 5) if nino34_dens is not None else None,
        },
        "baselines": {
            "saddle_frac_enso_field": 0.564,  # Phase 30: ENSO 場級
            "spearman_density_rho_phase31": 0.42,  # Phase 31: 格點 mask
        },
    }

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  ✅ {OUTPUT_JSON}")

    # 6. History append（同月已有記錄 → 覆蓋更新，確保 anomaly 版數值入 history）
    if os.path.exists(HISTORY_JSON):
        with open(HISTORY_JSON) as f:
            hist = json.load(f)
    else:
        hist = {"records": []}

    new_rec = {
        "month": data["month"],
        "saddle_frac": data["morse"]["saddle_frac"],
        "spearman_density_rho": data["correlations"]["spearman_density_rho"],
        "spearman_density_grad": data["correlations"]["spearman_density_grad"],
        "east_end_density": data["regional_density"]["east_end"],
        "nino34_density": data["regional_density"]["nino34"],
        "field": "sst_anomaly",   # 標記：anomaly 版（raw SST 版係 0.51x，唔同 baseline）
    }
    existing = [r for r in hist["records"] if r.get("month") == data["month"]]
    if existing:
        # 覆蓋（保留最新）
        hist["records"] = [r for r in hist["records"] if r.get("month") != data["month"]]
        hist["records"].append(new_rec)
        print(f"  🔄 {data['month']} updated (anomaly version)")
    else:
        hist["records"].append(new_rec)
        print(f"  ✅ {HISTORY_JSON} appended ({len(hist['records'])} records)")
    hist["records"].sort(key=lambda r: r["month"])
    # 保持最多 120 個月
    if len(hist["records"]) > 120:
        hist["records"] = hist["records"][-120:]
    with open(HISTORY_JSON, 'w') as f:
        json.dump(hist, f, indent=2, ensure_ascii=False)

    print("\n🎯 Saddle Monitor done.")


if __name__ == '__main__':
    main()
