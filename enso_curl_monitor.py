#!/usr/bin/env python3
"""
ENSO Watch — Wind Stress Curl Monitor（氣流-海面交界面鞍點監測）
==================================================================
將 Phase 35 方法（ENSO 氣流-海面交界面 wind stress curl 鞍點測試）加入
ENSO Watch 自動監測：每月下載最新月 ERA5 10m u/v 風 → bulk wind stress
τ = ρ_a C_D |U| U → 球面 curl τ → 2D Morse 鞍點分類 → 輸出交界面鞍點指標。

監測指標（基於已驗證發現）：
  1. curl τ 全場鞍點比例（Phase 35：相位無敏感 ~57.5%）
  2. 鞍點密度 × |curl τ| 空間相關（Phase 35b：格點 mask r≈0.24-0.28，
     null r≈0.004，z=+141~+222）→ 鞍點聚集喺氣流-海面耦合帶（ITCZ/冷舌鋒面）

數據：ERA5 reanalysis-era5-single-levels（CDS API，key 由環境變數提供）
  CDS_URL + CDS_KEY（GitHub Actions secrets 傳入）
輸出：curl_data.json（供 dashboard）+ curl_history.json（時間序列收集）

Dependencies: numpy, scipy, xarray, cdsapi
"""

import json, sys, os
from datetime import datetime, timezone
import numpy as np

# ─── Config ───
RHO_A = 1.225   # kg/m^3 空氣密度
CD = 1.3e-3     # bulk drag coefficient（Phase 35 一致）
OUTPUT_JSON = "curl_data.json"
HISTORY_JSON = "curl_history.json"
# 熱帶太平洋（同 Phase 35 一致）
AREA = [30.0, 120.0, -30.0, 290.0]   # N, W, S, E
GRID = [0.5, 0.5]
# 東端 vs Nino3.4（同 saddle monitor 一致）
EAST_LON = (270.0, 290.0)
NINO34_LON = (190.0, 240.0)
EQ_LAT = (-5.0, 5.0)


def fetch_latest_wind(year, month):
    """用 CDS API 下載指定年月嘅 ERA5 10m u/v（每月 1 號 00:00，同 Phase 35 一致）"""
    import cdsapi
    url = os.environ.get("CDS_URL", "https://cds.climate.copernicus.eu/api")
    key = os.environ.get("CDS_KEY", "")
    if not key:
        print("  ❌ 冇 CDS_KEY 環境變數 — 請喺 GitHub repo secrets 設定（CDS_URL + CDS_KEY）")
        sys.exit(1)
    c = cdsapi.Client(url=url, key=key)
    m2 = f"{month:02d}"
    out = {}
    for var, name in [('10m_u_component_of_wind', 'u10'),
                      ('10m_v_component_of_wind', 'v10')]:
        fn = f"era5_{name}_{year}{m2}.nc"
        if os.path.exists(fn) and os.path.getsize(fn) > 100_000:
            print(f"  ⏭️ {fn} 已存在")
            out[name] = fn
            continue
        print(f"  ⬇️ 下載 {name} {year}-{m2}...")
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': [var],
                'year': [str(year)],
                'month': [m2],
                'day': ['01'],
                'time': '00:00',
                'area': AREA,
                'grid': GRID,
                'format': 'netcdf',
            },
            fn)
        out[name] = fn
        print(f"  ✅ {fn}")
    return out


def load_wind(files):
    """載入 u10/v10 netCDF → 2D 場 + 座標"""
    import xarray as xr
    u = xr.open_dataset(files['u10'])['u10'].values.squeeze()
    v = xr.open_dataset(files['v10'])['v10'].values.squeeze()
    ds = xr.open_dataset(files['u10'])
    lat = ds['latitude'].values
    lon = ds['longitude'].values
    return u, v, lat, lon


def wind_stress(u, v):
    """Bulk formula: τ = ρ_a C_D |U| U（Phase 35 一致）"""
    speed = np.hypot(u, v)
    tau_x = RHO_A * CD * speed * u
    tau_y = RHO_A * CD * speed * v
    return tau_x, tau_y


def curl_sphere(tau_x, tau_y, lat, lon):
    """球面 curl τ = ∂τy/∂x - ∂τx/∂y（單位 Pa/m，標準化後用）"""
    R = 6371e3
    dphi = np.radians(np.abs(lat[1] - lat[0]))
    dlam = np.radians(np.abs(lon[1] - lon[0]))
    phi = np.radians(lat)
    dtauy_dx = np.gradient(tau_y, dlam, axis=1) / (R * np.cos(phi[:, None]))
    dtautx_dy = np.gradient(tau_x, dphi, axis=0) / R
    c = dtauy_dx - dtautx_dy
    return c


def hessian2d(f, spacing):
    fy = np.gradient(f, spacing, axis=0)
    fx = np.gradient(f, spacing, axis=1)
    fyy = np.gradient(fy, spacing, axis=0)
    fxx = np.gradient(fx, spacing, axis=1)
    fxy = np.gradient(fx, spacing, axis=0)
    return fxx, fyy, fxy


def morse_classify_2d(field, spacing, parabolic_frac=0.10):
    """2D Morse 分類（同 Phase 30/35 完全一致）"""
    valid = ~np.isnan(field)
    fxx, fyy, fxy = hessian2d(field, spacing)
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


def region_mean(field, valid, lat, lon, lat_r, lon_r):
    m = valid & (lat[:, None] >= lat_r[0]) & (lat[:, None] <= lat_r[1]) & \
        (lon[None, :] >= lon_r[0]) & (lon[None, :] <= lon_r[1])
    if m.sum() == 0:
        return None
    return float(np.nanmean(field[m]))


def main():
    print("=" * 60)
    print("ENSO Watch — Wind Stress Curl Monitor")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    # 最新月（跑嗰時減 1 個月，確保 ERA5 已出）
    now = datetime.now(timezone.utc)
    y, m = now.year, now.month
    if m == 1:
        y, m = y - 1, 12
    else:
        m -= 1
    print(f"Target month: {y}-{m:02d}")

    # 1. Fetch
    print("\n[1/4] Fetching ERA5 10m wind...")
    try:
        files = fetch_latest_wind(y, m)
        u, v, lat, lon = load_wind(files)
    except Exception as e:
        print(f"  ❌ Fetch failed: {e}")
        sys.exit(1)
    print(f"  OK: grid {lat.shape[0]}x{lon.shape[0]}, u range [{np.nanmin(u):.2f}, {np.nanmax(u):.2f}]")

    # 2. Wind stress curl
    print("\n[2/4] Computing wind stress curl...")
    tx, ty = wind_stress(u, v)
    curl_t = curl_sphere(tx, ty, lat, lon)
    # 標準化（Phase 35 一致）
    curl_t = (curl_t - np.nanmean(curl_t)) / np.nanstd(curl_t)
    spacing = abs(lat[1] - lat[0]) if len(lat) > 1 else 1.0
    M = morse_classify_2d(curl_t, spacing)
    denom = M['valid'] & ~M['parabolic']
    tot = denom.sum()
    saddle_frac = M['saddle'][denom].sum() / tot
    peak_frac = M['peak'][denom].sum() / tot
    trough_frac = M['trough'][denom].sum() / tot
    print(f"  curl τ saddle={saddle_frac*100:.2f}% peak={peak_frac*100:.2f}% trough={trough_frac*100:.2f}% (n={tot:,})")

    # 3. 鞍點密度 × |curl τ| 相關（Phase 35b 格點 mask 方法）
    print("\n[3/4] Saddle density × |curl τ| correlation...")
    mag = np.abs(curl_t)
    density = saddle_density_map(M['saddle'], M['valid'])
    valid2 = M['valid'] & ~np.isnan(density) & ~np.isnan(mag)
    from scipy.stats import spearmanr
    if valid2.sum() > 1000:
        r_curl, p_curl = spearmanr(mag[valid2], density[valid2])
        print(f"  spearman(density, |curl τ|) = {r_curl:.4f} (n={valid2.sum():,}, p={p_curl:.2g})")
    else:
        r_curl, p_curl = None, None
        print("  ⚠️ too few valid points")

    # 4. 區域鞍點密度
    print("\n[4/4] Regional saddle density...")
    east_dens = region_mean(density, M['valid'], lat, lon, EQ_LAT, EAST_LON)
    nino34_dens = region_mean(density, M['valid'], lat, lon, EQ_LAT, NINO34_LON)
    print(f"  east_end (270-290E)  : {east_dens if east_dens is None else round(east_dens, 4)}")
    print(f"  nino34  (190-240E)   : {nino34_dens if nino34_dens is None else round(nino34_dens, 4)}")

    # 5. Output
    print("\n[5/5] Writing JSON...")
    data = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "source": "ERA5 reanalysis single-levels via CDS API",
        "field": "wind stress curl τ (bulk formula, standardized)",
        "month": f"{y}-{m:02d}",
        "grid": {"nlat": len(lat), "nlon": len(lon),
                 "area": AREA},
        "morse": {
            "saddle_frac": round(float(saddle_frac), 5),
            "peak_frac": round(float(peak_frac), 5),
            "trough_frac": round(float(trough_frac), 5),
            "n_valid": int(tot),
        },
        "correlations": {
            "spearman_density_curl": round(float(r_curl), 4) if r_curl is not None else None,
        },
        "regional_density": {
            "east_end": round(float(east_dens), 5) if east_dens is not None else None,
            "nino34": round(float(nino34_dens), 5) if nino34_dens is not None else None,
        },
        "baselines": {
            "saddle_frac_phase35": 0.575,  # Phase 35: curl τ 全場鞍點比例
            "spearman_density_curl_phase35b": 0.26,  # Phase 35b: 格點 mask r≈0.24-0.28
        },
    }

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  ✅ {OUTPUT_JSON}")

    # 6. History append
    if os.path.exists(HISTORY_JSON):
        with open(HISTORY_JSON) as f:
            hist = json.load(f)
    else:
        hist = {"records": []}

    new_rec = {
        "month": data["month"],
        "saddle_frac": data["morse"]["saddle_frac"],
        "spearman_density_curl": data["correlations"]["spearman_density_curl"],
        "east_end_density": data["regional_density"]["east_end"],
        "nino34_density": data["regional_density"]["nino34"],
        "field": "curl_tau",
    }
    hist["records"] = [r for r in hist["records"] if r.get("month") != data["month"]]
    hist["records"].append(new_rec)
    hist["records"].sort(key=lambda r: r["month"])
    if len(hist["records"]) > 120:
        hist["records"] = hist["records"][-120:]
    with open(HISTORY_JSON, 'w') as f:
        json.dump(hist, f, indent=2, ensure_ascii=False)
    print(f"  ✅ {HISTORY_JSON} ({len(hist['records'])} records)")

    print("\n🎯 Curl Monitor done.")


if __name__ == '__main__':
    main()
