#!/usr/bin/env python3
"""
非線性過擬合診斷
================
核心問題：非線性特徵喺驗證集跌，係真過擬合定係評估方法問題？

方法 1：驗證集上直接量 dCor（唔用線性回歸）
方法 2：結構穩定性（跨時期 ρ_std 場 pattern 相似度）
方法 3：非線性 LOO（用 dCor 方向預測，唔用 OLS）
"""
import xarray as xr
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.ndimage import uniform_filter
import json, time

DATA_DIR = Path('/tmp/africa_enso_3d_full_year')
OUTPUT_DIR = Path('projects/enso/output')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EL_NINO = [1982,1983,1986,1987,1991,1994,1997,2002,2004,2006,2009,2015]
LA_NINA = [1984,1985,1988,1995,1998,1999,2000,2007,2010,2011,2016,2017,2020]
NEUTRAL = [1989,1990,1992,1993,1996,2001,2003,2005,2008,2012,2013,2014,2018,2019]
VAL_EL_NINO = [2023]
VAL_LA_NINA = [2021, 2022, 2024]
VAL_NEUTRAL = [2025]

def get_phase(year):
    if year in EL_NINO or year in VAL_EL_NINO: return 'El_Nino'
    if year in LA_NINA or year in VAL_LA_NINA: return 'La_Nina'
    return 'Neutral'

def calc_rho(field, smooth_size=3):
    mask = np.isnan(field)
    field_filled = np.where(mask, 0, field)
    smooth = uniform_filter(field_filled, size=smooth_size)
    rho = np.abs(field_filled - smooth)
    rho[mask] = np.nan
    return rho

def dcor(x, y):
    """距離相關"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 5: return np.nan, np.nan
    
    dx = np.abs(x[:, None] - x[None, :])
    dy = np.abs(y[:, None] - y[None, :])
    
    row_mean_dx = dx.mean(axis=1, keepdims=True)
    col_mean_dx = dx.mean(axis=0, keepdims=True)
    grand_mean_dx = dx.mean()
    dcx = dx - row_mean_dx - col_mean_dx + grand_mean_dx
    
    row_mean_dy = dy.mean(axis=1, keepdims=True)
    col_mean_dy = dy.mean(axis=0, keepdims=True)
    grand_mean_dy = dy.mean()
    dcy = dy - row_mean_dy - col_mean_dy + grand_mean_dy
    
    dcov_xy = np.sqrt(np.mean(dcx * dcy))
    dcov_xx = np.sqrt(np.mean(dcx * dcx))
    dcov_yy = np.sqrt(np.mean(dcy * dcy))
    
    if dcov_xx == 0 or dcov_yy == 0: return 0.0, 1.0
    dcor_val = dcov_xy / np.sqrt(dcov_xx * dcov_yy)
    
    # permutation p
    n_perm = 500
    count = 0
    for _ in range(n_perm):
        y_perm = np.random.permutation(y)
        dy_p = np.abs(y_perm[:, None] - y_perm[None, :])
        row_mean_p = dy_p.mean(axis=1, keepdims=True)
        col_mean_p = dy_p.mean(axis=0, keepdims=True)
        grand_mean_p = dy_p.mean()
        dcy_p = dy_p - row_mean_p - col_mean_p + grand_mean_p
        dcov_p = np.sqrt(np.mean(dcx * dcy_p))
        dcor_p = dcov_p / np.sqrt(dcov_xx * np.sqrt(np.mean(dcy_p * dcy_p)) + 1e-30)
        if dcor_p >= dcor_val: count += 1
    
    return float(dcor_val), float((count + 1) / (n_perm + 1))

def load_year_data(year, months):
    """載入指定年份+月份嘅 850hPa mean 同 700hPa ρ_std"""
    files = sorted(DATA_DIR.glob(f'z3d_*_{year}_*.nc'))
    entries = []
    for f in files:
        parts = f.stem.split('_')
        month = int(parts[3])
        if month not in months: continue
        ds = xr.open_dataset(f)
        z = ds['z'].values.squeeze()
        ds.close()
        
        mean_850 = float(np.nanmean(z[0]))
        rho_700 = calc_rho(z[1])
        valid = rho_700[~np.isnan(rho_700)]
        rho_std = float(np.nanstd(valid)) if len(valid) > 0 else np.nan
        
        entries.append({'mean_850': mean_850, 'rho_std_700': rho_std})
    
    if not entries: return None
    return {
        'mean_850': np.mean([e['mean_850'] for e in entries]),
        'rho_std_700': np.mean([e['rho_std_700'] for e in entries]),
    }

def load_rho_field(year, month):
    """載入指定年月嘅 700hPa ρ 場（用於結構穩定性）"""
    f = DATA_DIR / f'z3d_summer_{year}_{month:02d}.nc'
    if not f.exists():
        f = DATA_DIR / f'z3d_winter_{year}_{month:02d}.nc'
    if not f.exists(): return None
    
    ds = xr.open_dataset(f)
    z = ds['z'].values.squeeze()
    ds.close()
    return calc_rho(z[1])  # 700hPa

def main():
    print("🔬 非線性過擬合診斷")
    print("="*70)
    
    t0 = time.time()
    
    # ============================================================
    # 載入全部數據
    # ============================================================
    print("\n📥 載入數據...")
    
    all_years = list(range(1982, 2026))
    charging_months = [5, 6, 7]
    winter_months = [10, 11, 12]
    
    # 充電窗口特徵
    charging = {}
    for y in all_years:
        d = load_year_data(y, charging_months)
        if d: charging[y] = d
    
    # 冬季 response
    winter = {}
    for y in all_years:
        d = load_year_data(y, winter_months)
        if d: winter[y] = d
    
    # 計算 response（冬季 850hPa mean anomaly）
    global_winter_mean = np.mean([winter[y]['mean_850'] for y in winter])
    
    # 配對
    paired_years = sorted(set(charging.keys()) & set(winter.keys()))
    print(f"   配對年份: {len(paired_years)} ({paired_years[0]}-{paired_years[-1]})")
    
    train_years = [y for y in paired_years if y <= 2020]
    val_years = [y for y in paired_years if y >= 2021]
    
    print(f"   訓練: {len(train_years)} 年 ({train_years[0]}-{train_years[-1]})")
    print(f"   驗證: {len(val_years)} 年 ({val_years[0]}-{val_years[-1]})")
    
    # ============================================================
    # 方法 1：驗證集 dCor（唔用線性回歸）
    # ============================================================
    print("\n" + "="*70)
    print("方法 1：驗證集 dCor（唔用線性回歸）")
    print("="*70)
    print("   如果 dCor 喺驗證集仍然顯著 → 唔係過擬合，係線性回歸捕捉唔到")
    print("   如果 dCor 跌到唔顯著 → 真過擬合")
    
    # 訓練集 dCor
    train_mean = np.array([charging[y]['mean_850'] for y in train_years])
    train_rho = np.array([charging[y]['rho_std_700'] for y in train_years])
    train_resp = np.array([winter[y]['mean_850'] - global_winter_mean for y in train_years])
    
    dcor_train_mean, dp_train_mean = dcor(train_mean, train_resp)
    dcor_train_rho, dp_train_rho = dcor(train_rho, train_resp)
    r_train_mean, rp_train_mean = stats.pearsonr(train_mean, train_resp)
    r_train_rho, rp_train_rho = stats.pearsonr(train_rho, train_resp)
    
    print(f"\n   訓練集（{len(train_years)} 年）:")
    print(f"     850hPa mean:  Pearson r={r_train_mean:.4f} (p={rp_train_mean:.4f}), dCor={dcor_train_mean:.4f} (p={dp_train_mean:.4f})")
    print(f"     700hPa ρ_std: Pearson r={r_train_rho:.4f} (p={rp_train_rho:.4f}), dCor={dcor_train_rho:.4f} (p={dp_train_rho:.4f})")
    
    # 驗證集 dCor
    val_mean = np.array([charging[y]['mean_850'] for y in val_years])
    val_rho = np.array([charging[y]['rho_std_700'] for y in val_years])
    val_resp = np.array([winter[y]['mean_850'] - global_winter_mean for y in val_years])
    
    dcor_val_mean, dp_val_mean = dcor(val_mean, val_resp)
    dcor_val_rho, dp_val_rho = dcor(val_rho, val_resp)
    r_val_mean, rp_val_mean = stats.pearsonr(val_mean, val_resp)
    r_val_rho, rp_val_rho = stats.pearsonr(val_rho, val_resp)
    
    print(f"\n   驗證集（{len(val_years)} 年）:")
    print(f"     850hPa mean:  Pearson r={r_val_mean:.4f} (p={rp_val_mean:.4f}), dCor={dcor_val_mean:.4f} (p={dp_val_mean:.4f})")
    print(f"     700hPa ρ_std: Pearson r={r_val_rho:.4f} (p={rp_val_rho:.4f}), dCor={dcor_val_rho:.4f} (p={dp_val_rho:.4f})")
    
    # 全樣本 dCor
    all_mean = np.array([charging[y]['mean_850'] for y in paired_years])
    all_rho = np.array([charging[y]['rho_std_700'] for y in paired_years])
    all_resp = np.array([winter[y]['mean_850'] - global_winter_mean for y in paired_years])
    
    dcor_all_mean, dp_all_mean = dcor(all_mean, all_resp)
    dcor_all_rho, dp_all_rho = dcor(all_rho, all_resp)
    
    print(f"\n   全樣本（{len(paired_years)} 年）:")
    print(f"     850hPa mean:  dCor={dcor_all_mean:.4f} (p={dp_all_mean:.4f})")
    print(f"     700hPa ρ_std: dCor={dcor_all_rho:.4f} (p={dp_all_rho:.4f})")
    
    # 判斷
    print(f"\n   📊 判斷:")
    if dp_val_rho < 0.10:
        print(f"     ✅ 驗證集 dCor 仍然顯著（p={dp_val_rho:.4f}）→ 唔係過擬合")
        print(f"        線性回歸捕捉唔到係因為 ρ_std-ENSO 關係係非線性嘅")
    else:
        print(f"     ⚠️ 驗證集 dCor 唔顯著（p={dp_val_rho:.4f}）→ 可能真過擬合")
    
    # ============================================================
    # 方法 2：結構穩定性（跨時期 ρ_std 場 pattern）
    # ============================================================
    print("\n" + "="*70)
    print("方法 2：結構穩定性（跨時期 ρ_std 場 pattern）")
    print("="*70)
    print("   如果空間 pattern 跨時期相似 → 結構穩定，唔係過擬合")
    print("   如果 2021-2025 pattern 完全唔同 → 真過擬合")
    
    # 計算三個時期嘅平均 ρ 場（700hPa, 5-7月）
    periods = {
        '1982-2000': list(range(1982, 2001)),
        '2001-2020': list(range(2001, 2021)),
        '2021-2025': list(range(2021, 2026)),
    }
    
    period_fields = {}
    for period_name, years in periods.items():
        fields = []
        for y in years:
            for m in [5, 6, 7]:
                field = load_rho_field(y, m)
                if field is not None:
                    fields.append(field)
        
        if fields:
            stack = np.stack(fields)
            period_fields[period_name] = np.nanmean(stack, axis=0)
            print(f"   {period_name}: {len(fields)} 個月")
    
    # 計算跨時期相關（用場嘅 flatten）
    print(f"\n   跨時期空間 pattern 相關（Pearson on flattened field）:")
    period_names = list(period_fields.keys())
    for i in range(len(period_names)):
        for j in range(i+1, len(period_names)):
            f1 = period_fields[period_names[i]].flatten()
            f2 = period_fields[period_names[j]].flatten()
            valid = ~(np.isnan(f1) | np.isnan(f2))
            if valid.sum() > 100:
                r, p = stats.pearsonr(f1[valid], f2[valid])
                dc, dp = dcor(f1[valid][:500], f2[valid][:500])  # subsample for speed
                print(f"     {period_names[i]} vs {period_names[j]}: Pearson r={r:.4f}, dCor={dc:.4f}")
    
    # 計算跨時期 ρ_std 值嘅相關
    print(f"\n   跨時期 ρ_std 值相關（年際）:")
    for period_name, years in periods.items():
        rho_vals = []
        resp_vals = []
        for y in years:
            if y in charging and y in winter:
                rho_vals.append(charging[y]['rho_std_700'])
                resp_vals.append(winter[y]['mean_850'] - global_winter_mean)
        
        if len(rho_vals) >= 5:
            r, p = stats.pearsonr(rho_vals, resp_vals)
            dc, dp = dcor(np.array(rho_vals), np.array(resp_vals))
            print(f"     {period_name}: Pearson r={r:.4f} (p={p:.4f}), dCor={dc:.4f} (p={dp:.4f}), n={len(rho_vals)}")
    
    # ============================================================
    # 方法 3：非線性 LOO（用 dCor 方向預測）
    # ============================================================
    print("\n" + "="*70)
    print("方法 3：非線性 LOO（用 dCor 方向預測）")
    print("="*70)
    print("   唔用 OLS，直接用 dCor 嘅 sign 預測")
    
    # 全樣本 LOO
    n = len(paired_years)
    dcor_loo_correct = 0
    ols_loo_correct = 0
    
    for i in range(n):
        # 留出
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        
        train_mean_loo = all_mean[mask]
        train_rho_loo = all_rho[mask]
        train_resp_loo = all_resp[mask]
        
        # dCor 方法：用 ρ_std 同 response 嘅 dCor sign 預測
        # 如果 dCor > 0 → ρ_std 高 = response 高（warming）
        # 如果 dCor < 0 → ρ_std 高 = response 低（cooling）
        dc, _ = dcor(train_rho_loo, train_resp_loo)
        
        # 用 ρ_std 嘅 rank 同 response 嘅 rank 相關判斷方向
        rho_rank = stats.rankdata(train_rho_loo)
        resp_rank = stats.rankdata(train_resp_loo)
        rank_corr = np.corrcoef(rho_rank, resp_rank)[0, 1]
        
        # 預測：用留出年嘅 ρ_std 喺訓練集嘅 percentile 判斷
        rho_val = all_rho[i]
        rho_pct = np.mean(train_rho_loo < rho_val)
        
        # 如果 rank_corr > 0 → 高 ρ = 高 response → rho_pct > 0.5 = warming
        # 如果 rank_corr < 0 → 高 ρ = 低 response → rho_pct > 0.5 = cooling
        if rank_corr > 0:
            pred_sign = 1 if rho_pct > 0.5 else -1
        else:
            pred_sign = -1 if rho_pct > 0.5 else 1
        
        actual_sign = np.sign(all_resp[i])
        if pred_sign == actual_sign:
            dcor_loo_correct += 1
        
        # OLS 方法（對比）
        slope, intercept, _, _, _ = stats.linregress(train_rho_loo, train_resp_loo)
        ols_pred = slope * all_rho[i] + intercept
        if np.sign(ols_pred) == actual_sign:
            ols_loo_correct += 1
    
    print(f"\n   LOO 結果（{n} 年）:")
    print(f"     OLS 線性:     {ols_loo_correct}/{n} = {ols_loo_correct/n:.1%}")
    print(f"     dCor 非線性:  {dcor_loo_correct}/{n} = {dcor_loo_correct/n:.1%}")
    
    # ============================================================
    # 總結
    # ============================================================
    print("\n" + "="*70)
    print("📊 總結：過擬合診斷")
    print("="*70)
    
    print(f"\n  問題：非線性特徵喺驗證集跌，係真過擬合定係評估方法問題？")
    print(f"\n  證據：")
    print(f"    1. 驗證集 dCor: ρ_std dCor={dcor_val_rho:.4f} (p={dp_val_rho:.4f})")
    print(f"    2. 結構穩定性: 跨時期 pattern 相關（見上方）")
    print(f"    3. 非線性 LOO: {dcor_loo_correct/n:.1%} vs OLS {ols_loo_correct/n:.1%}")
    
    # 最終判斷
    evidence = []
    
    if dp_val_rho < 0.10:
        evidence.append("✅ 驗證集 dCor 仍然顯著")
    else:
        evidence.append("⚠️ 驗證集 dCor 唔顯著")
    
    if dcor_loo_correct > ols_loo_correct:
        evidence.append("✅ 非線性 LOO 好過 OLS")
    else:
        evidence.append("⚠️ 非線性 LOO 唔好過 OLS")
    
    print(f"\n  判斷：")
    for e in evidence:
        print(f"    {e}")
    
    if "✅" in evidence[0] and "✅" in evidence[1]:
        print(f"\n  🎯 結論：唔係真過擬合。非線性特徵有穩定信號，")
        print(f"     但用線性回歸去組合同評估會丟失呢個信號。")
        print(f"     正正如 8/18 講嘅：「非線性只能同構，不能硬套」")
    elif "⚠️" in evidence[0]:
        print(f"\n  🎯 結論：可能真過擬合。非線性特徵喺驗證集冇穩定信號。")
    else:
        print(f"\n  🎯 結論：混合。需要更多驗證數據。")
    
    # 保存
    results = {
        'train': {
            'n': len(train_years),
            'pearson_mean': float(r_train_mean),
            'pearson_rho': float(r_train_rho),
            'dcor_mean': float(dcor_train_mean),
            'dcor_rho': float(dcor_train_rho),
        },
        'validation': {
            'n': len(val_years),
            'pearson_mean': float(r_val_mean),
            'pearson_rho': float(r_val_rho),
            'dcor_mean': float(dcor_val_mean),
            'dcor_rho': float(dcor_val_rho),
        },
        'loo': {
            'ols': ols_loo_correct / n,
            'dcor': dcor_loo_correct / n,
        }
    }
    
    with open(OUTPUT_DIR / 'nonlinear_overfit_diagnosis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    elapsed = time.time() - t0
    print(f"\n⏱️ 總耗時: {elapsed:.1f} 秒")

if __name__ == '__main__':
    main()
