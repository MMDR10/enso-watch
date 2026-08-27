#!/usr/bin/env python3
"""
Africa 電容模型 — 三項深化測試
Deep-1: r=0.475 穩健性（不同窗口、不同 ENSO 指標）
Deep-2: 垂直結構深入（200hPa 充放電時序 × ENSO lag）
Deep-3: 預測模型（operational skill）
"""
import xarray as xr
import numpy as np
from pathlib import Path
from scipy import stats
import json, sys

DATA_DIR = Path('/tmp/africa_enso_3d_full_year')
OUTPUT_DIR = Path('projects/enso/output')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LEVELS = [850, 700, 600, 500, 400, 300, 250, 200]

# ENSO 分類
EL_NINO = [1982,1983,1986,1987,1991,1994,1997,2002,2004,2006,2009,2015]
LA_NINA = [1984,1985,1988,1995,1998,1999,2000,2007,2010,2011,2016,2017,2020]
NEUTRAL = [1989,1990,1992,1993,1996,2001,2003,2005,2008,2012,2013,2014,2018,2019]

STRONG_EN = [1982, 1997, 2015]
STRONG_LN = [1988, 1998, 1999, 2010, 2011]

def load_all():
    files = sorted(DATA_DIR.glob('z3d_*.nc'))
    data = []
    for f in files:
        parts = f.stem.split('_')
        year, month = int(parts[2]), int(parts[3])
        ds = xr.open_dataset(f)
        z = ds['z'].values.squeeze()
        lev_means = {lev: float(np.nanmean(z[i])) for i, lev in enumerate(LEVELS)}
        lev_means['total'] = float(np.nanmean(z))
        data.append({'year': year, 'month': month, **lev_means})
        ds.close()
    return data

def get_enso_phase(year):
    if year in EL_NINO: return 'El_Nino'
    if year in LA_NINA: return 'La_Nina'
    return 'Neutral'

# ============================================================
# Deep-1: r=0.475 穩健性
# ============================================================
def deep1_robustness(data):
    print("\n" + "="*70)
    print("DEEP-1: r=0.475 穩健性驗證")
    print("="*70)

    years = sorted(set(d['year'] for d in data))
    results = {}

    # --- 1a: 不同充電窗口 ---
    print("\n  1a: 不同充電窗口 -> 冬季(10-12月)響應")
    windows = {
        '4-6月 (original)': [4,5,6],
        '3-5月': [3,4,5],
        '5-7月': [5,6,7],
        '4-5月': [4,5],
        '5-6月': [5,6],
        '4月 only': [4],
        '5月 only': [5],
        '6月 only': [6],
    }
    winter_months = [10, 11, 12]

    window_results = {}
    for name, months in windows.items():
        charging = []
        response = []
        for y in years:
            ch = [d['total'] for d in data if d['year']==y and d['month'] in months]
            wi = [d['total'] for d in data if d['year']==y and d['month'] in winter_months]
            if ch and wi:
                charging.append(np.mean(ch))
                response.append(np.mean(wi))
        r, p = stats.pearsonr(charging, response)
        window_results[name] = {'r': round(r,4), 'p': round(p,4), 'n': len(charging)}
        sig = 'YES' if p < 0.05 else 'NO'
        print(f"    {name:20s}: r={r:+.4f}, p={p:.4f} {sig} (n={len(charging)})")
    results['window_robustness'] = window_results

    # --- 1b: 不同壓力層 ---
    print("\n  1b: 不同壓力層充電量 -> 冬季響應")
    layer_results = {}
    for lev in LEVELS:
        charging = []
        response = []
        for y in years:
            ch = [d[lev] for d in data if d['year']==y and d['month'] in [4,5,6]]
            wi = [d['total'] for d in data if d['year']==y and d['month'] in winter_months]
            if ch and wi:
                charging.append(np.mean(ch))
                response.append(np.mean(wi))
        r, p = stats.pearsonr(charging, response)
        layer_results[f'{lev}hPa'] = {'r': round(r,4), 'p': round(p,4)}
        sig = 'YES' if p < 0.05 else 'NO'
        print(f"    {lev} hPa: r={r:+.4f}, p={p:.4f} {sig}")
    results['layer_robustness'] = layer_results

    # --- 1c: 不同 ENSO 指標 ---
    print("\n  1c: 不同 ENSO 強度指標")
    winter_clim = np.mean([d['total'] for d in data if d['month'] in winter_months])
    charging_all = []
    enso_strength = []
    for y in years:
        ch = [d['total'] for d in data if d['year']==y and d['month'] in [4,5,6]]
        wi = [d['total'] for d in data if d['year']==y and d['month'] in winter_months]
        if ch and wi:
            charging_all.append(np.mean(ch))
            enso_strength.append(np.mean(wi) - winter_clim)

    r, p = stats.pearsonr(charging_all, enso_strength)
    print(f"    充電量 vs 冬季 anomaly: r={r:+.4f}, p={p:.4f}")
    results['enso_strength_proxy'] = {'r': round(r,4), 'p': round(p,4)}

    # --- 1d: Leave-one-out 穩健性 ---
    print("\n  1d: Leave-one-out 交叉驗證")
    charging_arr = np.array(charging_all)
    response_arr = np.array(enso_strength)
    loo_rs = []
    for i in range(len(years)):
        mask = np.ones(len(years), dtype=bool)
        mask[i] = False
        r_loo, _ = stats.pearsonr(charging_arr[mask], response_arr[mask])
        loo_rs.append(r_loo)
    loo_rs = np.array(loo_rs)
    print(f"    LOO r: mean={loo_rs.mean():.4f}, std={loo_rs.std():.4f}")
    print(f"    LOO r range: [{loo_rs.min():.4f}, {loo_rs.max():.4f}]")
    print(f"    全部 > 0: {(loo_rs > 0).sum()}/{len(loo_rs)}")
    results['loo'] = {
        'mean_r': round(float(loo_rs.mean()), 4),
        'std_r': round(float(loo_rs.std()), 4),
        'min_r': round(float(loo_rs.min()), 4),
        'max_r': round(float(loo_rs.max()), 4),
        'all_positive': int((loo_rs > 0).sum()),
        'total': len(loo_rs)
    }

    # --- 1e: 分 ENSO 相位相關 ---
    print("\n  1e: 分 ENSO 相位相關")
    for phase, yrs_list in [('El_Nino', EL_NINO), ('La_Nina', LA_NINA), ('Neutral', NEUTRAL)]:
        ch_p = [c for c, y in zip(charging_all, years) if y in yrs_list]
        rs_p = [r for r, y in zip(enso_strength, years) if y in yrs_list]
        if len(ch_p) >= 5:
            r_p, p_p = stats.pearsonr(ch_p, rs_p)
            print(f"    {phase:10s}: r={r_p:+.4f}, p={p_p:.4f} (n={len(ch_p)})")
            results[f'phase_{phase}'] = {'r': round(r_p,4), 'p': round(p_p,4), 'n': len(ch_p)}

    return results

# ============================================================
# Deep-2: 垂直結構深入（200hPa lag × ENSO）
# ============================================================
def deep2_vertical_lag(data):
    print("\n" + "="*70)
    print("DEEP-2: 垂直結構深入 -- 200hPa lag x ENSO")
    print("="*70)

    results = {}

    # --- 2a: 逐層 lag 相關 ---
    print("\n  2a: 逐層 lag 相關結構")
    years = sorted(set(d['year'] for d in data))
    sorted_data = sorted(data, key=lambda d: (d['year'], d['month']))

    for lev in [200, 500, 850]:
        z_series = np.array([d[lev] for d in sorted_data])
        print(f"\n    {lev} hPa:")
        lag_results = {}
        for lag in [1, 3, 6, 9, 12]:
            if lag < len(z_series):
                r = np.corrcoef(z_series[:-lag], z_series[lag:])[0,1]
                lag_results[lag] = round(r, 4)
                print(f"      Lag {lag:2d}月: r={r:.4f}")
        results[f'autocorr_{lev}hPa'] = lag_results

    # --- 2b: 200hPa 逐月 climatology x ENSO 相位 ---
    print("\n  2b: 200hPa 逐月 climatology x ENSO 相位")
    for phase, yrs_list, pname in [('El_Nino', EL_NINO, 'El Nino'), ('La_Nina', LA_NINA, 'La Nina'), ('Neutral', NEUTRAL, 'Neutral')]:
        monthly_200 = {}
        for m in range(1, 13):
            vals = [d[200] for d in data if d['month']==m and d['year'] in yrs_list]
            if vals:
                monthly_200[m] = round(float(np.mean(vals)), 1)
        print(f"\n    {pname}:")
        for m in range(1, 13):
            if m in monthly_200:
                print(f"      {m:02d}月: {monthly_200[m]:.1f}")
        results[f'monthly_200hPa_{phase}'] = monthly_200

    # --- 2c: 垂直相干性 ---
    print("\n  2c: 垂直相干性（層間 Pearson r）")
    z_by_level = {}
    for lev in LEVELS:
        z_by_level[lev] = np.array([d[lev] for d in sorted_data])

    for i in range(len(LEVELS)-1):
        lev_hi, lev_lo = LEVELS[i], LEVELS[i+1]
        r = np.corrcoef(z_by_level[lev_hi], z_by_level[lev_lo])[0,1]
        print(f"    {lev_hi}-{lev_lo} hPa: r={r:.4f}")
        results[f'coherence_{lev_hi}_{lev_lo}'] = round(r, 4)

    # --- 2d: 200hPa 充電量 vs 各層 lag 6月 ---
    print("\n  2d: 200hPa 夏季充電量 vs 各層 lag 6月")
    for lev in LEVELS:
        charging_200 = []
        future_lev = []
        for y in years:
            ch = [d[200] for d in data if d['year']==y and d['month'] in [4,5,6]]
            fu = [d[lev] for d in data if d['year']==y and d['month'] in [10,11,12]]
            if ch and fu:
                charging_200.append(np.mean(ch))
                future_lev.append(np.mean(fu))
        r, p = stats.pearsonr(charging_200, future_lev)
        sig = 'YES' if p < 0.05 else 'NO'
        print(f"    200hPa充電 -> {lev}hPa lag6: r={r:+.4f}, p={p:.4f} {sig}")
        results[f'cross_level_200_{lev}'] = {'r': round(r,4), 'p': round(p,4)}

    return results

# ============================================================
# Deep-3: 預測模型（operational skill）
# ============================================================
def deep3_prediction(data):
    print("\n" + "="*70)
    print("DEEP-3: 預測模型 -- Operational Skill")
    print("="*70)

    years = sorted(set(d['year'] for d in data))
    results = {}

    winter_clim = np.mean([d['total'] for d in data if d['month'] in [10,11,12]])

    X_all = []
    y_all = []
    year_list = []

    for y in years:
        features = []
        for lev in LEVELS:
            ch = [d[lev] for d in data if d['year']==y and d['month'] in [4,5,6]]
            if ch:
                features.append(np.mean(ch))
            else:
                features.append(np.nan)
        ch_total = [d['total'] for d in data if d['year']==y and d['month'] in [4,5,6]]
        wi = [d['total'] for d in data if d['year']==y and d['month'] in [10,11,12]]
        if ch_total and wi and not any(np.isnan(features)):
            X_all.append(features)
            y_all.append(np.mean(wi) - winter_clim)
            year_list.append(y)

    X_all = np.array(X_all)
    y_all = np.array(y_all)

    # --- 3a: 單變量線性回歸 ---
    print("\n  3a: 單變量線性回歸（total 充電量 -> 冬季 anomaly）")
    charging_total = np.array([np.mean([d['total'] for d in data if d['year']==y and d['month'] in [4,5,6]])
                               for y in year_list])
    slope, intercept, r, p, se = stats.linregress(charging_total, y_all)
    print(f"    y = {slope:.6f} * x + {intercept:.2f}")
    print(f"    r={r:.4f}, R2={r**2:.4f}, p={p:.4f}")
    results['univariate'] = {'r': round(r,4), 'R2': round(r**2,4), 'p': round(p,4)}

    # --- 3b: LOO 預測 skill ---
    print("\n  3b: LOO 預測 skill（單變量）")
    predictions = []
    actuals = []
    for i in range(len(year_list)):
        mask = np.ones(len(year_list), dtype=bool)
        mask[i] = False
        s, ic, _, _, _ = stats.linregress(charging_total[mask], y_all[mask])
        pred = s * charging_total[i] + ic
        predictions.append(pred)
        actuals.append(y_all[i])

    predictions = np.array(predictions)
    actuals = np.array(actuals)
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals)**2))
    direction_correct = np.sum(np.sign(predictions) == np.sign(actuals))
    threshold = np.std(actuals) * 0.5
    event_mask = np.abs(actuals) > threshold
    if event_mask.sum() > 0:
        event_correct = np.sum(np.sign(predictions[event_mask]) == np.sign(actuals[event_mask]))
        event_skill = event_correct / event_mask.sum()
    else:
        event_skill = 0

    print(f"    MAE = {mae:.2f} m2/s2")
    print(f"    RMSE = {rmse:.2f} m2/s2")
    print(f"    方向正確率 = {direction_correct}/{len(year_list)} = {direction_correct/len(year_list):.1%}")
    print(f"    事件預測 skill (>0.5sigma) = {event_skill:.1%}")
    results['loo_prediction'] = {
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'direction_accuracy': round(float(direction_correct/len(year_list)), 3),
        'event_skill': round(float(event_skill), 3),
        'n': len(year_list)
    }

    # --- 3c: 多層特徵 LOO ---
    print("\n  3c: 多層特徵 LOO（最佳單層 vs 全層）")
    best_lev_r = 0
    best_lev = 0
    for li, lev in enumerate(LEVELS):
        feat = X_all[:, li]
        s, ic, r_lev, _, _ = stats.linregress(feat, y_all)
        if abs(r_lev) > abs(best_lev_r):
            best_lev_r = r_lev
            best_lev = lev

    print(f"    最佳單層: {best_lev} hPa (r={best_lev_r:.4f})")

    best_idx = LEVELS.index(best_lev)
    best_feat = X_all[:, best_idx]
    loo_preds_best = []
    for i in range(len(year_list)):
        mask = np.ones(len(year_list), dtype=bool)
        mask[i] = False
        s, ic, _, _, _ = stats.linregress(best_feat[mask], y_all[mask])
        loo_preds_best.append(s * best_feat[i] + ic)
    loo_preds_best = np.array(loo_preds_best)
    mae_best = np.mean(np.abs(loo_preds_best - actuals))
    dir_best = np.sum(np.sign(loo_preds_best) == np.sign(actuals))

    print(f"    最佳單層 LOO: MAE={mae_best:.2f}, 方向={dir_best}/{len(year_list)}={dir_best/len(year_list):.1%}")

    loo_preds_all = []
    for i in range(len(year_list)):
        mask = np.ones(len(year_list), dtype=bool)
        mask[i] = False
        preds_lev = []
        for li in range(len(LEVELS)):
            s, ic, _, _, _ = stats.linregress(X_all[mask, li], y_all[mask])
            preds_lev.append(s * X_all[i, li] + ic)
        loo_preds_all.append(np.mean(preds_lev))
    loo_preds_all = np.array(loo_preds_all)
    mae_all = np.mean(np.abs(loo_preds_all - actuals))
    dir_all = np.sum(np.sign(loo_preds_all) == np.sign(actuals))

    print(f"    全層平均 LOO: MAE={mae_all:.2f}, 方向={dir_all}/{len(year_list)}={dir_all/len(year_list):.1%}")

    results['multilayer'] = {
        'best_level': best_lev,
        'best_level_r': round(best_lev_r, 4),
        'best_level_LOO_MAE': round(mae_best, 2),
        'best_level_LOO_direction': round(float(dir_best/len(year_list)), 3),
        'all_level_LOO_MAE': round(mae_all, 2),
        'all_level_LOO_direction': round(float(dir_all/len(year_list)), 3),
    }

    # --- 3d: 逐事件預測表 ---
    print("\n  3d: 逐事件預測表")
    print(f"    {'年份':>6s} {'相位':>10s} {'實際':>10s} {'預測':>10s} {'方向':>6s}")
    for i, y in enumerate(year_list):
        phase = get_enso_phase(y)
        actual = actuals[i]
        pred = loo_preds_best[i]
        direction = 'OK' if np.sign(pred) == np.sign(actual) else 'XX'
        print(f"    {y:>6d} {phase:>10s} {actual:>+10.2f} {pred:>+10.2f} {direction:>6s}")

    return results

# ============================================================
# Main
# ============================================================
def main():
    print("Africa 電容模型 -- 三項深化測試")
    print("="*70)

    print("\nLoading 468 files...")
    data = load_all()
    print(f"   Years: {min(d['year'] for d in data)}-{max(d['year'] for d in data)}")

    test = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    all_results = {}
    if test == 0 or test == 1:
        all_results['deep1'] = deep1_robustness(data)
    if test == 0 or test == 2:
        all_results['deep2'] = deep2_vertical_lag(data)
    if test == 0 or test == 3:
        all_results['deep3'] = deep3_prediction(data)

    out_file = OUTPUT_DIR / 'africa_capacitor_deep3.json'
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved: {out_file}")

if __name__ == '__main__':
    main()
