#!/usr/bin/env python3
"""
Africa 電容非線性預測模型
============================
用 ρ 雜訊密度特徵建 LOO 預測，同線性方法對比
目標：睇非線性特徵能否提升 6 個月 ENSO 預測
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

LEVELS = [850, 700, 600, 500, 400, 300, 250, 200]

EL_NINO = [1982,1983,1986,1987,1991,1994,1997,2002,2004,2006,2009,2015]
LA_NINA = [1984,1985,1988,1995,1998,1999,2000,2007,2010,2011,2016,2017,2020]
NEUTRAL = [1989,1990,1992,1993,1996,2001,2003,2005,2008,2012,2013,2014,2018,2019]

def get_phase(year):
    if year in EL_NINO: return 'El_Nino'
    if year in LA_NINA: return 'La_Nina'
    return 'Neutral'

def calc_rho(field, smooth_size=3):
    mask = np.isnan(field)
    field_filled = np.where(mask, 0, field)
    if field.ndim == 2:
        smooth = uniform_filter(field_filled, size=smooth_size)
    else:
        smooth = uniform_filter(field_filled, size=smooth_size)
    rho = np.abs(field_filled - smooth)
    rho[mask] = np.nan
    return rho

def calc_rho_stats(field, smooth_size=3):
    rho = calc_rho(field, smooth_size)
    valid = rho[~np.isnan(rho)]
    if len(valid) == 0:
        return {k: np.nan for k in ['mean','std','median','q75','q90','iqr','skew','kurtosis']}
    from scipy.stats import skew, kurtosis
    return {
        'mean': float(np.nanmean(valid)),
        'std': float(np.nanstd(valid)),
        'median': float(np.nanmedian(valid)),
        'q75': float(np.nanpercentile(valid, 75)),
        'q90': float(np.nanpercentile(valid, 90)),
        'iqr': float(np.nanpercentile(valid, 75) - np.nanpercentile(valid, 25)),
        'skew': float(skew(valid)),
        'kurtosis': float(kurtosis(valid)),
    }

def load_features(charging_months=[5,6,7]):
    """載入所有特徵（線性 + 非線性），按年聚合"""
    files = sorted(DATA_DIR.glob('z3d_*.nc'))
    
    # 按年+月分組
    monthly = {}
    for f in files:
        parts = f.stem.split('_')
        year, month = int(parts[2]), int(parts[3])
        ds = xr.open_dataset(f)
        z = ds['z'].values.squeeze()  # (8, 41, 91)
        ds.close()
        
        entry = {'year': year, 'month': month}
        
        # 線性特徵：各層 mean
        for i, lev in enumerate(LEVELS):
            entry[f'{lev}hPa_mean'] = float(np.nanmean(z[i]))
        
        # 非線性特徵：各層 ρ 統計量
        for i, lev in enumerate(LEVELS):
            rho_stats = calc_rho_stats(z[i])
            for k, v in rho_stats.items():
                entry[f'{lev}hPa_rho_{k}'] = v
        
        key = (year, month)
        monthly[key] = entry
    
    # 按年聚合（充電窗口平均）
    years = sorted(set(y for y, m in monthly.keys()))
    
    # 冬季 response（10-12月 850hPa mean）
    winter_z = {}
    for (y, m), entry in monthly.items():
        if m in [10, 11, 12]:
            if y not in winter_z:
                winter_z[y] = []
            winter_z[y].append(entry['850hPa_mean'])
    
    global_winter_mean = np.mean([np.mean(v) for v in winter_z.values()])
    
    features_by_year = {}
    for y in years:
        # 充電窗口特徵
        ch_entries = [monthly[(y, m)] for m in charging_months if (y, m) in monthly]
        if not ch_entries or y not in winter_z:
            continue
        
        feat = {'year': y, 'phase': get_phase(y)}
        
        # 聚合所有特徵
        all_keys = [k for k in ch_entries[0].keys() if k not in ('year', 'month')]
        for k in all_keys:
            vals = [e[k] for e in ch_entries if k in e and not np.isnan(e[k])]
            if vals:
                feat[k] = np.mean(vals)
        
        # 冬季 response
        feat['enso_response'] = np.mean(winter_z[y]) - global_winter_mean
        
        features_by_year[y] = feat
    
    return features_by_year

def loo_predict(X, y, years):
    """Leave-one-out 線性預測"""
    n = len(X)
    predictions = np.zeros(n)
    
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        
        X_train, y_train = X[mask], y[mask]
        
        # 簡單線性回歸
        slope, intercept, _, _, _ = stats.linregress(X_train, y_train)
        predictions[i] = slope * X[i] + intercept
    
    return predictions

def loo_predict_multivariate(X, y):
    """Leave-one-out 多變量線性預測"""
    n = len(X)
    predictions = np.zeros(n)
    
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        
        X_train, y_train = X[mask], y[mask]
        
        # 加截距
        X_aug = np.column_stack([np.ones(len(X_train)), X_train])
        
        # OLS
        try:
            beta = np.linalg.lstsq(X_aug, y_train, rcond=None)[0]
            x_new = np.concatenate([[1], X[i]])
            predictions[i] = np.dot(beta, x_new)
        except:
            predictions[i] = np.mean(y_train)
    
    return predictions

def evaluate(y_true, y_pred, phases):
    """評估預測表現"""
    n = len(y_true)
    
    # 整體
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    direction_correct = np.mean(np.sign(y_true) == np.sign(y_pred))
    
    # 分相位
    phase_results = {}
    for phase_name in ['El_Nino', 'La_Nina', 'Neutral']:
        mask = np.array([p == phase_name for p in phases])
        if mask.sum() > 0:
            phase_mae = np.mean(np.abs(y_true[mask] - y_pred[mask]))
            phase_dir = np.mean(np.sign(y_true[mask]) == np.sign(y_pred[mask]))
            phase_results[phase_name] = {
                'n': int(mask.sum()),
                'mae': float(phase_mae),
                'direction': float(phase_dir)
            }
    
    return {
        'n': n,
        'mae': float(mae),
        'rmse': float(rmse),
        'direction': float(direction_correct),
        'phases': phase_results
    }

def main():
    print("🔬 Africa 電容非線性預測模型")
    print("="*70)
    
    t0 = time.time()
    
    # 載入特徵
    print("\n📥 載入特徵（線性 + 非線性）...")
    features = load_features(charging_months=[5, 6, 7])
    years = sorted(features.keys())
    print(f"   {len(years)} 年")
    
    # 準備數據
    y = np.array([features[y]['enso_response'] for y in years])
    phases = [features[y]['phase'] for y in years]
    
    # ============================================================
    # Model 1: 線性基準（850hPa mean）
    # ============================================================
    print("\n" + "="*70)
    print("MODEL 1: 線性基準（850hPa mean）")
    print("="*70)
    
    X1 = np.array([features[y]['850hPa_mean'] for y in years])
    pred1 = loo_predict(X1, y, years)
    r1, p1 = stats.pearsonr(X1, y)
    eval1 = evaluate(y, pred1, phases)
    
    print(f"  Pearson r: {r1:.4f}, p={p1:.4f}")
    print(f"  MAE: {eval1['mae']:.2f}")
    print(f"  方向正確率: {eval1['direction']:.1%}")
    for pn, pv in eval1['phases'].items():
        print(f"    {pn}: n={pv['n']}, MAE={pv['mae']:.2f}, 方向={pv['direction']:.1%}")
    
    # ============================================================
    # Model 2: 非線性（700hPa ρ_std）
    # ============================================================
    print("\n" + "="*70)
    print("MODEL 2: 非線性（700hPa ρ_std）")
    print("="*70)
    
    X2 = np.array([features[y]['700hPa_rho_std'] for y in years])
    pred2 = loo_predict(X2, y, years)
    r2, p2 = stats.pearsonr(X2, y)
    eval2 = evaluate(y, pred2, phases)
    
    print(f"  Pearson r: {r2:.4f}, p={p2:.4f}")
    print(f"  MAE: {eval2['mae']:.2f}")
    print(f"  方向正確率: {eval2['direction']:.1%}")
    for pn, pv in eval2['phases'].items():
        print(f"    {pn}: n={pv['n']}, MAE={pv['mae']:.2f}, 方向={pv['direction']:.1%}")
    
    # ============================================================
    # Model 3: 非線性（300hPa ρ_median）
    # ============================================================
    print("\n" + "="*70)
    print("MODEL 3: 非線性（300hPa ρ_median）")
    print("="*70)
    
    X3 = np.array([features[y]['300hPa_rho_median'] for y in years])
    pred3 = loo_predict(X3, y, years)
    r3, p3 = stats.pearsonr(X3, y)
    eval3 = evaluate(y, pred3, phases)
    
    print(f"  Pearson r: {r3:.4f}, p={p3:.4f}")
    print(f"  MAE: {eval3['mae']:.2f}")
    print(f"  方向正確率: {eval3['direction']:.1%}")
    for pn, pv in eval3['phases'].items():
        print(f"    {pn}: n={pv['n']}, MAE={pv['mae']:.2f}, 方向={pv['direction']:.1%}")
    
    # ============================================================
    # Model 4: 線性 + 非線性組合（2 變量）
    # ============================================================
    print("\n" + "="*70)
    print("MODEL 4: 線性 + 非線性組合（850hPa mean + 700hPa ρ_std）")
    print("="*70)
    
    X4 = np.column_stack([X1, X2])
    pred4 = loo_predict_multivariate(X4, y)
    eval4 = evaluate(y, pred4, phases)
    
    print(f"  MAE: {eval4['mae']:.2f}")
    print(f"  方向正確率: {eval4['direction']:.1%}")
    for pn, pv in eval4['phases'].items():
        print(f"    {pn}: n={pv['n']}, MAE={pv['mae']:.2f}, 方向={pv['direction']:.1%}")
    
    # ============================================================
    # Model 5: 全非線性特徵（多層 ρ）
    # ============================================================
    print("\n" + "="*70)
    print("MODEL 5: 全非線性特徵（8層 ρ_std + ρ_median）")
    print("="*70)
    
    rho_features = []
    for lev in LEVELS:
        rho_features.append(f'{lev}hPa_rho_std')
        rho_features.append(f'{lev}hPa_rho_median')
    
    X5 = np.array([[features[y].get(feat, np.nan) for feat in rho_features] for y in years])
    # 去掉有 NaN 嘅列
    valid_cols = ~np.any(np.isnan(X5), axis=0)
    X5 = X5[:, valid_cols]
    
    pred5 = loo_predict_multivariate(X5, y)
    eval5 = evaluate(y, pred5, phases)
    
    print(f"  特徵數: {X5.shape[1]}")
    print(f"  MAE: {eval5['mae']:.2f}")
    print(f"  方向正確率: {eval5['direction']:.1%}")
    for pn, pv in eval5['phases'].items():
        print(f"    {pn}: n={pv['n']}, MAE={pv['mae']:.2f}, 方向={pv['direction']:.1%}")
    
    # ============================================================
    # Model 6: 最優組合（線性 + 全非線性）
    # ============================================================
    print("\n" + "="*70)
    print("MODEL 6: 最優組合（850hPa mean + 全非線性）")
    print("="*70)
    
    X6 = np.column_stack([X1.reshape(-1, 1), X5])
    pred6 = loo_predict_multivariate(X6, y)
    eval6 = evaluate(y, pred6, phases)
    
    print(f"  特徵數: {X6.shape[1]}")
    print(f"  MAE: {eval6['mae']:.2f}")
    print(f"  方向正確率: {eval6['direction']:.1%}")
    for pn, pv in eval6['phases'].items():
        print(f"    {pn}: n={pv['n']}, MAE={pv['mae']:.2f}, 方向={pv['direction']:.1%}")
    
    # ============================================================
    # 總結對比
    # ============================================================
    print("\n" + "="*70)
    print("📊 總結對比")
    print("="*70)
    
    models = [
        ('M1: 線性 850hPa mean', eval1),
        ('M2: 非線性 700hPa ρ_std', eval2),
        ('M3: 非線性 300hPa ρ_median', eval3),
        ('M4: 線性+非線性 2變量', eval4),
        ('M5: 全非線性 16變量', eval5),
        ('M6: 線性+全非線性', eval6),
    ]
    
    print(f"\n  {'模型':<35} {'MAE':>8} {'方向':>8} {'El_Niño':>10} {'La_Niña':>10} {'Neutral':>10}")
    print(f"  {'-'*85}")
    
    for name, ev in models:
        en_dir = ev['phases'].get('El_Nino', {}).get('direction', 0)
        ln_dir = ev['phases'].get('La_Nina', {}).get('direction', 0)
        nt_dir = ev['phases'].get('Neutral', {}).get('direction', 0)
        print(f"  {name:<35} {ev['mae']:>8.2f} {ev['direction']:>7.1%} {en_dir:>9.1%} {ln_dir:>9.1%} {nt_dir:>9.1%}")
    
    # 找最佳
    best_mae = min(models, key=lambda x: x[1]['mae'])
    best_dir = max(models, key=lambda x: x[1]['direction'])
    best_neutral = max(models, key=lambda x: x[1]['phases'].get('Neutral', {}).get('direction', 0))
    
    print(f"\n  最低 MAE: {best_mae[0]} ({best_mae[1]['mae']:.2f})")
    print(f"  最高方向: {best_dir[0]} ({best_dir[1]['direction']:.1%})")
    print(f"  最佳 Neutral: {best_neutral[0]} ({best_neutral[1]['phases'].get('Neutral', {}).get('direction', 0):.1%})")
    
    # ============================================================
    # 逐事件詳情（最佳模型）
    # ============================================================
    print("\n" + "="*70)
    print("📋 逐事件詳情（M1 線性 vs M4 組合 vs M6 全組合）")
    print("="*70)
    
    print(f"\n  {'年份':<6} {'相位':<10} {'實際':>8} {'M1線性':>8} {'M4組合':>8} {'M6全組':>8} {'M1方向':>6} {'M4方向':>6} {'M6方向':>6}")
    print(f"  {'-'*75}")
    
    for i, yr in enumerate(years):
        actual = y[i]
        p1_val = pred1[i]
        p4_val = pred4[i]
        p6_val = pred6[i]
        
        d1 = "OK" if np.sign(actual) == np.sign(p1_val) else "XX"
        d4 = "OK" if np.sign(actual) == np.sign(p4_val) else "XX"
        d6 = "OK" if np.sign(actual) == np.sign(p6_val) else "XX"
        
        phase = phases[i]
        print(f"  {yr:<6} {phase:<10} {actual:>+8.2f} {p1_val:>+8.2f} {p4_val:>+8.2f} {p6_val:>+8.2f} {d1:>6} {d4:>6} {d6:>6}")
    
    # 保存結果
    results = {
        'models': {},
        'best': {
            'mae': best_mae[0],
            'direction': best_dir[0],
            'neutral': best_neutral[0],
        }
    }
    for name, ev in models:
        results['models'][name] = ev
    
    with open(OUTPUT_DIR / 'africa_nonlinear_predictor.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    elapsed = time.time() - t0
    print(f"\n⏱️ 總耗時: {elapsed:.1f} 秒")
    print(f"📁 結果: {OUTPUT_DIR / 'africa_nonlinear_predictor.json'}")

if __name__ == '__main__':
    main()
