#!/usr/bin/env python3
"""
skill A/B：非洲預測鏈增量測試（MKP 2026-11-09 指示：「分單獨一項加埋非洲預測鏈看看有沒有分別」）
====================================================================================
問題：喺同一預測 pipeline 之內，加埋非洲電容特徵到底有冇帶來增量 skill？

設計（全部同年對齊，冇 look-ahead）：
  預測時點 = 年 Y 七月底（充電窗 [5,6,7] 完結）
  目標     = winter DJF ONI(Y→Y+1) = mean(Dec(Y), Jan(Y+1), Feb(Y+1))
  相位     = DJF ONI ≥ +0.5 El_Nino / ≤ −0.5 La_Nina / 其餘 Neutral

  特徵組（全部只用 Y 年七月底前資訊）：
  ONI_MJJ    = mean(ONI May,Jun,Jul Y)          — persistence（慣常基準）
  ONI_trend  = ONI_MJJ − mean(ONI Feb,Mar,Apr Y) — 慣常趨勢項
  AS_MJJ/RHO_MJJ    = 非洲特徵 MJJ 均值          （lean block）
  AS_slope/RHO_slope = MJJ 三月線性斜率           （full block 加項）

  模型 = OLS（numpy 手寫，無 sklearn）→ 預測 DJF ONI → ±0.5 閾值分相位
  評估 = 逐年 LOO（44 個 winter：1982-83 … 2025-26）
  顯著性 = 非洲 block 跨年 shuffle permutation 1000 次（保 block 內部結構）

  變體：
  A_pers    = ONI_MJJ                          （最簡基準）
  A_full    = ONI_MJJ + ONI_trend              （較強基準，B 嘅公平對手）
  B1        = A_full + lean block               （加非洲項）
  B2        = A_full + lean + full              （加非洲項闊版）
  C_africa  = 非洲 block 獨跑（分單獨一項，唔用 ONI）

數據：project/enso/africa_features_528.npz（1982-04→2025-12，AS/RHO）
     project/enso/release/oni.csv（NOAA CPC ONI，1950→2026）
紀律：唔跑唔寫 — 本 script 產出 output JSON + report，數字以 output 為準。
"""

import json
import time
from pathlib import Path

import numpy as np

BASE = Path('/app/working/workspaces/tygtDc/project/enso')
NPZ = BASE / 'africa_features_528.npz'
ONI_CSV = BASE / 'release' / 'oni.csv'
OUT_DIR = BASE / 'output'
OUT_DIR.mkdir(exist_ok=True)

MONTHS = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']


def load_oni():
    """wide CSV → dict[(year, month1to12)] = anomaly"""
    oni = {}
    for line in ONI_CSV.read_text().splitlines():
        parts = line.split()
        if len(parts) != 13:
            continue
        try:
            yr = int(parts[0])
        except ValueError:
            continue
        vals = [float(x) for x in parts[1:]]
        for m, v in enumerate(vals, start=1):
            oni[(yr, m)] = v
    return oni


def winter_djf_oni(oni, y):
    """DJF ONI of winter y→y+1 = mean(Dec(y), Jan(y+1), Feb(y+1))"""
    try:
        return np.mean([oni[(y, 12)], oni[(y + 1, 1)], oni[(y + 1, 2)]])
    except KeyError:
        return np.nan


def phase_of(v):
    return 'El_Nino' if v >= 0.5 else ('La_Nina' if v <= -0.5 else 'Neutral')


def loo_ols(X, y):
    """逐年留一 OLS，回傳 LOO 預測值"""
    n = len(y)
    preds = np.empty(n)
    for i in range(n):
        idx = [j for j in range(n) if j != i]
        beta, *_ = np.linalg.lstsq(X[idx], y[idx], rcond=None)
        preds[i] = X[i] @ beta
    return preds


def metrics(y_true, y_pred):
    from math import sqrt
    ph_true = [phase_of(v) for v in y_true]
    ph_pred = [phase_of(v) for v in y_pred]
    acc = np.mean([a == b for a, b in zip(ph_true, ph_pred)])
    mae = float(np.mean(np.abs(y_true - y_pred)))
    r = float(np.corrcoef(y_true, y_pred)[0, 1])
    # 分相位準確率（per-class recall）
    per_class = {}
    for ph in ['El_Nino', 'La_Nina', 'Neutral']:
        sel = [i for i, p in enumerate(ph_true) if p == ph]
        if sel:
            hit = sum(1 for i in sel if ph_pred[i] == ph)
            per_class[ph] = {'n': len(sel), 'hit': hit, 'recall': hit / len(sel)}
    # 基準：climatology 永遠估最多嗰類
    counts = {ph: ph_true.count(ph) for ph in set(ph_true)}
    clim_acc = max(counts.values()) / len(ph_true)
    return {'acc': float(acc), 'mae_oni': mae, 'r_oni': r,
            'per_class': per_class, 'clim_acc': float(clim_acc),
            'class_counts': counts}


def main():
    t0 = time.time()
    oni = load_oni()
    d = np.load(NPZ)
    dates, AS, RHO = d['dates'], d['AS'].astype(float), d['RHO'].astype(float)
    feat_month = {}
    for dt, a, r in zip(dates, AS, RHO):
        yr, mo = int(dt[:4]), int(dt[5:7])
        feat_month[(yr, mo)] = (a, r)

    # 組年份樣本：非洲 MJJ 特徵 + 目標 DJF ONI
    rows = []
    for y in range(1982, 2026):
        try:
            a_mjj = [feat_month[(y, m)][0] for m in (5, 6, 7)]
            r_mjj = [feat_month[(y, m)][1] for m in (5, 6, 7)]
            a_fma = [feat_month[(y, m)][0] for m in (2, 3, 4)]
            r_fma = [feat_month[(y, m)][1] for m in (2, 3, 4)]
        except KeyError:
            continue
        oni_mjj = np.mean([oni[(y, m)] for m in (5, 6, 7)])
        oni_fma = np.mean([oni[(y, m)] for m in (2, 3, 4)])
        tgt = winter_djf_oni(oni, y)
        if not np.isfinite(tgt):
            continue
        xs = np.arange(3.0)
        rows.append(dict(
            year=y, target=tgt,
            oni_mjj=float(oni_mjj), oni_trend=float(oni_mjj - oni_fma),
            as_mean=float(np.mean(a_mjj)), rho_mean=float(np.mean(r_mjj)),
            as_slope=float(np.polyfit(xs, a_mjj, 1)[0]),
            rho_slope=float(np.polyfit(xs, r_mjj, 1)[0]),
        ))

    n = len(rows)
    y = np.array([r['target'] for r in rows])
    ones = np.ones(n)

    oni_mjj = np.array([r['oni_mjj'] for r in rows])
    oni_tr = np.array([r['oni_trend'] for r in rows])
    as_m = np.array([r['as_mean'] for r in rows])
    rho_m = np.array([r['rho_mean'] for r in rows])
    as_s = np.array([r['as_slope'] for r in rows])
    rho_s = np.array([r['rho_slope'] for r in rows])

    X_pers = np.column_stack([ones, oni_mjj])
    X_full = np.column_stack([ones, oni_mjj, oni_tr])
    X_b1 = np.column_stack([ones, oni_mjj, oni_tr, as_m, rho_m])
    X_b2 = np.column_stack([ones, oni_mjj, oni_tr, as_m, rho_m, as_s, rho_s])
    X_c = np.column_stack([ones, as_m, rho_m, as_s, rho_s])

    variants = {
        'A_pers_ONI_MJJ': X_pers,
        'A_full_ONI_MJJ+trend': X_full,
        'B1_A_full+africa_lean': X_b1,
        'B2_A_full+africa_full': X_b2,
        'C_africa_only': X_c,
    }
    results = {}
    preds_store = {}
    for name, X in variants.items():
        preds = loo_ols(X, y)
        preds_store[name] = preds
        results[name] = metrics(y, preds)

    # ── permutation：非洲 block 跨年 shuffle，1000 次，睇 Δacc(B1−A_full) ──
    rng = np.random.default_rng(20261109)
    n_perm = 1000
    acc_a = results['A_full_ONI_MJJ+trend']['acc']
    acc_b1 = results['B1_A_full+africa_lean']['acc']
    acc_b2 = results['B2_A_full+africa_full']['acc']
    delta_b1 = acc_b1 - acc_a
    delta_b2 = acc_b2 - acc_a
    perm_d1 = np.empty(n_perm)
    perm_d2 = np.empty(n_perm)
    for k in range(n_perm):
        idx = rng.permutation(n)
        Xp_b1 = np.column_stack([ones, oni_mjj, oni_tr, as_m[idx], rho_m[idx]])
        Xp_b2 = np.column_stack([ones, oni_mjj, oni_tr, as_m[idx], rho_m[idx],
                                 as_s[idx], rho_s[idx]])
        pa1 = loo_ols(Xp_b1, y)
        pa2 = loo_ols(Xp_b2, y)
        perm_d1[k] = metrics(y, pa1)['acc'] - acc_a
        perm_d2[k] = metrics(y, pa2)['acc'] - acc_a
    p_b1 = float(np.mean(perm_d1 >= delta_b1 - 1e-12))
    p_b2 = float(np.mean(perm_d2 >= delta_b2 - 1e-12))

    # ── 前後半期穩定性（1982-2003 vs 2004-2025 各 22 個 winter）──
    def half_split(name):
        preds = preds_store[name]
        out = {}
        for tag, lo, hi in [('1982-2003', 0, 22), ('2004-2025', 22, n)]:
            sub_t, sub_p = y[lo:hi], preds[lo:hi]
            out[tag] = {
                'acc': float(np.mean([phase_of(a) == phase_of(b)
                                      for a, b in zip(sub_t, sub_p)])),
                'mae': float(np.mean(np.abs(sub_t - sub_p))),
            }
        return out

    report = {
        'experiment': 'skill A/B — Africa capacitor incremental test',
        'prompted_by': 'MKP 2026-11-09: 分單獨一項加埋非洲預測鏈看看有沒有分別',
        'design': {
            'forecast_time': 'end of July, year Y (charging window [5,6,7])',
            'target': 'DJF ONI of winter Y→Y+1 = mean(Dec Y, Jan Y+1, Feb Y+1)',
            'lead_time': '~5 months to DJF start',
            'model': 'OLS per-variant, LOO over winters',
            'features_note': 'Africa block = AS/RHO (africa_features_528.npz); ONI block from release/oni.csv',
            'n_winters': n,
            'years': [rows[0]['year'], rows[-1]['year']],
            'class_counts': results['A_pers_ONI_MJJ']['class_counts'],
        },
        'variants': results,
        'delta_vs_A_full': {
            'B1_minus_A_full_acc': float(delta_b1),
            'B2_minus_A_full_acc': float(delta_b2),
            'B1_perm_p_one_sided': p_b1,
            'B2_perm_p_one_sided': p_b2,
            'n_perm': n_perm,
        },
        'half_period_stability': {
            'A_full': half_split('A_full_ONI_MJJ+trend'),
            'B1': half_split('B1_A_full+africa_lean'),
            'C_africa_only': half_split('C_africa_only'),
        },
        'context_monitor_claim': {
            'note': 'operational monitor M1/M4/M6 use z850/rho_700 features — different construction from AS/RHO block here; numbers not directly comparable',
            'M1': 0.718, 'M4': 0.769, 'M6': 0.795,
        },
        'runtime_sec': round(time.time() - t0, 1),
    }

    out_json = OUT_DIR / 'ab_africa_skill.json'
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n[written] {out_json}")


if __name__ == '__main__':
    main()
