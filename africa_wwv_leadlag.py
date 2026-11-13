#!/usr/bin/env python3
"""
Africa charging state → Pacific WWV lead-lag test（真數據）
==========================================================
11-09 懸空嘅「傳導路徑」戰場：如果非洲充電係經海洋慢路（熱含量）傳去 ENSO，
非洲充電態應該 *領先* 太平洋 WWV。falsification 條件事先寫死：

  PASS（海洋路活）：非洲 MJJ 領先 WWV 顯著正相關（lag>0），且 LOO 增量過 permutation（p<0.05）
  DEAD（海洋路死）：峰值喺 lag≤0（WWV 領先/同時），或 LOO 增量 null/負

數據（全部真數據）：
  - 非洲特徵：africa_features_528.npz（ERA5 850hPa AS/RHO，528 月）
  - WWV：NOAA PMEL Warm Water Volume 月度 anomaly（1980-01→2026-09）
  - ONI：CPC（對照用）

方法（同 11-09 ab_africa_skill 一致）：OLS、逐年 LOO、非洲 block 跨年 shuffle
permutation 1000 次（seed 20261113，保 block 內部結構）。
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
N_PERM = 1000
SEED = 20261113


def load_wwv():
    rows = {}
    with open(os.path.join(HERE, "wwv.dat")) as f:
        for line in f:
            p = line.split()
            if len(p) == 3 and p[0].isdigit() and len(p[0]) == 6:
                ym = f"{p[0][:4]}-{p[0][4:]}"
                rows[ym] = float(p[2])   # Anomaly column
    return rows


def load_africa():
    d = np.load(os.path.join(HERE, "..", "africa_features_528.npz"), allow_pickle=True)
    out = {}
    for i, m in enumerate(d["dates"]):
        out[str(m)] = (float(d["AS"][i]), float(d["RHO"][i]))
    return out


def load_oni():
    # CPC detrend.nino34（同 ab_africa_skill 用嘅源；本地 cache 冇就用 detrend Nino3.4）
    p = os.path.join(HERE, "..", "output", "ab_africa_skill.json")
    return json.load(open(p)).get("data_sources", {})


def mj(afr, y, key):
    """year Y 嘅 MJJ 均值；key 0=AS 1=RHO"""
    vals = [afr[f"{y}-{m:02d}"][key] for m in (5, 6, 7) if f"{y}-{m:02d}" in afr]
    return float(np.mean(vals)) if len(vals) == 3 else float("nan")


def slope(afr, y, key):
    v = [afr[f"{y}-{m:02d}"][key] for m in (5, 6, 7) if f"{y}-{m:02d}" in afr]
    return float(v[2] - v[0]) if len(v) == 3 else float("nan")


def djf_mean(wwv, y):
    vals = [wwv[f"{y}-12"], wwv.get(f"{y+1}-01"), wwv.get(f"{y+1}-02")]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if len(vals) == 3 else float("nan")


def loo_ols(X, y):
    n = len(y)
    preds = np.zeros(n)
    Xd = np.column_stack([np.ones(n), X])
    for i in range(n):
        mask = np.ones(n, bool)
        mask[i] = False
        beta, *_ = np.linalg.lstsq(Xd[mask], y[mask], rcond=None)
        preds[i] = Xd[i] @ beta
    return preds


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    r = float(np.corrcoef(y_true, y_pred)[0, 1])
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return r, mae


def main():
    wwv = load_wwv()
    afr = load_africa()
    print(f"WWV months: {len(wwv)} ({min(wwv)}→{max(wwv)})  Africa months: {len(afr)}")

    # ── Test 1: lead-lag scan ──
    # lag 定義：相對非洲 MJJ(Y)（=7 月 Y），lag k 個月 = WWV(Y,k) anomaly
    # lag −5(=2月Y) … +17(=12月Y+1)
    afr_mjj = {y: (mj(afr, y, 0), mj(afr, y, 1)) for y in range(1982, 2026)}
    lag_rows = []
    for k in range(-5, 18):
        aa, bb, ys = [], [], []
        for y in range(1982, 2026):
            mm = 7 + k
            yy = y + (mm - 1) // 12
            mo = (mm - 1) % 12 + 1
            key = f"{yy}-{mo:02d}"
            if key in wwv and not np.isnan(afr_mjj[y][0]):
                aa.append(afr_mjj[y][0])       # AS MJJ
                bb.append(wwv[key])
                ys.append(y)
        r = float(np.corrcoef(aa, bb)[0, 1])
        lag_rows.append({"lag_m": k, "month": f"lag{k}", "r_AS_wwv": round(r, 3), "n": len(aa)})
    rs = [x["r_AS_wwv"] for x in lag_rows]
    best = lag_rows[int(np.argmax(np.abs(rs)))]
    print("\n== Test 1: lead-lag corr(AS_MJJ, WWV_anom) ==")
    print(" ".join(f"{x['lag_m']:+d}:{x['r_AS_wwv']:+.2f}" for x in lag_rows))
    print(f"peak |r| = {best['r_AS_wwv']:+.3f} @ lag {best['lag_m']:+d} 個月")

    # ── Test 2: LOO increment — 非洲特徵加落 WWV persistence 預測 DJF WWV ──
    years = [y for y in range(1982, 2026)]
    y_target = np.array([djf_mean(wwv, y) for y in years])
    y_mjj = np.array([wwv.get(f"{y}-07", np.nan) for y in years])
    trend = np.array([wwv.get(f"{y}-07", np.nan) - wwv.get(f"{y}-02", np.nan) for y in years])
    a_as = np.array([afr_mjj[y][0] for y in years])
    a_rho = np.array([afr_mjj[y][1] for y in years])
    a_as_sl = np.array([slope(afr, y, 0) for y in years])
    a_rho_sl = np.array([slope(afr, y, 1) for y in years])
    ok = ~(np.isnan(y_target) | np.isnan(y_mjj) | np.isnan(trend) | np.isnan(a_as))
    y_target, y_mjj, trend = y_target[ok], y_mjj[ok], trend[ok]
    a_as, a_rho, a_as_sl, a_rho_sl = a_as[ok], a_rho[ok], a_as_sl[ok], a_rho_sl[ok]
    n = len(y_target)
    print(f"\n== Test 2: LOO predict DJF WWV anomaly (n={n}) ==")

    Xa = np.column_stack([y_mjj, trend])
    Xb = np.column_stack([y_mjj, trend, a_as, a_rho, a_as_sl, a_rho_sl])
    ra, maea = metrics(y_target, loo_ols(Xa, y_target))
    rb, maeb = metrics(y_target, loo_ols(Xb, y_target))
    print(f"A_full (WWV MJJ+trend):     r={ra:.3f} MAE={maea:.4f}")
    print(f"B (+Africa AS/RHO+slopes):  r={rb:.3f} MAE={maeb:.4f}  Δr={rb-ra:+.3f}")

    # permutation：非洲 block 跨年 shuffle（4 列同步打亂）
    rng = np.random.default_rng(SEED)
    ablock = np.column_stack([a_as, a_rho, a_as_sl, a_rho_sl])
    cnt = 0
    prs = []
    for _ in range(N_PERM):
        idx = rng.permutation(n)
        Xp = np.column_stack([y_mjj, trend, ablock[idx]])
        rp, _ = metrics(y_target, loo_ols(Xp, y_target))
        prs.append(rp)
        if rp >= rb:
            cnt += 1
    p_perm = (cnt + 1) / (N_PERM + 1)
    print(f"permutation p(one-sided, B≥null) = {p_perm:.3f}  (null r mean={np.mean(prs):.3f})")

    # ── Test 3（對照）：corr(AS_MJJ, DJF ONI) 同 corr(AS_MJJ, DJF WWV) 直接對比 ──
    # ONI 由 ab_africa_skill.json 無存序列 → 用 detrend Nino3.4 現拉唔到，改用 WWV 做唯一目標
    r_direct = float(np.corrcoef(a_as, y_target)[0, 1])
    r_direct_rho = float(np.corrcoef(a_rho, y_target)[0, 1])
    print(f"\ndirect corr(AS_MJJ, DJF WWV) = {r_direct:+.3f}   corr(RHO_MJJ, DJF WWV) = {r_direct_rho:+.3f}")

    # ── 判決 ──
    if best["lag_m"] > 0 and p_perm < 0.05:
        verdict = "PASS — 非洲領先 WWV 且增量過 permutation：海洋慢路有活證據"
    elif best["lag_m"] <= 0:
        verdict = f"DEAD — 峰值喺 lag {best['lag_m']:+d}（WWV 領先/同時）：海洋慢路方向唔成立"
    else:
        verdict = "NULL — 非洲領先但增量唔過 permutation：海洋慢路未證實（信息冗餘）"
    print(f"\n判決: {verdict}")

    out = {
        "experiment": "Africa charging → Pacific WWV lead-lag (real-data transmission-path test)",
        "prompted_by": "MKP 2026-11-13: ENSO未爆是非洲一直有傳能量入去又什解釋 → 傳導路徑 falsifiable 測試",
        "data": {"africa": "africa_features_528.npz (ERA5 850hPa AS/RHO, 528 months)",
                 "wwv": "NOAA PMEL wwv.dat 1980-01→2026-09 (anomaly column)",
                 "real_data_only": True},
        "design": {"target": "DJF WWV anomaly (Dec Y, Jan-Feb Y+1)",
                   "forecast_time": "end of July Y (charging window complete)",
                   "model": "OLS, LOO over 44 winters", "perm": f"block shuffle {N_PERM}x seed {SEED}",
                   "falsification": "PASS: Africa leads (lag>0) + LOO increment p<0.05; DEAD: peak lag<=0; NULL: lead but no increment"},
        "test1_leadlag": {"rows": lag_rows, "peak": best},
        "test2_loo": {"n": int(n), "A_full_r": round(ra, 3), "A_full_mae": round(maea, 4),
                      "B_plus_africa_r": round(rb, 3), "B_plus_africa_mae": round(maeb, 4),
                      "delta_r": round(rb - ra, 3), "perm_p": round(p_perm, 3),
                      "null_r_mean": round(float(np.mean(prs)), 3)},
        "test3_direct": {"corr_AS_MJJ_djf_wwv": round(r_direct, 3),
                         "corr_RHO_MJJ_djf_wwv": round(r_direct_rho, 3)},
        "verdict": verdict,
    }
    with open(os.path.join(HERE, "africa_wwv_leadlag.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("saved africa_wwv_leadlag.json")


if __name__ == "__main__":
    main()
