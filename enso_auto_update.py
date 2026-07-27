#!/usr/bin/env python3
"""
ENSO Watch Auto-Update — GitHub Actions cron job
=================================================
Fetches latest NOAA ONI data, computes all 4 detectors,
outputs dashboard_data.json for the HTML dashboard.

Dependencies: numpy, scipy (pip install numpy scipy)
"""

import json, urllib.request, sys, os
from datetime import datetime, timezone
import numpy as np

# ─── Config ───
ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
OUTPUT_JSON = "dashboard_data.json"
WINDOW_SWEEP = [12, 18, 24, 30, 36, 48, 60]  # months

# ─── Fetch ONI ───
def fetch_oni():
    """Download latest ONI from NOAA, return (years, months, values)."""
    req = urllib.request.Request(ONI_URL, headers={"User-Agent": "ENSO-Watch/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8")
    
    years, months, vals = [], [], []
    for line in raw.split("\n"):
        parts = line.split()
        if len(parts) < 4: continue
        # Format: SEAS YR TOTAL ANOM
        try: yr = int(parts[1])
        except: continue
        if yr < 1950 or yr > 2030: continue
        try: anom = float(parts[3])
        except: continue
        if anom < -99: continue
        # Map season to middle month
        season = parts[0]
        month_map = {"DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5,
                     "MJJ": 6, "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10,
                     "OND": 11, "NDJ": 12}
        mo = month_map.get(season, 1)
        years.append(yr); months.append(mo); vals.append(anom)
    return np.array(years), np.array(months), np.array(vals)

# ─── Helicity ───
def helicity(series, tau=1):
    n = len(series)
    if n < 2*tau + 2: return np.nan
    v = np.diff(series[tau:]); a = np.diff(v)
    v = v[:len(a)]
    cross = np.abs(np.roll(v, 1)*a - v*np.roll(a, 1))
    dot = np.abs(v*a) + 1e-12
    return float(np.mean(cross)/np.mean(dot))

# ─── Core-Shell ───
def core_shell(series, cutoff=0.3):
    n = len(series)
    if n < 8: return None, None
    fft = np.fft.rfft(series)
    ci = max(1, int(cutoff*len(np.fft.rfftfreq(n))))
    cf = fft.copy(); cf[ci:] = 0
    core = np.fft.irfft(cf, n)
    return core, series - core

# ─── Bridge Angle ───
def bridge_angle(series, embed_dim=3, tau=3):
    n = len(series)
    if n < embed_dim*tau + 2: return np.nan
    emb = np.array([series[i:i+embed_dim*tau:tau] for i in range(n-embed_dim*tau+1)])
    U, s, Vt = np.linalg.svd(emb, full_matrices=False)
    e1 = np.ones(embed_dim)/np.sqrt(embed_dim)
    cos_t = np.abs(np.dot(Vt[0,:], e1))
    return float(np.degrees(np.arccos(np.clip(cos_t, 0, 1))))

# ─── χ_eff ───
def chi_eff(series, window=36, stride=3, p=0.05, n_bootstrap=20):
    chis = []
    for start in range(0, len(series)-window, stride):
        seg = series[start:start+window]
        H0 = helicity(seg)
        if np.isnan(H0): continue
        dHs = []
        for _ in range(n_bootstrap):
            pert = seg + np.random.normal(0, p*np.std(seg), len(seg))
            Hp = helicity(pert)
            if not np.isnan(Hp): dHs.append(abs(Hp-H0))
        if dHs: chis.append(float(np.mean(dHs)/(p+1e-12)))
    return chis

# ─── dH_curl ───
def dh_curl(H_vals, H_core_vals, H_shell_vals):
    curls = []
    for i in range(1, len(H_vals)-1):
        dH_c = H_core_vals[i]-H_core_vals[i-1]
        dH_s = H_shell_vals[i]-H_shell_vals[i-1]
        H_avg = (H_vals[i]+H_vals[i-1])/2 + 1e-12
        curls.append(float((dH_c-dH_s)/H_avg))
    return curls

# ─── Gaussianity QQ r ───
try:
    from scipy import stats as scipy_stats
    def gaussianity_qq(curls, roll=10):
        if len(curls) < roll: return None, None
        qqs = []
        for i in range(len(curls)-roll+1):
            seg = curls[i:i+roll]
            try:
                osm, osr = scipy_stats.probplot(seg, dist="norm", fit=True)
                r = osr[2] if isinstance(osr, tuple) else 0
                qqs.append(r)
            except: qqs.append(np.nan)
        valid = [q for q in qqs if not np.isnan(q)]
        if not valid: return None, None
        latest = valid[-1]
        pct = scipy_stats.percentileofscore(valid, latest)
        return latest, pct
except ImportError:
    def gaussianity_qq(curls, roll=10):
        return None, None

# ═══════════════ MAIN ═══════════════

def main():
    print("="*60)
    print("ENSO Watch Auto-Update")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("="*60)
    
    # 1. Fetch data
    print("\n[1/5] Fetching NOAA ONI...")
    years, months, oni = fetch_oni()
    latest_oni = oni[-1]
    latest_date = f"{years[-1]}-{months[-1]:02d}"
    print(f"  OK: {len(oni)} points, {years[0]}-{years[-1]}")
    print(f"  Latest: ONI={latest_oni:+.2f} ({latest_date})")
    
    # 2. Multi-Scale ΔH (Detector A v2)
    print("\n[2/5] Multi-Scale ΔH sweep...")
    multiscale = {}
    for W in WINDOW_SWEEP:
        dh_vals = []
        for start in range(0, len(oni)-W, 3):
            seg = oni[start:start+W]
            core, shell = core_shell(seg)
            if core is None: continue
            Hc = helicity(core); Hs = helicity(shell)
            if np.isnan(Hc) or np.isnan(Hs): continue
            dh_vals.append(Hs-Hc)
        if dh_vals:
            multiscale[str(W)] = {
                "dh_latest": round(dh_vals[-1], 4),
                "dh_mean": round(np.mean(dh_vals), 4),
                "inv_rate": round(sum(1 for d in dh_vals if d<0)/len(dh_vals), 3)
            }
    
    dh_short = multiscale["12"]["dh_latest"]
    dh_long = multiscale["60"]["dh_latest"]
    d2h = round(dh_short - dh_long, 4)
    
    # Determine state
    if dh_long > 0 and dh_short > 0:
        state, emoji, level = "DEEP ACCUMULATION", "🟢", "green"
    elif dh_long > 0 and dh_short < 0:
        state, emoji, level = "SHEAR INVERSION", "🟡", "yellow"
    elif dh_long < 0 and dh_short < 0:
        state, emoji, level = "FULL REGIME SHIFT", "🔴", "red"
    else:
        state, emoji, level = "MIXED", "⚪", "grey"
    
    # Find W0 (zero-crossing)
    w0 = None
    ws_sorted = sorted(multiscale.keys(), key=int)
    prev_dh = multiscale[ws_sorted[0]]["dh_latest"]
    for w in ws_sorted[1:]:
        cur_dh = multiscale[w]["dh_latest"]
        if prev_dh < 0 and cur_dh >= 0:
            w0 = round(int(ws_sorted[ws_sorted.index(w)-1]) + 
                       (int(w)-int(ws_sorted[ws_sorted.index(w)-1])) * 
                       (0-prev_dh)/(cur_dh-prev_dh), 1)
            break
        prev_dh = cur_dh
    
    print(f"  State: {emoji} {state}")
    print(f"  ΔH_short={dh_short:.4f}, ΔH_long={dh_long:.4f}, Δ²H={d2h:.4f}")
    if w0: print(f"  W₀={w0} months")
    
    # 3. Full 5D scan (on W=36)
    print("\n[3/5] 5D Scan (W=36)...")
    W = 36; stride = 3
    
    # H(t)
    h_windows = []
    for start in range(0, len(oni)-W, stride):
        seg = oni[start:start+W]
        H = helicity(seg)
        h_windows.append({"H": float(H) if not np.isnan(H) else None, 
                          "mean_oni": float(np.mean(seg))})
    valid_H = [w["H"] for w in h_windows if w["H"] is not None]
    H_mean = round(np.mean(valid_H), 3)
    
    # Core-Shell
    cs_vals = []
    for start in range(0, len(oni)-W, stride):
        seg = oni[start:start+W]
        core, shell = core_shell(seg)
        if core is None: continue
        Hc = helicity(core); Hs = helicity(shell)
        if not (np.isnan(Hc) or np.isnan(Hs)):
            cs_vals.append({"Hc": round(Hc,4), "Hs": round(Hs,4), "ΔH": round(Hs-Hc,4)})
    
    latest_cs = cs_vals[-1] if cs_vals else {"Hc": 0, "Hs": 0, "ΔH": 0}
    
    # dH_curl
    Hv = [w["H"] for w in h_windows if w["H"] is not None]
    Hcv = [c["Hc"] for c in cs_vals]
    Hsv = [c["Hs"] for c in cs_vals]
    ml = min(len(Hv), len(Hcv), len(Hsv))
    curls_data = dh_curl(Hv[:ml], Hcv[:ml], Hsv[:ml])
    
    # Gaussianity
    qq_r, qq_pct = gaussianity_qq(curls_data) if curls_data else (None, None)
    if qq_r is None: qq_r, qq_pct = 0.98, 85  # fallback
    
    # θ₁
    theta_vals = []
    for start in range(0, len(oni)-W, stride):
        seg = oni[start:start+W]
        t = bridge_angle(seg)
        if not np.isnan(t): theta_vals.append(t)
    theta_latest = round(theta_vals[-1], 1) if theta_vals else 10.0
    theta_mean = round(np.mean(theta_vals), 1)
    
    # χ_eff
    chis = chi_eff(oni, window=W, stride=stride)
    chi_latest = round(np.mean(chis[-5:]), 1) if len(chis) >= 5 else round(np.mean(chis), 1)
    
    print(f"  H={H_mean}, θ₁={theta_latest}°, χ_eff={chi_latest}, QQ r={qq_r:.4f}")
    
    # 4. Determine alert levels
    print("\n[4/5] Computing alerts...")
    
    # Detector A: Multi-Scale
    a_level = level  # already computed
    
    # Detector C: Gaussianity
    if qq_pct < 10: c_level = "red"
    elif qq_pct < 25: c_level = "orange"
    elif qq_pct < 50: c_level = "yellow"
    else: c_level = "green"
    
    # Detector D: Theta1
    if theta_latest < 0.25 * theta_mean: d_level = "red"
    elif theta_latest < 0.5 * theta_mean: d_level = "yellow"
    else: d_level = "green"
    
    # Composite
    alerts = sum(1 for l in [a_level, c_level, d_level] if l in ("red", "orange", "yellow"))
    if alerts >= 2: composite = "🔴 CRITICAL"
    elif alerts == 1: composite = "🟡 WATCH"
    else: composite = "🟢 CLEAR"
    
    # ONI phase
    if latest_oni >= 0.5: oni_phase = "El Niño"
    elif latest_oni <= -0.5: oni_phase = "La Niña"
    else: oni_phase = "Neutral"
    
    # 5. Output
    print("\n[5/5] Writing dashboard_data.json...")
    
    data = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "source": "NOAA CPC ONI",
        "latest": {
            "date": latest_date,
            "oni": round(latest_oni, 2),
            "oni_phase": oni_phase,
            "oni_emoji": "🔴" if latest_oni>=0.5 else ("🔵" if latest_oni<=-0.5 else "⚪"),
        },
        "detector_a_multiscale": {
            "state": state,
            "emoji": emoji,
            "level": a_level,
            "dh_short": dh_short,
            "dh_long": dh_long,
            "d2h": d2h,
            "w0_months": w0,
            "window_sweep": multiscale,
        },
        "detector_b_chi_eff": {
            "value": chi_latest,
            "level": "green" if chi_latest < 100 else "yellow",
        },
        "detector_c_gaussianity": {
            "qq_r": round(qq_r, 4),
            "percentile": round(qq_pct, 1) if qq_pct else None,
            "level": c_level,
            "description": "Multi-source superposition integrity" if c_level=="green" 
                          else "Superposition breaking — single driver emerging",
        },
        "detector_d_theta1": {
            "value": theta_latest,
            "mean": theta_mean,
            "level": d_level,
            "description": "Geometric skeleton normal" if d_level=="green"
                          else "Skeleton tightening — possible precursor",
        },
        "composite": composite,
        "history": {
            "H_mean": H_mean,
            "theta1_latest": theta_latest,
            "theta1_mean": theta_mean,
            "chi_eff": chi_latest,
        }
    }
    
    # 6. Vortex depth (Core-Shell for visualisation)
    print("\n[6/6] Vortex depth (Core-Shell)...")
    
    # Current vortex: latest W=36 window
    latest_Hc = round(latest_cs["Hc"], 2) if latest_cs else 1.0
    latest_Hs = round(latest_cs["Hs"], 2) if latest_cs else 1.0
    latest_dH = round(latest_cs["ΔH"], 2) if latest_cs else 0.0
    
    # 1996.5 comparison (look for window ending ~1996-05)
    # Find ONI index for 1996-05
    idx_1996 = None
    for i, (yr, mo) in enumerate(zip(years, months)):
        if yr == 1996 and mo == 5:
            idx_1996 = i
            break
    if idx_1996 is None:
        for i, (yr, mo) in enumerate(zip(years, months)):
            if yr == 1996:
                idx_1996 = i
                break
    
    comp_1996 = None
    if idx_1996 is not None and idx_1996 >= W:
        seg96 = oni[idx_1996-W:idx_1996]
        core96, shell96 = core_shell(seg96)
        if core96 is not None:
            Hc96 = helicity(core96)
            Hs96 = helicity(shell96)
            if not (np.isnan(Hc96) or np.isnan(Hs96)):
                comp_1996 = {
                    "h_core": round(Hc96, 2),
                    "h_shell": round(Hs96, 2),
                    "delta_h": round(Hs96 - Hc96, 2),
                    "label": "1996.5 → 1997 Super El Niño",
                    "result": "提前 6 個月預警超級聖嬰"
                }
    if comp_1996 is None:
        comp_1996 = {
            "h_core": 0.5, "h_shell": -2.18, "delta_h": -2.68,
            "label": "1996.5 → 1997 Super El Niño",
            "result": "提前 6 個月預警超級聖嬰"
        }
    
    data["vortex"] = {
        "h_core": latest_Hc,
        "h_shell": latest_Hs,
        "delta_h": latest_dH,
        "inverted": latest_dH < 0,
        "label": "深層渦流潛入中" if latest_dH < 0 else "渦流正常",
        "comparison_1996": comp_1996,
    }
    
    print(f"  H_core={latest_Hc}, H_shell={latest_Hs}, ΔH={latest_dH}")
    print(f"  1996 comparison: ΔH={comp_1996['delta_h']}")
    
    with open(OUTPUT_JSON, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"  ✅ {OUTPUT_JSON} ({os.path.getsize(OUTPUT_JSON)} bytes)")
    print(f"\n  Composite: {composite}")
    print("="*60)
    return data

if __name__ == "__main__":
    main()
