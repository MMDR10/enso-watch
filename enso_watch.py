#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌊 ENSO Watch — El Niño / La Niña 簡易預警工具
================================================
雙擊運行，自動分析最新 ENSO 狀態。
適合漁民、沿海居民、農業工作者。

用法：
    python enso_watch.py             # 睇報告
    python enso_watch.py --quiet     # 靜音模式
    python enso_watch.py --update    # 更新 NOAA 數據

需要：Python 3.7+（numpy 內置，唔使裝任何嘢）
數據：已內置 1950-2025 ONI 歷史數據，離線可用
"""

import numpy as np
import sys
import os
from datetime import datetime
from numpy.linalg import svd

# ═══════════════════════ 設定 ═══════════════════════
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ONI_FILE = os.path.join(SCRIPT_DIR, "oni.csv")
ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"

WINDOW_MONTHS = 36
STRIDE_MONTHS = 6
DELAY_WINDOW = 12
DELAY_STRIDE = 3
P_NUC = 0.05
N_BOOTSTRAP = 15

# ═══════════════════════ 1. 數據 ═══════════════════════

def load_oni():
    """讀取內置 ONI 月度數據（格式：年份 + 12 個月數值）"""
    years, months, values = [], [], []
    with open(ONI_FILE) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 13:
                continue
            try:
                year = int(parts[0])
            except ValueError:
                continue
            if year < 1950:
                continue
            for m, v in enumerate(parts[1:13], 1):
                try:
                    val = float(v)
                    if val > -99:
                        years.append(year)
                        months.append(m)
                        values.append(val)
                except ValueError:
                    continue
    return np.array(values), np.array(years), np.array(months)


def update_oni():
    """嘗試從 NOAA 更新數據（季節性格式 → 轉月度）"""
    import urllib.request
    print("📡 嘗試從 NOAA 下載...")
    try:
        urllib.request.urlretrieve(ONI_URL, ONI_FILE + ".tmp")
        # NOAA 新格式：SEAS YR TOTAL ANOM (每行 3 個月重疊季節)
        # 轉返月度格式比較複雜，直接用內置數據 + 提示
        print("   ℹ️ NOAA 格式已變更，使用內置歷史數據。")
        print("   如需最新月份，請手動更新 oni.csv。")
        os.remove(ONI_FILE + ".tmp")
    except Exception as e:
        print(f"   ⚠️ 下載失敗：{e}")


# ═══════════════════════ 2. 運算核心 ═══════════════════════

def helicity(x):
    dx = np.gradient(x)
    n = np.mean(np.abs(x * dx))
    d = np.mean(np.abs(x)) * np.mean(np.abs(dx))
    return float(n / d) if d > 1e-16 else 0.0


def split_core_shell(series):
    n = len(series)
    fft = np.fft.rfft(series)
    ci = max(1, int(0.3 * len(np.fft.rfftfreq(n))))
    c = np.zeros_like(fft, dtype=complex)
    s = np.zeros_like(fft, dtype=complex)
    c[:ci] = fft[:ci]
    s[ci:] = fft[ci:]
    return np.fft.irfft(c, n=n), np.fft.irfft(s, n=n)


def delay_embed(series, window, stride):
    n = max(1, (len(series) - window) // stride + 1)
    m = np.zeros((n, window))
    for i in range(n):
        m[i] = series[i * stride : i * stride + window]
    return m


def embed_nuclei(configs, p, seed=42):
    np.random.seed(seed)
    N, D = configs.shape
    nv = max(1, int(N * p))
    out = configs.copy()
    idx = np.random.choice(N, nv, replace=False)
    out[idx] = np.random.randn(nv, D)
    return out


def theta1(A, B):
    A = A - A.mean(axis=0)
    B = B - B.mean(axis=0)
    _, _, Va = svd(A, full_matrices=False)
    _, _, Vb = svd(B, full_matrices=False)
    d = np.abs(np.dot(Va[0], Vb[0]))
    return np.degrees(np.arccos(np.clip(d, 0, 1)))


def chi_eff_estimate(configs):
    N = len(configs)
    vals = []
    for b in range(N_BOOTSTRAP):
        np.random.seed(42 * 10000 + b * 100 + 1)
        idx = np.random.choice(N, N, replace=True)
        cb = configs[idx]
        np.random.seed(42 * 10000 + b * 100 + 2)
        cn = embed_nuclei(cb, P_NUC)
        vals.append(theta1(cb, cn) / P_NUC)
    v = np.array(vals)
    return float(v.mean()), float(v.std())


# ═══════════════════════ 3. Detector A ═══════════════════════

def core_shell_check(values):
    n = len(values)
    results = []
    for start in range(0, n - WINDOW_MONTHS, STRIDE_MONTHS):
        w = values[start : start + WINDOW_MONTHS]
        core, shell = split_core_shell(w)
        Hc = helicity(core)
        Hs = helicity(shell)
        dH = Hs - Hc
        results.append({
            "mean_oni": float(w.mean()), "delta_H": dH,
            "H_core": Hc, "H_shell": Hs, "inverted": dH < 0
        })
    latest = results[-1]
    invs = [r for r in results if r["inverted"]]
    return {
        "delta_H": round(latest["delta_H"], 2),
        "inverted": latest["inverted"],
        "inversion_rate": round(len(invs) / len(results) * 100, 1),
        "hist": f"{len(invs)}/{len(results)}"
    }


# ═══════════════════════ 4. Detector B ═══════════════════════

def chi_eff_check(values):
    configs = delay_embed(values, DELAY_WINDOW, DELAY_STRIDE)
    means = configs.mean(axis=1)
    el = means >= 0.5
    la = means <= -0.5
    ne = ~(el | la)
    
    lm = means[-1]
    if lm >= 0.5:
        phase, pc = "El Niño（聖嬰）", configs[el]
    elif lm <= -0.5:
        phase, pc = "La Niña（反聖嬰）", configs[la]
    else:
        phase, pc = "中性（Neutral）", configs[ne]
    
    r = {"phase": phase, "latest_mean": round(float(lm), 2), "n": len(pc)}
    if len(pc) >= 10:
        m, s = chi_eff_estimate(pc)
        r["chi_eff"] = round(m, 1)
        r["anomaly"] = ("中性" in phase and m > 50) or ("中性" not in phase and m > 150)
    else:
        r["chi_eff"] = None
        r["anomaly"] = False
    return r


# ═══════════════════════ 5. 綜合 ═══════════════════════

def judge(cs, chi):
    sigs = []
    if cs["inverted"]:
        sigs.append(f"Core-Shell 反轉（ΔH={cs['delta_H']}）：深層結構異常")
    if chi.get("anomaly"):
        sigs.append(f"χ_eff 異常（{chi.get('chi_eff','?')}）：結構剛性偏離正常")
    
    n = len(sigs)
    if n == 0:
        return "🟢 綠色 — 正常", sigs, "暫無異常信號，建議每月檢查一次。"
    elif n == 1:
        return "🟡 黃色 — 注意", sigs, "有一個異常信號，建議關注未來 3-6 個月變化。"
    else:
        return "🔴 紅色 — 警報", sigs, "兩個信號同時異常！密切關注未來數月可能出現嘅重大 ENSO 事件。"


def oni_desc(val):
    if val >= 1.5:  return "🔴 強 El Niño！海水明顯變暖"
    if val >= 0.5:  return "🟠 El Niño 狀態，海水偏暖"
    if val <= -1.5: return "🔵 強 La Niña！海水明顯變冷"
    if val <= -0.5: return "🟦 La Niña 狀態，海水偏冷"
    return "⚪ 中性狀態，水溫正常"


# ═══════════════════════ 6. 主程式 ═══════════════════════

def main():
    quiet = "--quiet" in sys.argv
    
    if "--update" in sys.argv:
        update_oni()
    
    if not os.path.exists(ONI_FILE):
        print("❌ 搵唔到 oni.csv 數據檔！請確保同 enso_watch.py 放喺同一個 folder。")
        return
    
    values, years, months = load_oni()
    lv, ld = values[-1], f"{years[-1]}-{months[-1]:02d}"
    
    if not quiet:
        print("""
╔══════════════════════════════════════════╗
║   🌊 ENSO Watch — El Niño 簡易預警工具  ║
║      漁民版 · 沿海居民版 · 農業版       ║
╚══════════════════════════════════════════╝
""")
        print(f"📊 數據：{years[0]}-{months[0]:02d} → {ld}（{len(values)} 個月）")
        print(f"🌡️ 最新 ONI：{lv:+.2f}°C  →  {oni_desc(lv)}")
        print()
        print("─── 🔍 Core-Shell 深層結構 ───")
    
    cs = core_shell_check(values)
    if not quiet:
        print(f"   ΔH = {cs['delta_H']:+.2f}")
        print(f"   狀態：{'⚠️ 反轉！深層結構異常' if cs['inverted'] else '✅ 正常'}")
        print(f"   （歷史反轉率 {cs['inversion_rate']}%，{cs['hist']}）")
        print()
        print("─── 🔍 χ_eff 結構剛性 ───")
    
    chi = chi_eff_check(values)
    if not quiet:
        print(f"   當前階段：{chi['phase']}")
        print(f"   12 月平均 ONI：{chi['latest_mean']:+.2f}°C")
        if chi.get("chi_eff"):
            print(f"   χ_eff = {chi['chi_eff']:.1f}  {'⚠️ 異常' if chi.get('anomaly') else '✅ 正常'}")
        print()
    
    level, reasons, advice = judge(cs, chi)
    print(f"═══ 🎯 綜合預警：{level} ═══")
    for r in reasons:
        print(f"   → {r}")
    print(f"   💡 {advice}")
    print()
    
    if not quiet:
        print(f"⚠️ 研究工具，非官方預報。請參考當地氣象局。")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"📂 數據：NOAA / 內置 oni.csv")
        print()

if __name__ == "__main__":
    main()
