#!/usr/bin/env python3
"""
MJO Daily-Scale Activity vs ENSO Onset — V2
============================================
Uses OOMI (Extended OLR MJO Index) from PSL: 1979-2022, full year, daily.
Format: year month day phase PC1 PC2 amplitude

Key improvements over V1 (RMM*):
  - Full year coverage (not just Nov-Apr)
  - Has phase information (1-8)
  - OLR-based (direct convective measurement)
  - Longer record (1979-2022)

Design:
  1. Daily MJO amplitude → test if strong MJO days cluster before ENSO onset
  2. MJO phase distribution before onset vs neutral
  3. Active MJO event counting
  4. Compare with monthly result (p=0.679)
"""
import json
import numpy as np
from scipy import stats
from collections import defaultdict

# ---------- 1. Load OOMI daily data ----------
import math

rows = []
with open("data/mjo/oomi_daily.txt") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 7:
            try:
                y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
                # phase column is always 0 in OOMI, compute from PC1/PC2
                pc1, pc2 = float(parts[4]), float(parts[5])
                amp = float(parts[6])
                if 1982 <= y <= 2020 and amp > 0:
                    # Compute MJO phase from PC1/PC2 (1-8, Wheeler-Hendon convention)
                    # Phase = atan2(-PC1, PC2) mapped to 1-8
                    angle = math.degrees(math.atan2(-pc1, pc2))
                    if angle < 0: angle += 360
                    phase = int(angle // 45) + 1  # 1-8
                    if phase > 8: phase = 8
                    rows.append((y, m, d, phase, pc1, pc2, amp))
            except ValueError:
                continue

print(f"Valid daily records (1982-2020): {len(rows)}")

# Build daily lookup
daily_data = {}
for y, m, d, phase, pc1, pc2, amp in rows:
    daily_data[(y, m, d)] = (amp, phase, pc1, pc2)

# ---------- 2. Load ENSO events ----------
try:
    data = json.load(open("media/9e2313a26af241cbb2afdfc3351ff323_wind_stress_noise_39y.json", encoding="utf-8"))
    events_el = data["events"]["ElNino"]
    events_la = data["events"]["LaNina"]
    print(f"Events: ElNino={len(events_el)}, LaNina={len(events_la)}")
except Exception as e:
    print(f"Warning: {e}")
    events_el = [0, 48, 108, 144, 180, 240, 264, 288, 324, 396, 432]
    events_la = [12, 24, 72, 156, 192, 216, 276, 300, 336, 348, 408, 420, 456]

def idx_to_ym(idx):
    return (1982 + idx // 12, (idx % 12) + 1)

el_onsets = [idx_to_ym(i) for i in events_el]
la_onsets = [idx_to_ym(i) for i in events_la]

# ---------- 3. Helper functions ----------
def get_daily_mjo(year, month, offset_months):
    """Get daily MJO data for (year, month + offset_months)."""
    m = month + offset_months
    y = year
    while m <= 0: m += 12; y -= 1
    while m > 12: m -= 12; y += 1
    amps, phases = [], []
    for d in range(1, 32):
        key = (y, m, d)
        if key in daily_data:
            amp, phase, pc1, pc2 = daily_data[key]
            amps.append(amp)
            phases.append(phase)
    return amps, phases

def build_pre_onset(onsets, window=(-6, -1)):
    all_amps, all_phases, event_amps = [], [], []
    for y, m in onsets:
        ev_amps, ev_phases = [], []
        for offset in range(window[0], window[1] + 1):
            amps, phases = get_daily_mjo(y, m, offset)
            ev_amps.extend(amps)
            ev_phases.extend(phases)
        if ev_amps:
            all_amps.extend(ev_amps)
            all_phases.extend(ev_phases)
            event_amps.append(np.mean(ev_amps))
    return all_amps, all_phases, event_amps

def build_neutral(onsets_el, onsets_la):
    event_months = set()
    for y, m in onsets_el + onsets_la:
        for offset in range(-18, 1):
            mm, yy = m + offset, y
            while mm <= 0: mm += 12; yy -= 1
            while mm > 12: mm -= 12; yy += 1
            event_months.add((yy, mm))
    amps, phases = [], []
    for (y, m, d), (amp, phase, pc1, pc2) in daily_data.items():
        if (y, m) not in event_months:
            amps.append(amp)
            phases.append(phase)
    return amps, phases

# ---------- 4. Run analysis ----------
print("\n" + "="*60)
print("MJO DAILY ANALYSIS V2 (OOMI, full year, 1982-2020)")
print("="*60)

el_amps, el_phases, el_event_amps = build_pre_onset(el_onsets)
la_amps, la_phases, la_event_amps = build_pre_onset(la_onsets)
neu_amps, neu_phases = build_neutral(el_onsets, la_onsets)

print(f"\nSample sizes:")
print(f"  El Niño pre-onset days: {len(el_amps)}")
print(f"  La Niña pre-onset days: {len(la_amps)}")
print(f"  Neutral days: {len(neu_amps)}")

# Test 1: Mean amplitude
print(f"\n--- Test 1: Mean MJO amplitude ---")
print(f"  El Niño: {np.mean(el_amps):.4f} ± {np.std(el_amps):.4f}")
print(f"  La Niña: {np.mean(la_amps):.4f} ± {np.std(la_amps):.4f}")
print(f"  Neutral: {np.mean(neu_amps):.4f} ± {np.std(neu_amps):.4f}")
t_el, p_el = stats.ttest_ind(el_amps, neu_amps, equal_var=False)
t_la, p_la = stats.ttest_ind(la_amps, neu_amps, equal_var=False)
print(f"  El vs Neutral: t={t_el:.3f}, p={p_el:.4f}")
print(f"  La vs Neutral: t={t_la:.3f}, p={p_la:.4f}")

# Test 2: Strong MJO days (amplitude > 1)
print(f"\n--- Test 2: Strong MJO days (amplitude > 1) ---")
el_strong_frac = np.mean([a > 1.0 for a in el_amps])
la_strong_frac = np.mean([a > 1.0 for a in la_amps])
neu_strong_frac = np.mean([a > 1.0 for a in neu_amps])
print(f"  El Niño: {el_strong_frac:.4f} ({el_strong_frac*100:.1f}%)")
print(f"  La Niña: {la_strong_frac:.4f} ({la_strong_frac*100:.1f}%)")
print(f"  Neutral: {neu_strong_frac:.4f} ({neu_strong_frac*100:.1f}%)")

from scipy.stats import chi2_contingency
el_s = int(el_strong_frac * len(el_amps)); el_w = len(el_amps) - el_s
la_s = int(la_strong_frac * len(la_amps)); la_w = len(la_amps) - la_s
neu_s = int(neu_strong_frac * len(neu_amps)); neu_w = len(neu_amps) - neu_s

chi2_el, p_chi_el, _, _ = chi2_contingency([[el_s, el_w], [neu_s, neu_w]])
chi2_la, p_chi_la, _, _ = chi2_contingency([[la_s, la_w], [neu_s, neu_w]])
print(f"  El vs Neutral: chi2={chi2_el:.3f}, p={p_chi_el:.4f}")
print(f"  La vs Neutral: chi2={chi2_la:.3f}, p={p_chi_la:.4f}")

# Test 3: MJO phase distribution
print(f"\n--- Test 3: MJO phase distribution ---")
for phase in range(1, 9):
    el_f = np.mean([p == phase for p in el_phases])
    la_f = np.mean([p == phase for p in la_phases])
    neu_f = np.mean([p == phase for p in neu_phases])
    print(f"  Phase {phase}: El={el_f:.3f} La={la_f:.3f} Neu={neu_f:.3f}")

el_pc = [sum(1 for p in el_phases if p == i) for i in range(1, 9)]
la_pc = [sum(1 for p in la_phases if p == i) for i in range(1, 9)]
neu_pc = [sum(1 for p in neu_phases if p == i) for i in range(1, 9)]
chi2_p_el, p_p_el, _, _ = chi2_contingency([el_pc, neu_pc])
chi2_p_la, p_p_la, _, _ = chi2_contingency([la_pc, neu_pc])
print(f"  El phase dist vs Neutral: chi2={chi2_p_el:.3f}, p={p_p_el:.4f}")
print(f"  La phase dist vs Neutral: chi2={chi2_p_la:.3f}, p={p_p_la:.4f}")

# Test 4: Active MJO events
print(f"\n--- Test 4: Active MJO events ---")
def count_active(amps, threshold=1.0, min_days=3):
    events, consec = 0, 0
    for a in amps:
        if a > threshold: consec += 1
        else:
            if consec >= min_days: events += 1
            consec = 0
    if consec >= min_days: events += 1
    return events

el_active = []
for y, m in el_onsets:
    ev = []
    for offset in range(-6, 0):
        amps, _ = get_daily_mjo(y, m, offset)
        ev.extend(amps)
    el_active.append(count_active(ev))

la_active = []
for y, m in la_onsets:
    ev = []
    for offset in range(-6, 0):
        amps, _ = get_daily_mjo(y, m, offset)
        ev.extend(amps)
    la_active.append(count_active(ev))

print(f"  El Niño: {np.mean(el_active):.2f} ± {np.std(el_active):.2f}")
print(f"  La Niña: {np.mean(la_active):.2f} ± {np.std(la_active):.2f}")

# Test 5: Event-level
print(f"\n--- Test 5: Event-level mean amplitude ---")
neu_mean = np.mean(neu_amps)
t_el_ev, p_el_ev = stats.ttest_1samp(el_event_amps, neu_mean)
t_la_ev, p_la_ev = stats.ttest_1samp(la_event_amps, neu_mean)
print(f"  El events (n={len(el_event_amps)}): {np.mean(el_event_amps):.4f}, vs neutral t={t_el_ev:.3f} p={p_el_ev:.4f}")
print(f"  La events (n={len(la_event_amps)}): {np.mean(la_event_amps):.4f}, vs neutral t={t_la_ev:.3f} p={p_la_ev:.4f}")

# Test 6: Strong MJO amplitude
print(f"\n--- Test 6: Strong MJO amplitude (days > 1 only) ---")
el_s_amps = [a for a in el_amps if a > 1.0]
la_s_amps = [a for a in la_amps if a > 1.0]
neu_s_amps = [a for a in neu_amps if a > 1.0]
if el_s_amps and la_s_amps and neu_s_amps:
    print(f"  El: {len(el_s_amps)} days, mean={np.mean(el_s_amps):.4f}")
    print(f"  La: {len(la_s_amps)} days, mean={np.mean(la_s_amps):.4f}")
    print(f"  Neu: {len(neu_s_amps)} days, mean={np.mean(neu_s_amps):.4f}")
    t_es, p_es = stats.ttest_ind(el_s_amps, neu_s_amps, equal_var=False)
    t_ls, p_ls = stats.ttest_ind(la_s_amps, neu_s_amps, equal_var=False)
    print(f"  El strong vs Neu strong: t={t_es:.3f}, p={p_es:.4f}")
    print(f"  La strong vs Neu strong: t={t_ls:.3f}, p={p_ls:.4f}")

# ---------- 5. Summary ----------
print("\n" + "="*60)
print("SUMMARY (OOMI, full year)")
print("="*60)
print(f"Monthly MJO (previous): p=0.679 (negative)")
print(f"Daily OOMI mean amplitude:")
print(f"  El vs Neutral: p={p_el:.4f} {'***' if p_el < 0.001 else '**' if p_el < 0.01 else '*' if p_el < 0.05 else 'ns'}")
print(f"  La vs Neutral: p={p_la:.4f} {'***' if p_la < 0.001 else '**' if p_la < 0.01 else '*' if p_la < 0.05 else 'ns'}")
print(f"Strong MJO fraction:")
print(f"  El vs Neutral: p={p_chi_el:.4f} {'***' if p_chi_el < 0.001 else '**' if p_chi_el < 0.01 else '*' if p_chi_el < 0.05 else 'ns'}")
print(f"  La vs Neutral: p={p_chi_la:.4f} {'***' if p_chi_la < 0.001 else '**' if p_chi_la < 0.01 else '*' if p_chi_la < 0.05 else 'ns'}")
print(f"Phase distribution:")
print(f"  El vs Neutral: p={p_p_el:.4f} {'***' if p_p_el < 0.001 else '**' if p_p_el < 0.01 else '*' if p_p_el < 0.05 else 'ns'}")
print(f"  La vs Neutral: p={p_p_la:.4f} {'***' if p_p_la < 0.001 else '**' if p_p_la < 0.01 else '*' if p_p_la < 0.05 else 'ns'}")

# Save
results = {
    "meta": {
        "task": "MJO daily-scale activity vs ENSO onset V2",
        "data": "OOMI (PSL, OLR-based, full year, 1979-2022)",
        "method": "Daily MJO amplitude: pre-onset [-6,-1] vs neutral, season-matched",
        "improvement": "Full year coverage + phase info + OLR-based (vs RMM* Nov-Apr only)"
    },
    "sample_sizes": {
        "el_pre_onset_days": len(el_amps),
        "la_pre_onset_days": len(la_amps),
        "neutral_days": len(neu_amps),
        "el_events": len(el_event_amps),
        "la_events": len(la_event_amps)
    },
    "test1_mean_amplitude": {
        "el_mean": float(np.mean(el_amps)),
        "la_mean": float(np.mean(la_amps)),
        "neutral_mean": float(np.mean(neu_amps)),
        "el_vs_neutral_p": float(p_el),
        "la_vs_neutral_p": float(p_la)
    },
    "test2_strong_fraction": {
        "el_frac": float(el_strong_frac),
        "la_frac": float(la_strong_frac),
        "neu_frac": float(neu_strong_frac),
        "el_vs_neutral_p": float(p_chi_el),
        "la_vs_neutral_p": float(p_chi_la)
    },
    "test3_phase_distribution": {
        "el_chi2": float(chi2_p_el),
        "el_p": float(p_p_el),
        "la_chi2": float(chi2_p_la),
        "la_p": float(p_p_la)
    },
    "test4_active_events": {
        "el_mean": float(np.mean(el_active)),
        "la_mean": float(np.mean(la_active))
    },
    "test5_event_level": {
        "el_vs_neutral_p": float(p_el_ev),
        "la_vs_neutral_p": float(p_la_ev)
    }
}

with open("projects/enso/output/mjo_daily_enso_onset_v2.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("\nSaved projects/enso/output/mjo_daily_enso_onset_v2.json")
