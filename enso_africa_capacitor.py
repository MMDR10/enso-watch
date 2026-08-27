#!/usr/bin/env python3
"""
Africa Capacitor Monitor — ENSO 6-month ahead prediction
========================================================
Downloads ERA5 850hPa/700hPa geopotential for Africa-Indian Ocean region,
computes capacitor charge features, predicts winter ENSO state.

Model: M1 (850hPa linear) + M4 (850hPa + 700hPa ρ_std nonlinear)
Training: 1982-2020, LOO cross-validation
Accuracy: M1 71.8% / M4 76.9% overall, Neutral 78.6% (M4)

Requires: CDS_URL + CDS_KEY secrets (ERA5 API)
"""

import json, os, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ─── Config ───
OUTPUT_JSON = "africa_capacitor_data.json"
OUTPUT_HISTORY = "africa_capacitor_history.json"

# Africa-Indian Ocean region
AREA = [35, -15, 15, 50]  # N, W, S, E
LEVELS = [850, 700]

# Charging window: May-Jul
CHARGING_MONTHS = [5, 6, 7]

# M1 coefficients (from LOO training 1982-2020)
M1_SLOPE = 0.011133
M1_INTERCEPT = -164.434

# M4 coefficients [z850, rho_700, intercept]
M4_COEFFS = [0.008897, 835.962, -132.100]

# Historical accuracy
HIST_ACCURACY = {
    "M1": {"direction": 0.718, "el_nino": 0.833, "la_nina": 0.769, "neutral": 0.571},
    "M4": {"direction": 0.769, "el_nino": 0.667, "la_nina": 0.846, "neutral": 0.786}
}


def download_era5(year, month):
    """Download ERA5 monthly mean geopotential for Africa region."""
    try:
        import cdsapi
    except ImportError:
        print("ERROR: cdsapi not installed")
        return None

    cds_url = os.environ.get("CDS_URL", "")
    cds_key = os.environ.get("CDS_KEY", "")
    if not cds_key:
        print("WARNING: No CDS_KEY, skipping Africa capacitor monitor")
        return None

    c = cdsapi.Client(url=cds_url, key=cds_key)
    outfile = f"/tmp/z3d_{year}_{month:02d}.nc"

    try:
        c.retrieve(
            'reanalysis-era5-pressure-levels-monthly-means',
            {
                'product_type': 'monthly_averaged_reanalysis',
                'variable': 'geopotential',
                'pressure_level': [str(l) for l in LEVELS],
                'year': str(year),
                'month': f'{month:02d}',
                'time': '00:00',
                'area': AREA,
                'format': 'netcdf',
            },
            outfile
        )
        return outfile
    except Exception as e:
        print(f"ERROR downloading ERA5 {year}-{month:02d}: {e}")
        return None


def compute_features(nc_file):
    """Compute Africa capacitor features from ERA5 NetCDF."""
    try:
        import xarray as xr
    except ImportError:
        # Fallback: use netCDF4
        import netCDF4 as nc
        ds = nc.Dataset(nc_file)
        z = ds.variables['z'][:]
        ds.close()
    else:
        ds = xr.open_dataset(nc_file)
        z = ds['z'].values
        ds.close()

    z = np.squeeze(z)  # (n_levels, lat, lon) or (lat, lon) if single level

    features = {}
    n_levels = z.shape[0] if z.ndim == 3 else 1

    for i, lev in enumerate(LEVELS[:n_levels]):
        level_data = z[i] if z.ndim == 3 else z
        valid = level_data[~np.isnan(level_data)]
        if len(valid) < 10:
            continue

        mean_val = float(np.nanmean(valid))
        features[f'z{lev}_mean'] = mean_val

        # ρ = spatial std / |spatial mean| (roughness)
        rho = float(np.std(valid) / abs(np.mean(valid))) if np.mean(valid) != 0 else 0
        features[f'rho_{lev}'] = rho

    return features


def predict(features):
    """Make ENSO prediction using M1 and M4 models."""
    z850 = features.get('z850_mean')
    rho_700 = features.get('rho_700')

    if z850 is None:
        return None

    # M1: linear
    m1_raw = M1_SLOPE * z850 + M1_INTERCEPT

    # M4: nonlinear combination
    m4_raw = M4_COEFFS[0] * z850 + M4_COEFFS[1] * (rho_700 or 0) + M4_COEFFS[2]

    def classify(val):
        if val > 0.33:
            return "El_Nino"
        elif val < -0.33:
            return "La_Nina"
        return "Neutral"

    return {
        "M1": {"raw": round(m1_raw, 3), "class": classify(m1_raw)},
        "M4": {"raw": round(m4_raw, 3), "class": classify(m4_raw)},
    }


def main():
    now = datetime.now(timezone.utc)
    year = now.year
    month = now.month

    print(f"Africa Capacitor Monitor — {now.strftime('%Y-%m-%d')}")
    print(f"Target: Winter {year}-{year+1} ENSO state")

    # Only run during or after charging window (May-Jul)
    # If before May, use previous year's data
    if month < 5:
        charge_year = year - 1
        print(f"Before charging window, using {charge_year} data")
    else:
        charge_year = year

    # Download and compute features for each charging month
    all_features = {}
    for m in CHARGING_MONTHS:
        nc_file = download_era5(charge_year, m)
        if nc_file is None:
            print(f"Skipping month {m} (download failed)")
            continue
        feats = compute_features(nc_file)
        for k, v in feats.items():
            if k not in all_features:
                all_features[k] = []
            all_features[k].append(v)
        # Cleanup
        try:
            os.remove(nc_file)
        except:
            pass

    if not all_features:
        print("No features computed. Exiting.")
        # Write empty result
        result = {
            "timestamp": now.isoformat(),
            "status": "no_data",
            "message": "ERA5 download failed or no CDS key"
        }
        with open(OUTPUT_JSON, 'w') as f:
            json.dump(result, f, indent=2)
        return

    # Average features across charging months
    features = {k: float(np.mean(v)) for k, v in all_features.items()}
    n_months = len(list(all_features.values())[0])

    print(f"\nFeatures ({n_months} months):")
    for k, v in sorted(features.items()):
        print(f"  {k}: {v:.6f}")

    # Predict
    prediction = predict(features)
    if prediction is None:
        print("Prediction failed")
        return

    print(f"\nPrediction (Winter {charge_year}-{charge_year+1}):")
    print(f"  M1 (850hPa linear):     {prediction['M1']['class']} (raw: {prediction['M1']['raw']:+.3f})")
    print(f"  M4 (850hPa + ρ_std):    {prediction['M4']['class']} (raw: {prediction['M4']['raw']:+.3f})")

    agree = prediction['M1']['class'] == prediction['M4']['class']
    print(f"  Models agree: {'✅ YES' if agree else '⚠️ NO'}")

    # Build result
    result = {
        "timestamp": now.isoformat(),
        "charge_year": charge_year,
        "target_winter": f"{charge_year}-{charge_year+1}",
        "charging_months": n_months,
        "features": {k: round(v, 6) for k, v in features.items()},
        "prediction": prediction,
        "models_agree": agree,
        "historical_accuracy": HIST_ACCURACY,
        "status": "ok"
    }

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved {OUTPUT_JSON}")

    # Append to history
    history = []
    if os.path.exists(OUTPUT_HISTORY):
        try:
            with open(OUTPUT_HISTORY) as f:
                history = json.load(f)
        except:
            history = []

    # Avoid duplicate entries for same charge_year
    history = [h for h in history if h.get("charge_year") != charge_year]
    history.append(result)
    history.sort(key=lambda x: x.get("charge_year", 0))

    with open(OUTPUT_HISTORY, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Saved {OUTPUT_HISTORY} ({len(history)} entries)")


if __name__ == '__main__':
    main()
