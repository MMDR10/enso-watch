# 數據來源與說明 / Data Sources and Documentation

## 1. ERA5 再分析數據（Copernicus Climate Data Store）

| 項目 | 說明 |
|------|------|
| **來源** | ECMWF Copernicus Climate Data Store (CDS) |
| **產品** | `reanalysis-era5-pressure-levels-monthly-means` |
| **變量** | Geopotential (`z`) |
| **壓力層** | 850, 700, 600, 500, 400, 300, 250, 200 hPa |
| **空間範圍** | Africa–Indian Ocean: 20°N–20°S, 30°E–120°E |
| **時間範圍** | 1982–2026（月均值） |
| **下載方式** | CDS API (`cdsapi` Python package) |
| **API 設定** | 需要 `CDS_URL` + `CDS_KEY`（GitHub Secrets） |
| **授權** | Copernicus License（免費註冊） |
| **用途** | Africa 電容模型：850/700 hPa 位勢高度 → ENSO 預測 |

**下載腳本：** `enso_africa_capacitor.py`（自動下載 5–7 月月均值）

**ERA5 引用：**
> Hersbach, H., et al. (2020). The ERA5 global reanalysis. Quarterly Journal of the Royal Meteorological Society, 146(730), 1999–2049. https://doi.org/10.1002/qj.3803

---

## 2. OOMI — Extended OLR MJO Index（NOAA PSL）

| 項目 | 說明 |
|------|------|
| **來源** | NOAA Physical Sciences Laboratory (PSL) |
| **URL** | https://psl.noaa.gov/mjo/mjoindex/ |
| **檔案** | `oomi.1x.txt`（日均值） |
| **格式** | 純文字：year month day phase PC1 PC2 amplitude |
| **時間範圍** | 1979–2022（全日圈） |
| **物理量** | OLR-based MJO index（射出長波輻射） |
| **下載方式** | HTTP GET（免 key） |
| **用途** | MJO 日尺度分析：El Niño 前 MJO 增強（p<0.0001） |

**OOMI 引用：**
> Kiladis, G. N., et al. (2014). A Comparison of OLR and Circulation-Based Indices for Tracking the MJO. Monthly Weather Review, 142(5), 1697–1715. https://doi.org/10.1175/MWR-D-13-00301.1

---

## 3. RMM* — Filtered Real-time MJO Index（NOAA PSL）

| 項目 | 說明 |
|------|------|
| **來源** | NOAA Physical Sciences Laboratory (PSL) |
| **URL** | https://psl.noaa.gov/mjo/mjoindex/ |
| **檔案** | `rmm_star_data.txt`（日均值） |
| **格式** | 純文字：year month day phase RMM1 RMM2 amplitude |
| **時間範圍** | 1979–2021（僅 Nov–Apr） |
| **特殊處理** | LIM filter 去除 ENSO 低頻變異 |
| **用途** | MJO 日尺度分析 V1（已被 OOMI V2 取代） |

**注意：** RMM* 使用 LIM filter 去除 ENSO 變異，可能扭曲 MJO-ENSO 關係。本研究 V2 改用 OOMI（全日圈、無 ENSO filter），結果物理上更合理。

---

## 4. ONI — Oceanic Niño Index（NOAA CPC）

| 項目 | 說明 |
|------|------|
| **來源** | NOAA Climate Prediction Center (CPC) |
| **URL** | https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt |
| **格式** | 純文字：SEAS YR TOTAL ANOM |
| **時間範圍** | 1950–present |
| **物理量** | Niño 3.4 區域 SST anomaly（3 個月滑動平均） |
| **下載方式** | HTTP GET（免 key） |
| **用途** | ENSO 相位分類（El Niño / La Niña / Neutral） |

**ONI 引用：**
> Climate Prediction Center. (2026). Cold & Warm Episodes by Season. NOAA/NWS/NCEP. https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php

---

## 5. ENSO 事件分類

基於 ONI 指數，使用以下標準分類：

| 相位 | 條件 | 事件年份 |
|------|------|---------|
| **El Niño** | ONI ≥ +0.5°C 持續 5 個月 | 1982, 1983, 1986, 1987, 1991, 1994, 1997, 2002, 2004, 2006, 2009, 2015 |
| **La Niña** | ONI ≤ −0.5°C 持續 5 個月 | 1984, 1985, 1988, 1995, 1998, 1999, 2000, 2007, 2010, 2011, 2016, 2017, 2020 |
| **Neutral** | −0.5°C < ONI < +0.5°C | 其餘年份 |

---

## 6. 模型訓練與驗證

| 項目 | 說明 |
|------|------|
| **訓練期** | 1982–2020（39 年） |
| **驗證方法** | Leave-One-Out Cross-Validation (LOO) |
| **樣本外驗證** | 2021–2025（dCor 0.674 > 訓練集 0.421） |
| **時間範疇** | 現代衛星時代氣候體制（Post-1976/77 PDO Regime） |

---

## 7. 引用本研究

如使用本研究代碼或數據，請引用：

```
DR. (2026). Africa Capacitor Model: Predicting ENSO 6 Months Ahead with 
850 hPa Geopotential Height. GitHub: MMDR10/enso-watch. 
CC BY 4.0.
```

---

## 8. 授權

- 本研究代碼：MIT License
- 研究文章：CC BY 4.0
- 原始數據：遵循各來源授權（ERA5: Copernicus License; NOAA: Public Domain）
