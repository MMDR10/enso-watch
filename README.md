# 🌊 ENSO Watch — 聖嬰自動預警系統

多維度 ENSO 監測 + 預測系統，GitHub Actions 每月自動更新（9/15/22 號），GitHub Pages 靜態部署。

## 📊 四條監測線

| 監測線 | 數據來源 | 量度對象 | 需要 API Key |
|:--|:--|:--|:--|
| **ONI 顯示** | NOAA CPC | 官方 ENSO 相位指數 | ❌ 免 key |
| **OISST 鞍點結構** | NOAA OISST v2.1 | SST anomaly 空間鞍點比例 + 鞍點密度×ρ 相關 | ❌ 免 key |
| **curl τ 交界面鞍點** | ERA5 10m 風場 | 氣流-海面耦合帶鞍點密度 | ✅ CDS key |
| **Africa 電容預測** | ERA5 850/700 hPa | 6 個月提前 ENSO 預測（M1/M4 模型） | ✅ CDS key |

## 🔮 Africa 電容模型（2026-08 新增）

用 Africa-Indian Ocean 區域 850 hPa 位勢高度（5–7 月平均）提前 6 個月預測冬季 ENSO 狀態。

| 模型 | 特徵 | 整體正確率 | El Niño | La Niña | Neutral |
|:--|:--|:--|:--|:--|:--|
| M1 線性 | 850hPa mean | 71.8% | 83.3% | 76.9% | 57.1% |
| M4 非線性 | 850hPa + 700hPa ρ_std | 76.9% | 66.7% | 84.6% | **78.6%** |

訓練期：1982–2020（39 年 LOO）。樣本外驗證：dCor 0.674 > 訓練集 0.421。

📄 詳細報告：[中文版](ENSO_AFRICA_CAPACITOR_ZH.md) | [English](ENSO_AFRICA_CAPACITOR_EN.md)

## 🔄 探測器（原有）

| 探測器 | 量度對象 | 物理意義 |
|:--|:--|:--|
| **A · Multi-Scale Δ²H** | 雙窗口熱力剪切力 | 短波深層 vs 長波背景嘅跨尺度張力 |
| **B · χ_eff** | 結構剛性 | 系統對外界擾動嘅敏感度 |
| **C · Gaussianity** | 多源疊加完整性 | 驅動源平衡狀態（CLT） |
| **D · θ₁** | 幾何骨架角度 | 相空間幾何結構嘅緊緻度 |

## 🛠️ 手動運行

```bash
pip install numpy scipy netCDF4 xarray cdsapi
python enso_auto_update.py           # ONI + 探測器
python enso_saddle_monitor.py        # OISST 鞍點（免 key）
python enso_curl_monitor.py          # curl τ 鞍點（需 CDS key）
python enso_africa_capacitor.py      # Africa 電容預測（需 CDS key）
```

## 📁 檔案結構

```
├── enso_auto_update.py           ← ONI + 四探測器
├── enso_saddle_monitor.py        ← OISST anomaly 鞍點結構
├── enso_curl_monitor.py          ← ERA5 curl τ 交界面鞍點
├── enso_africa_capacitor.py      ← Africa 電容 6 月預測（NEW）
├── enso_dashboard.html           ← 靜態儀表板
├── dashboard_data.json           ← ONI + 探測器數據
├── saddle_data.json / history    ← 鞍點結構數據
├── curl_data.json / history      ← curl τ 數據
├── africa_capacitor_data.json    ← Africa 電容預測（NEW）
├── ENSO_AFRICA_CAPACITOR_ZH.md   ← 研究報告（中文）
├── ENSO_AFRICA_CAPACITOR_EN.md   ← 研究報告（English）
├── ENSO_PUBLIC_NOTE.md / _EN.md  ← ENSO 簡易預測筆記
├── .github/workflows/
│   └── enso_update.yml           ← GitHub Actions cron job
└── README.md
```

## ⚠️ 免責聲明

呢個係研究工具，唔係官方預報。請參考當地氣象局嘅官方 ENSO 預報。

數據來源：NOAA CPC (ONI)、NOAA NCEI (OISST)、Copernicus CDS (ERA5)

## 📜 License

CC BY 4.0
