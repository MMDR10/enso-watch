# 🌊 ENSO Watch — 聖嬰自動預警儀表板

多維度 ENSO 預警系統，GitHub Actions 自動更新，GitHub Pages 靜態部署。

## 🚀 快速部署（3 步）

### 1. Fork / 建立 Repo

將呢個 `release/` 目錄嘅內容 push 到一個 GitHub repo（例如 `MMDR10/enso-watch`）。

### 2. 啟用 GitHub Pages

Settings → Pages → Source: **Deploy from a branch** → Branch: `main` → Save。

等 1-2 分鐘，Dashboard 會喺 `https://<user>.github.io/enso-watch/enso_dashboard.html` 上線。

### 3. 啟用 GitHub Actions

Actions tab → 啟用 workflows → 搵 "ENSO Watch Auto-Update" → Enable。

之後每 9 號、15 號、22 號（NOAA 更新 ONI 前後）自動 fetch 最新數據、重新計算、push 更新。

## 📊 四個探測器

| 探測器 | 量度對象 | 物理意義 |
|:--|:--|:--|
| **A · Multi-Scale Δ²H** | 雙窗口熱力剪切力 | 短波深層 vs 長波背景嘅跨尺度張力 |
| **B · χ_eff** | 結構剛性 | 系統對外界擾動嘅敏感度 |
| **C · Gaussianity** | 多源疊加完整性 | 驅動源平衡狀態（CLT） |
| **D · θ₁** | 幾何骨架角度 | 相空間幾何結構嘅緊緻度 |

## 🔄 三級相位狀態機（Detector A）

| 狀態 | 條件 | 警報 |
|:--|:--|:--|
| Deep Accumulation | ΔH_long > 0 且 ΔH_short > 0 | 🟢 CLEAR |
| **Shear Inversion** | ΔH_long > 0 但 ΔH_short < 0 | 🟡 WARNING |
| Full Regime Shift | ΔH_long < 0 且 ΔH_short < 0 | 🔴 CRITICAL |

## 🛠️ 手動運行

```bash
pip install numpy scipy
python enso_auto_update.py
```

會 fetch NOAA 最新 ONI → 計算全部四個探測器 → 輸出 `dashboard_data.json`。

## 📁 檔案結構

```
release/
├── enso_auto_update.py      ← GitHub Actions 自動執行嘅主腳本
├── enso_dashboard.html       ← 靜態儀表板（讀 dashboard_data.json）
├── dashboard_data.json       ← 自動生成嘅最新數據
├── .github/workflows/
│   └── enso_update.yml       ← GitHub Actions cron job
└── README.md
```

## ⚠️ 免責聲明

呢個係研究工具，唔係官方預報。請參考當地氣象局嘅官方 ENSO 預報。

數據來源：NOAA Climate Prediction Center (ONI index)

## 📜 License

MIT
