# 全自動化深度技術分析與量化回測報告 / Automated Deep Technical Analysis & Backtest

- 資料期間 Data Window: 2022-01-01 至 2026-01-26
- 執行環境 Execution Environment: (AI_trader) conda
- 多執行緒 Multi-threading: 啟用（平行回測與指標計算） / Enabled for parallel backtests and indicators

## 方法 Methods (Academic Neutral Tone / 學術中性語氣)
- 策略掃描以 `ai_trader.backtesting.strategies.classic` 內建策略為基準。  Strategy scan is based on built-in classic strategies for single stocks.
- CrossSMAStrategy 參數調整為 Fast=10, Slow=30；RSI 策略採 RsiBollingerBandsStrategy，Oversold=30, Overbought=70。  CrossSMAStrategy uses Fast=10/Slow=30; RSI uses RsiBollingerBandsStrategy with 30/70 levels.
- 合理價估算結合布林通道、ATR 與移動平均：  Price estimation combines Bollinger Bands, ATR, and moving averages.
  Buy = (Lower Band + 0.25×ATR + MA Anchor)/2；    Take Profit = (Upper Band − 0.25×ATR + MA Anchor)/2；    Stop Loss = min(Lower Band − 0.75×ATR, MA Anchor − 1.0×ATR)。
  MA Anchor = (SMA20 + SMA50)/2。  MA Anchor uses SMA20 and SMA50.

## 標的 2313 / Ticker 2313
- 資料來源 Data Source: `data/tw_stock/2313_2022-01-01_to_2026-01-26.csv`
- 截至日期 As-of Date: 2026-01-26
- 收盤價 Close: 177.00
- 圖表 Chart: `docs/figures/2313_technical.png`

### 技術指標 Technical Indicators
- SMA10/30/50/200: 149.00 / 115.36 / 101.95 / 76.62
- SMA10 vs SMA30: bullish (短中期 Short/Mid Trend)
- SMA50 vs SMA200: bullish (中長期 Mid/Long Trend)
- RSI14: 84.98
- MACD/Signal/Hist: 18.96 / 14.42 / 4.54
- Bollinger(20): upper 179.22, mid 126.41, lower 73.59
- ATR14: 10.46
- 20D Avg Volume: 85401993
- 52W High/Low: 177.00 / 44.15, 距高點 From High: 0.00% , 距低點 From Low: 300.91%
- 報酬 Return 1M(21d): 78.79%
- 報酬 Return 3M(63d): 100.45%
- 報酬 Return 6M(126d): 183.20%
- 報酬 Return 1Y(252d): 164.57%

### 策略回測 Backtest Summary
| Strategy | Final | Return% | Sharpe | MaxDD% | Trades | WinRate% | Note |
|---|---:|---:|---:|---:|---:|---:|---|
| BuyHoldStrategy | 3887687.35 | 288.77 | 1.02 | 49.87 | 1 | NA | OK |
| RSRSStrategy | 3861588.49 | 286.16 | 1.00 | 50.08 | 1 | NA | OK |
| ROCStochStrategy | 3524483.55 | 252.45 | 0.92 | 49.50 | 1 | NA | OK |
| ROCMAStrategy | 3143069.89 | 214.31 | 0.86 | 22.70 | 18 | 55.56 | OK |
| AlphaRSIProStrategy | 2611019.15 | 161.10 | 0.72 | 36.58 | 1 | NA | OK |
| NaiveSMAStrategy | 2166464.28 | 116.65 | 0.56 | 37.64 | 67 | 25.37 | OK |
| BBandsStrategy | 2047948.07 | 104.79 | 1.70 | 35.26 | 11 | 90.91 | OK |
| MomentumStrategy | 1999271.62 | 99.93 | 0.51 | 35.40 | 62 | 27.42 | OK |
| CrossSMAStrategy | 1976258.78 | 97.63 | 0.51 | 29.85 | 19 | 42.11 | OK |
| VCPStrategy | 1795779.83 | 79.58 | 0.45 | 27.34 | 4 | 25.00 | OK |
| MACDStrategy | 1730176.19 | 73.02 | 0.48 | 22.90 | 16 | 18.75 | OK |
| TripleRsiStrategy | 1677363.76 | 67.74 | 0.45 | 35.82 | 5 | 20.00 | OK |
| RsiBollingerBandsStrategy | 1476257.87 | 47.63 | 0.74 | 26.08 | 4 | 100.00 | OK |
| TurtleTradingStrategy | 1368606.23 | 36.86 | 0.29 | 34.88 | 19 | 15.79 | OK |
| RiskAverseStrategy | 1243490.16 | 24.35 | 0.28 | 28.43 | 8 | 62.50 | OK |
| AdaptiveRSIStrategy | 1041627.08 | 4.16 | -0.01 | 35.65 | 3 | 66.67 | OK |
| NaiveROCStrategy | 1024368.00 | 2.44 | 0.15 | 55.46 | 15 | 26.67 | OK |
| DoubleTopStrategy | 948167.43 | -5.18 | -0.98 | 6.23 | 1 | 0.00 | OK |
| HybridAlphaRSIStrategy | 939246.48 | -6.08 | -0.38 | 36.58 | 1 | 0.00 | OK |

### 合理價位診斷 Clinical Financial Diagnosis
- 建議買入點 Suggested Buy: 95.19  (BB/ATR/MA blend)
- 獲利了結點 Suggested Take Profit: 145.39  (BB/ATR/MA blend)
- 風險停損位 Suggested Stop Loss: 65.74  (BB/ATR/MA blend)

### 視覺化 Visualization
![](docs/figures/2313_technical.png)

## 標的 6282 / Ticker 6282
- 資料來源 Data Source: `data/tw_stock/6282_2022-01-01_to_2026-01-26.csv`
- 截至日期 As-of Date: 2026-01-26
- 收盤價 Close: 58.80
- 圖表 Chart: `docs/figures/6282_technical.png`

### 技術指標 Technical Indicators
- SMA10/30/50/200: 57.81 / 47.62 / 44.13 / 33.12
- SMA10 vs SMA30: bullish (短中期 Short/Mid Trend)
- SMA50 vs SMA200: bullish (中長期 Mid/Long Trend)
- RSI14: 68.20
- MACD/Signal/Hist: 5.05 / 4.23 / 0.82
- Bollinger(20): upper 66.39, mid 50.66, lower 34.94
- ATR14: 4.42
- 20D Avg Volume: 108333944
- 52W High/Low: 63.80 / 22.15, 距高點 From High: -7.84% , 距低點 From Low: 165.46%
- 報酬 Return 1M(21d): 43.77%
- 報酬 Return 3M(63d): 33.64%
- 報酬 Return 6M(126d): 116.18%
- 報酬 Return 1Y(252d): 100.34%

### 策略回測 Backtest Summary
| Strategy | Final | Return% | Sharpe | MaxDD% | Trades | WinRate% | Note |
|---|---:|---:|---:|---:|---:|---:|---|
| RSRSStrategy | 1847903.95 | 84.79 | 0.60 | 53.18 | 2 | 50.00 | OK |
| TripleRsiStrategy | 1619587.57 | 61.96 | 0.70 | 19.89 | 3 | 66.67 | OK |
| RiskAverseStrategy | 1616331.88 | 61.63 | 0.74 | 13.35 | 3 | 100.00 | OK |
| BuyHoldStrategy | 1594532.03 | 59.45 | 0.44 | 58.90 | 1 | NA | OK |
| RsiBollingerBandsStrategy | 1355865.45 | 35.59 | 0.73 | 12.96 | 3 | 100.00 | OK |
| MACDStrategy | 1313634.06 | 31.36 | 0.34 | 20.18 | 15 | 20.00 | OK |
| VCPStrategy | 1184968.17 | 18.50 | 0.36 | 14.26 | 1 | 100.00 | OK |
| BBandsStrategy | 1125711.55 | 12.57 | 0.40 | 30.31 | 10 | 60.00 | OK |
| ROCStochStrategy | 1125188.04 | 12.52 | 0.18 | 57.45 | 1 | NA | OK |
| NaiveROCStrategy | 1070034.74 | 7.00 | 0.13 | 57.49 | 11 | 27.27 | OK |
| CrossSMAStrategy | 985515.84 | -1.45 | 0.04 | 51.14 | 19 | 36.84 | OK |
| AdaptiveRSIStrategy | 960910.20 | -3.91 | -0.08 | 35.48 | 2 | 50.00 | OK |
| DoubleTopStrategy | 925744.24 | -7.43 | -0.84 | 7.43 | 2 | 0.00 | OK |
| ROCMAStrategy | 920414.29 | -7.96 | -0.31 | 26.80 | 18 | 38.89 | OK |
| TurtleTradingStrategy | 707611.17 | -29.24 | -0.61 | 48.22 | 16 | 18.75 | OK |
| MomentumStrategy | 701988.13 | -29.80 | -0.23 | 66.93 | 86 | 23.26 | OK |
| NaiveSMAStrategy | 583823.96 | -41.62 | -0.24 | 67.52 | 84 | 23.81 | OK |
| AlphaRSIProStrategy | 1000000.00 | 0.00 | NA | 0.00 | NA | NA | OK |
| HybridAlphaRSIStrategy | 1000000.00 | 0.00 | NA | 0.00 | NA | NA | OK |

### 合理價位診斷 Clinical Financial Diagnosis
- 建議買入點 Suggested Buy: 41.72  (BB/ATR/MA blend)
- 獲利了結點 Suggested Take Profit: 56.34  (BB/ATR/MA blend)
- 風險停損位 Suggested Stop Loss: 31.63  (BB/ATR/MA blend)

### 視覺化 Visualization
![](docs/figures/6282_technical.png)

## 標的 6188 / Ticker 6188
- 資料缺失，無法分析。 / Data unavailable; analysis skipped.

## 標的 6285 / Ticker 6285
- 資料來源 Data Source: `data/tw_stock/6285_2022-01-01_to_2026-01-26.csv`
- 截至日期 As-of Date: 2026-01-26
- 收盤價 Close: 157.50
- 圖表 Chart: `docs/figures/6285_technical.png`

### 技術指標 Technical Indicators
- SMA10/30/50/200: 126.10 / 110.46 / 106.50 / 116.92
- SMA10 vs SMA30: bullish (短中期 Short/Mid Trend)
- SMA50 vs SMA200: bearish (中長期 Mid/Long Trend)
- RSI14: 84.57
- MACD/Signal/Hist: 11.13 / 5.64 / 5.49
- Bollinger(20): upper 150.94, mid 114.97, lower 79.00
- ATR14: 7.54
- 20D Avg Volume: 17772198
- 52W High/Low: 157.50 / 96.00, 距高點 From High: 0.00% , 距低點 From Low: 64.06%
- 報酬 Return 1M(21d): 54.41%
- 報酬 Return 3M(63d): 29.10%
- 報酬 Return 6M(126d): 34.62%
- 報酬 Return 1Y(252d): 28.05%

### 策略回測 Backtest Summary
| Strategy | Final | Return% | Sharpe | MaxDD% | Trades | WinRate% | Note |
|---|---:|---:|---:|---:|---:|---:|---|
| BBandsStrategy | 2196578.29 | 119.66 | 1.83 | 25.26 | 13 | 92.31 | OK |
| ROCStochStrategy | 2049649.84 | 104.96 | 0.49 | 39.26 | 2 | 50.00 | OK |
| RSRSStrategy | 2021660.32 | 102.17 | 0.49 | 44.71 | 2 | 50.00 | OK |
| BuyHoldStrategy | 1989357.02 | 98.94 | 0.48 | 42.59 | 1 | NA | OK |
| AdaptiveRSIStrategy | 1487650.07 | 48.77 | 0.39 | 26.22 | 3 | 66.67 | OK |
| RsiBollingerBandsStrategy | 1272035.83 | 27.20 | 0.74 | 14.72 | 5 | 80.00 | OK |
| NaiveROCStrategy | 1226195.96 | 22.62 | 0.25 | 48.30 | 11 | 36.36 | OK |
| CrossSMAStrategy | 1148827.82 | 14.88 | 0.24 | 52.92 | 20 | 25.00 | OK |
| ROCMAStrategy | 1133566.42 | 13.36 | 0.17 | 42.16 | 16 | 18.75 | OK |
| TurtleTradingStrategy | 1060620.38 | 6.06 | 0.10 | 44.67 | 17 | 23.53 | OK |
| DoubleTopStrategy | 1001279.65 | 0.13 | -19.04 | 1.70 | 1 | 100.00 | OK |
| VCPStrategy | 943156.93 | -5.68 | -0.94 | 9.59 | 1 | 0.00 | OK |
| RiskAverseStrategy | 853378.29 | -14.66 | -0.03 | 50.34 | 6 | 50.00 | OK |
| NaiveSMAStrategy | 844208.39 | -15.58 | 0.01 | 55.91 | 69 | 26.09 | OK |
| TripleRsiStrategy | 808927.95 | -19.11 | -0.51 | 36.02 | 5 | 20.00 | OK |
| MACDStrategy | 724266.25 | -27.57 | -0.91 | 36.77 | 18 | 27.78 | OK |
| MomentumStrategy | 653752.11 | -34.62 | -0.09 | 65.73 | 91 | 26.37 | OK |
| AlphaRSIProStrategy | 1000000.00 | 0.00 | NA | 0.00 | NA | NA | OK |
| HybridAlphaRSIStrategy | 1000000.00 | 0.00 | NA | 0.00 | NA | NA | OK |

### 合理價位診斷 Clinical Financial Diagnosis
- 建議買入點 Suggested Buy: 95.81  (BB/ATR/MA blend)
- 獲利了結點 Suggested Take Profit: 129.90  (BB/ATR/MA blend)
- 風險停損位 Suggested Stop Loss: 73.35  (BB/ATR/MA blend)

### 視覺化 Visualization
![](docs/figures/6285_technical.png)

---
本報告為量化指標推估，用於風險管理與情境比較，不構成投資建議。  
This report provides quantitative estimates for risk management and scenario comparison; not investment advice.