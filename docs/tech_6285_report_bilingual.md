# 啟基(6285) 近一年技術分析與回測報告 / WNC (6285) 1Y Technical Analysis & Backtest Report

## 資料與期間 / Data & Period
- 資料檔案: `data/tw_stock/6285_2025-02-02_to_2026-02-02.csv`
- 資料區間: 2025-02-03 ～ 2026-01-30（共 249 筆）
- Data source file: `data/tw_stock/6285_2025-02-02_to_2026-02-02.csv`
- Date range: 2025-02-03 to 2026-01-30 (249 rows)

## 價格極值 / Price Extremes
- 最高價(High): 192.00（2026-01-30）
- 最低價(Low): 96.00（2025-11-19）
- 最高收盤(Close): 185.00（2026-01-30）
- 最低收盤(Close): 96.00（2025-11-19）
- Highest High: 192.00 (2026-01-30)
- Lowest Low: 96.00 (2025-11-19)
- Highest Close: 185.00 (2026-01-30)
- Lowest Close: 96.00 (2025-11-19)

## 最新指標（最後一筆交易日）/ Latest Indicators (Last Trading Day)
- 日期 / Date: 2026-01-30
- 收盤 / Close: 185.00
- SMA5/20/60: 169.20 / 129.47 / 110.95
- EMA12/26: 150.87 / 132.45
- RSI14: 90.02
- MACD/Signal/Hist: 18.414 / 11.781 / 6.633
- BB(20,2) Mid/Upper/Lower: 129.47 / 183.17 / 75.78
- Stoch %K/%D: 91.81 / 91.33
- ATR14: 9.00
- ROC12: 72.90%

## 經典策略回測比較 / Classic Strategy Backtest Comparison
資料檔案 / Data file: `data/tw_stock/6285_2025-02-02_to_2026-02-02.csv`

| Strategy | Status | Return % | Sharpe | Max Drawdown % | Trades | Win Rate % |
|---|---|---|---|---|---|---|
| BBandsStrategy | ok | 28.69 | 17.73 | 13.46 | 4 | 100.00 |
| BuyHoldStrategy | ok | 27.43 | 0.45 | 36.46 | 1 | 0.00 |
| CrossSMAStrategy | ok | 21.99 | 0.41 | 31.51 | 5 | 0.00 |
| DoubleTopStrategy | ok | 0.00 |  | 0.00 | 0 |  |
| MACDStrategy | ok | -10.70 | -1.19 | 12.63 | 2 | 0.00 |
| MomentumStrategy | ok | 21.21 | 0.40 | 36.25 | 33 | 33.33 |
| NaiveROCStrategy | ok | 22.94 | 0.42 | 30.05 | 3 | 0.00 |
| NaiveSMAStrategy | ok | -0.52 | 0.21 | 44.15 | 19 | 10.53 |
| RiskAverseStrategy | ok | 16.50 | 0.88 | 4.16 | 1 | 100.00 |
| ROCMAStrategy | ok | 23.67 | 0.42 | 30.56 | 5 | 0.00 |
| ROCStochStrategy | ok | 55.31 | 0.67 | 24.26 | 1 | 0.00 |
| RsiBollingerBandsStrategy | ok | 16.44 | 1.49 | 11.71 | 2 | 100.00 |
| RSIStrategy | ok | 16.07 | 0.43 | 23.09 | 1 | 100.00 |
| RSRSStrategy | ok | 23.69 | 0.42 | 37.00 | 1 | 0.00 |
| TripleRsiStrategy | ok | 17.02 | 0.88 | 0.00 | 1 | 0.00 |
| TurtleTradingStrategy | ok | 9.80 | 0.28 | 19.04 | 5 | 0.00 |
| VCPStrategy | ok | 0.00 |  | 0.00 | 0 |  |
| AlphaRSIProStrategy | ok | 0.00 |  | 0.00 | 0 |  |
| AdaptiveRSIStrategy | ok | 17.02 | 0.38 | 37.52 | 1 | 0.00 |
| HybridAlphaRSIStrategy | ok | 0.00 |  | 0.00 | 0 |  |

## 策略參數最佳化（格點搜尋）/ Strategy Optimization (Grid Search)
資料檔案 / Data file: `data/tw_stock/6285_2025-02-02_to_2026-02-02.csv`

### BBandsStrategy
- 最佳參數 / Best params: `{'period': 25, 'devfactor': 2.5}`
- 報酬 / Return: 28.79%
- Sharpe: 16.25
- 最大回撤 / Max Drawdown: 11.71%
- 交易數 / Trades: 3
- 勝率 / Win Rate: 100.00%

### CrossSMAStrategy
- 最佳參數 / Best params: `{'fast': 3, 'slow': 40}`
- 報酬 / Return: 35.70%
- Sharpe: 0.56
- 最大回撤 / Max Drawdown: 23.81%
- 交易數 / Trades: 5
- 勝率 / Win Rate: 0.00%

### MACDStrategy
- 最佳參數 / Best params: `{'fastperiod': 15, 'slowperiod': 30, 'signalperiod': 12}`
- 報酬 / Return: 0.11%
- Sharpe: -16.58
- 最大回撤 / Max Drawdown: 5.89%
- 交易數 / Trades: 1
- 勝率 / Win Rate: 100.00%

### RSIStrategy
- 最佳參數 / Best params: `{'rsi_period': 14, 'oversold': 25, 'overbought': 65}`
- 報酬 / Return: 51.96%
- Sharpe: 18.16
- 最大回撤 / Max Drawdown: 11.88%
- 交易數 / Trades: 2
- 勝率 / Win Rate: 100.00%

### RsiBollingerBandsStrategy
- 最佳參數 / Best params: `{'rsi_period': 14, 'bb_period': 20, 'bb_dev': 1.5, 'oversold': 25, 'overbought': 70}`
- 報酬 / Return: 40.67%
- Sharpe: 3.04
- 最大回撤 / Max Drawdown: 9.54%
- 交易數 / Trades: 2
- 勝率 / Win Rate: 100.00%

## 產出檔案 / Output Files
- 技術指標表 / Indicator table: `docs/tech_6285_indicators.csv`
- 技術圖表 / Technical charts: `docs/tech_6285_charts.png`
- 回測對比圖 / Backtest scatter: `docs/tech_6285_backtest_scatter.png`
- 回測結果表 / Backtest table: `docs/tech_6285_backtests.csv`, `docs/tech_6285_backtests.md`
- 參數最佳化 / Optimization: `docs/tech_6285_optimizations.csv`, `docs/tech_6285_optimizations.md`
