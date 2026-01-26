# 2383 深度技術分析與量化回測 / Deep Technical Analysis & Quant Backtest

## 資料與環境 / Data & Environment
- 資料區間 / Data range: 2022-01-01 至 2026-01-26
- 樣本數 / Samples: 987
- 環境 / Environment: AI_trader conda
- 計算方式 / Compute: multi-threading for backtests

## 策略回測結果 / Strategy Backtest Results
- CrossSMAStrategy (Fast=10, Slow=30)
  - Sharpe: 0.7751166799160945
  - Total Return (%): 113.6762
  - Max Drawdown (%): 36.1642
  - Trades: 17
  - Win Rate (%): 35.294117647058826
- RSIStrategy (Oversold=30, Overbought=70)
  - Sharpe: 0.5168700253216902
  - Total Return (%): 50.8512
  - Max Drawdown (%): 44.4567
  - Trades: 3
  - Win Rate (%): 66.66666666666666

## 技術診斷（Medical Journal Style）/ Technical Diagnosis
以 20 日布林通道與 14 日 ATR 指標進行量化評估，並以審慎區間界定進出場風險。
Using 20-day Bollinger Bands and 14-day ATR, we define conservative zones for entry/exit risk.

### 今日指標 / Today’s Indicators
- Date: 2026-01-26
- Close: 1740.0000
- BB Mid: 1631.2500
- BB Upper: 1758.9781
- BB Lower: 1503.5219
- ATR(14): 90.0000

### 合理價位估算 / Valuation Levels
以 1503.5219 與 1758.9781 為基準，
並以 ATR(14) 90.0000 作為波動補正。
Based on BB lower/upper and ATR(14), the following levels are computed.

- 建議買入點 / Suggested Buy: 1548.5219
- 獲利了結點 / Take Profit: 1713.9781
- 風險停損點 / Stop Loss: 1413.5219

## 圖表 / Chart
![2383 chart](docs/2383_analysis.png)
