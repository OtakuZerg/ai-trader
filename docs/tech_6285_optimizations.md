# 6285 Strategy Optimization (Grid Search)

資料檔案: `data/tw_stock/6285_2025-02-02_to_2026-02-02.csv`

## BBandsStrategy

- Best params: `{'period': 25, 'devfactor': 2.5}`
- Return: 28.79%
- Sharpe: 16.25
- Max Drawdown: 11.71%
- Trades: 3
- Win Rate: 100.00%

## CrossSMAStrategy

- Best params: `{'fast': 3, 'slow': 40}`
- Return: 35.70%
- Sharpe: 0.56
- Max Drawdown: 23.81%
- Trades: 5
- Win Rate: 0.00%

## MACDStrategy

- Best params: `{'fastperiod': 15, 'slowperiod': 30, 'signalperiod': 12}`
- Return: 0.11%
- Sharpe: -16.58
- Max Drawdown: 5.89%
- Trades: 1
- Win Rate: 100.00%

## RSIStrategy

- Best params: `{'rsi_period': 14, 'oversold': 25, 'overbought': 65}`
- Return: 51.96%
- Sharpe: 18.16
- Max Drawdown: 11.88%
- Trades: 2
- Win Rate: 100.00%

## RsiBollingerBandsStrategy

- Best params: `{'rsi_period': 14, 'bb_period': 20, 'bb_dev': 1.5, 'oversold': 25, 'overbought': 70}`
- Return: 40.67%
- Sharpe: 3.04
- Max Drawdown: 9.54%
- Trades: 2
- Win Rate: 100.00%

