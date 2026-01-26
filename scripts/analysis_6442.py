#!/usr/bin/env python
"""Technical analysis and backtest report for TW stock 6442.
技術分析與回測報告（台股 6442）。
"""

from __future__ import annotations

import glob
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ai_trader.backtesting.strategies.classic.rsi import RsiBollingerBandsStrategy
from ai_trader.backtesting.strategies.classic.sma import CrossSMAStrategy
from ai_trader.utils.backtest import run_backtest


# === 資料與路徑設定 / Data and path setup ===
DATA_GLOB = "data/tw_stock/6442_*.csv"
CASH = 1_000_000
COMMISSION = 0.001425
SIZER_PARAMS = {"percents": 95}
ANALYZERS = ["sharpe", "drawdown", "returns", "trades"]


@dataclass
class BacktestSummary:
    """回測摘要 / Backtest summary."""

    strategy: str
    final_value: float
    total_return_pct: float
    sharpe: Optional[float]
    max_drawdown_pct: Optional[float]
    trades: Optional[int]
    win_rate_pct: Optional[float]


def _latest_data_file() -> Path:
    """挑選最新的 6442 資料檔 / Pick the newest 6442 data file."""
    candidates = [Path(p) for p in glob.glob(DATA_GLOB)]
    if not candidates:
        raise FileNotFoundError(f"No data files found with pattern: {DATA_GLOB}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_price_data(path: Path) -> pd.DataFrame:
    """讀取與整理資料 / Load and clean price data."""
    df = pd.read_csv(path, parse_dates=["date"])
    # 依日期排序並去重 / Sort by date and drop duplicates
    df = df.sort_values("date").drop_duplicates("date")
    return df.set_index("date")


def _calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """計算 RSI / Compute RSI."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _indicator_report(df: pd.DataFrame) -> Dict[str, Any]:
    """計算技術指標 / Compute technical indicators."""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    sma10 = close.rolling(10).mean()
    sma20 = close.rolling(20).mean()
    sma30 = close.rolling(30).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()

    rsi14 = _calc_rsi(close, 14)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal

    bb_std = close.rolling(20).std()
    bb_upper = sma20 + 2 * bb_std
    bb_lower = sma20 - 2 * bb_std

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr14 = tr.rolling(14).mean()

    last_date = close.index[-1]
    last_close = close.iloc[-1]

    # 報酬區間 / Return windows
    windows = {"1M(21d)": 21, "3M(63d)": 63, "6M(126d)": 126, "1Y(252d)": 252}
    returns = {}
    for label, n in windows.items():
        if len(close) > n:
            returns[label] = (last_close / close.iloc[-n - 1] - 1) * 100
        else:
            returns[label] = None

    # 52 週高低 / 52-week high/low
    lookback = 252 if len(close) >= 252 else len(close)
    window = close.tail(lookback)
    hi_52 = window.max()
    lo_52 = window.min()
    pct_from_hi = (last_close / hi_52 - 1) * 100
    pct_from_lo = (last_close / lo_52 - 1) * 100

    # 趨勢判斷 / Trend state
    sma_signal = "insufficient_data"
    if pd.notna(sma10.iloc[-1]) and pd.notna(sma30.iloc[-1]):
        if sma10.iloc[-1] > sma30.iloc[-1]:
            sma_signal = "bullish"
        elif sma10.iloc[-1] < sma30.iloc[-1]:
            sma_signal = "bearish"
        else:
            sma_signal = "neutral"

    trend_50_200 = None
    if pd.notna(sma50.iloc[-1]) and pd.notna(sma200.iloc[-1]):
        trend_50_200 = "bullish" if sma50.iloc[-1] > sma200.iloc[-1] else "bearish"

    return {
        "asof": last_date,
        "last_close": float(last_close),
        "sma10": float(sma10.iloc[-1]) if pd.notna(sma10.iloc[-1]) else None,
        "sma20": float(sma20.iloc[-1]) if pd.notna(sma20.iloc[-1]) else None,
        "sma30": float(sma30.iloc[-1]) if pd.notna(sma30.iloc[-1]) else None,
        "sma50": float(sma50.iloc[-1]) if pd.notna(sma50.iloc[-1]) else None,
        "sma200": float(sma200.iloc[-1]) if pd.notna(sma200.iloc[-1]) else None,
        "rsi14": float(rsi14.iloc[-1]) if pd.notna(rsi14.iloc[-1]) else None,
        "macd": float(macd.iloc[-1]),
        "macd_signal": float(macd_signal.iloc[-1]),
        "macd_hist": float(macd_hist.iloc[-1]),
        "bb_upper": float(bb_upper.iloc[-1]) if pd.notna(bb_upper.iloc[-1]) else None,
        "bb_mid": float(sma20.iloc[-1]) if pd.notna(sma20.iloc[-1]) else None,
        "bb_lower": float(bb_lower.iloc[-1]) if pd.notna(bb_lower.iloc[-1]) else None,
        "atr14": float(atr14.iloc[-1]) if pd.notna(atr14.iloc[-1]) else None,
        "volume_20d_avg": float(volume.tail(20).mean()),
        "hi_52": float(hi_52),
        "lo_52": float(lo_52),
        "pct_from_hi": float(pct_from_hi),
        "pct_from_lo": float(pct_from_lo),
        "returns": returns,
        "sma_signal": sma_signal,
        "trend_50_200": trend_50_200,
    }


def _summarize_backtest(strategy_cls) -> BacktestSummary:
    """執行回測並輸出摘要 / Run backtest and summarize."""
    results = run_backtest(
        strategy=strategy_cls,
        data_source=_latest_data_file(),
        cash=CASH,
        commission=COMMISSION,
        sizer_type="percent",
        sizer_params=SIZER_PARAMS,
        analyzers=ANALYZERS,
        print_output=False,
    )

    strat = results[0]
    final_value = float(strat.broker.getvalue())
    total_return_pct = (final_value / CASH - 1) * 100

    sharpe = None
    if hasattr(strat.analyzers, "sharpe"):
        sharpe = strat.analyzers.sharpe.get_analysis().get("sharperatio")

    max_drawdown_pct = None
    if hasattr(strat.analyzers, "drawdown"):
        dd = strat.analyzers.drawdown.get_analysis().get("max", {})
        max_drawdown_pct = dd.get("drawdown")

    trades = None
    win_rate_pct = None
    if hasattr(strat.analyzers, "trades"):
        ta = strat.analyzers.trades.get_analysis()
        total = ta.get("total", {}).get("total")
        won = ta.get("won", {}).get("total")
        if total:
            trades = int(total)
            if won is not None:
                win_rate_pct = (won / total) * 100 if total else None

    return BacktestSummary(
        strategy=strategy_cls.__name__,
        final_value=final_value,
        total_return_pct=total_return_pct,
        sharpe=sharpe,
        max_drawdown_pct=max_drawdown_pct,
        trades=trades,
        win_rate_pct=win_rate_pct,
    )


def _price_estimates(ind: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """以 BB + ATR 估算價格區間 / Estimate levels using BB + ATR."""
    bb_upper = ind.get("bb_upper")
    bb_lower = ind.get("bb_lower")
    atr14 = ind.get("atr14")

    if bb_upper is None or bb_lower is None or atr14 is None:
        return {"buy": None, "take_profit": None, "stop_loss": None}

    # 以波動緩衝調整 / Apply volatility buffer
    buy = bb_lower + 0.25 * atr14
    take_profit = bb_upper - 0.25 * atr14
    stop_loss = bb_lower - 0.75 * atr14

    return {"buy": buy, "take_profit": take_profit, "stop_loss": stop_loss}


def _fmt(value: Optional[float], digits: int = 2) -> str:
    """格式化數值 / Format numeric output."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NA"
    return f"{value:.{digits}f}"


def main() -> None:
    """主程序 / Main entry."""
    data_path = _latest_data_file()
    df = _load_price_data(data_path)

    # 多執行緒計算 / Multi-threaded execution
    with ThreadPoolExecutor(max_workers=3) as executor:
        ind_future = executor.submit(_indicator_report, df)
        sma_future = executor.submit(_summarize_backtest, CrossSMAStrategy)
        rsi_future = executor.submit(_summarize_backtest, RsiBollingerBandsStrategy)

        indicators = ind_future.result()
        sma_summary = sma_future.result()
        rsi_summary = rsi_future.result()

    levels = _price_estimates(indicators)

    # === 報告輸出 / Report output ===
    print("=" * 72)
    print("技術分析與價格估算報告 Technical Analysis & Price Estimation Report")
    print("=" * 72)
    print(f"資料來源 Data Source: {data_path}")
    print(f"截至日期 As-of Date: {indicators['asof'].date()}")
    print(f"收盤價 Close: {_fmt(indicators['last_close'])}")
    print()

    print("方法 Methods (Medical Journal Style / 醫學期刊風格)")
    print(
        "- 本研究採用 20 日布林通道（±2 標準差）與 14 日 ATR 作為波動基準；\n"
        "  We used 20-day Bollinger Bands (±2 SD) and 14-day ATR as volatility baselines."
    )
    print(
        "- 建議買入點=下軌+0.25×ATR；獲利了結點=上軌−0.25×ATR；風險停損點=下軌−0.75×ATR。\n"
        "  Buy=Lower Band+0.25×ATR; Take Profit=Upper Band−0.25×ATR; Stop Loss=Lower Band−0.75×ATR."
    )
    print(
        "- 指標與回測皆以同一資料區間（2022-01-01 起）計算，並以交易日收盤價為基準。\n"
        "  Indicators and backtests share the same data window and use daily closes."
    )
    print()

    print("技術指標 Technical Indicators")
    print(
        f"- SMA10/30/50/200: {_fmt(indicators['sma10'])} / {_fmt(indicators['sma30'])} / "
        f"{_fmt(indicators['sma50'])} / {_fmt(indicators['sma200'])}"
    )
    print(f"- SMA10 vs SMA30: {indicators['sma_signal']} (短中期趨勢 Short/Mid Trend)")
    print(f"- SMA50 vs SMA200: {indicators['trend_50_200']} (中長期趨勢 Mid/Long Trend)")
    print(f"- RSI14: {_fmt(indicators['rsi14'])}")
    print(
        f"- MACD/Signal/Hist: {_fmt(indicators['macd'])} / {_fmt(indicators['macd_signal'])} / "
        f"{_fmt(indicators['macd_hist'])}"
    )
    print(
        f"- Bollinger(20): upper {_fmt(indicators['bb_upper'])}, mid {_fmt(indicators['bb_mid'])}, "
        f"lower {_fmt(indicators['bb_lower'])}"
    )
    print(f"- ATR14: {_fmt(indicators['atr14'])}")
    print(f"- 20D Avg Volume: {_fmt(indicators['volume_20d_avg'], 0)}")
    print(
        f"- 52W High/Low: {_fmt(indicators['hi_52'])} / {_fmt(indicators['lo_52'])}, "
        f"距高點距離 From High: {_fmt(indicators['pct_from_hi'])}% , "
        f"距低點距離 From Low: {_fmt(indicators['pct_from_lo'])}%"
    )
    for label, val in indicators["returns"].items():
        print(f"- 報酬 Return {label}: {_fmt(val)}%")
    print()

    print("策略回測 Backtest Summary")
    print(
        f"- CrossSMAStrategy: Final {_fmt(sma_summary.final_value)} | Return "
        f"{_fmt(sma_summary.total_return_pct)}% | Sharpe {_fmt(sma_summary.sharpe)} | "
        f"MaxDD {_fmt(sma_summary.max_drawdown_pct)}% | Trades {sma_summary.trades} | "
        f"WinRate {_fmt(sma_summary.win_rate_pct)}%"
    )
    print(
        f"- RSI Strategy (RsiBollingerBandsStrategy): Final {_fmt(rsi_summary.final_value)} | "
        f"Return {_fmt(rsi_summary.total_return_pct)}% | Sharpe {_fmt(rsi_summary.sharpe)} | "
        f"MaxDD {_fmt(rsi_summary.max_drawdown_pct)}% | Trades {rsi_summary.trades} | "
        f"WinRate {_fmt(rsi_summary.win_rate_pct)}%"
    )
    print()

    print("價格估算 Price Estimation")
    print(
        f"- 建議買入點 Suggested Buy: {_fmt(levels['buy'])}\n"
        f"  (下軌+0.25×ATR / Lower Band + 0.25×ATR)"
    )
    print(
        f"- 獲利了結點 Suggested Take Profit: {_fmt(levels['take_profit'])}\n"
        f"  (上軌−0.25×ATR / Upper Band − 0.25×ATR)"
    )
    print(
        f"- 風險停損點 Suggested Stop Loss: {_fmt(levels['stop_loss'])}\n"
        f"  (下軌−0.75×ATR / Lower Band − 0.75×ATR)"
    )
    print()

    print("結論 Conclusion")
    print(
        "- 本結果為量化指標推估，適用於風險管理與情境比較；不構成投資建議。\n"
        "  Results are quantitative estimates for risk management and scenario comparison; not investment advice."
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
