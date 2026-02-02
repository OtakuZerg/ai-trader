#!/usr/bin/env python
"""Automated technical analysis & backtest report.
全自動化技術分析與回測報告。
"""

from __future__ import annotations

import glob
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd

from ai_trader.backtesting.strategies import classic
from ai_trader.backtesting.strategies.classic.rsi import RsiBollingerBandsStrategy
from ai_trader.backtesting.strategies.classic.sma import CrossSMAStrategy
from ai_trader.utils.backtest import run_backtest

# 非互動式繪圖後端 / Non-interactive plotting backend
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# === 全域設定 / Global settings ===
TICKERS = ["2313", "6282", "6188", "6285"]
START_DATE = "2022-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
DATA_GLOB_TMPL = "data/tw_stock/{ticker}_*.csv"
REPORT_PATH = Path("docs/2383_Analysis_Report.md")
FIG_DIR = Path("docs/figures")

CASH = 1_000_000
COMMISSION = 0.001425
SIZER_PARAMS = {"percents": 95}
ANALYZERS = ["sharpe", "drawdown", "returns", "trades"]

# 參數調整 / Parameter overrides
CROSS_SMA_PARAMS = {"fast": 10, "slow": 30}
RSI_PARAMS = {"oversold": 30, "overbought": 70}


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
    error: Optional[str] = None


@dataclass
class TickerResult:
    """個股結果彙整 / Per-ticker result container."""

    ticker: str
    data_path: Optional[Path]
    indicators: Optional[Dict[str, Any]]
    price_levels: Optional[Dict[str, Optional[float]]]
    backtests: List[BacktestSummary]
    chart_path: Optional[Path]
    notes: List[str]


def _find_latest_file(ticker: str) -> Optional[Path]:
    """尋找最新資料檔 / Find the newest data file."""
    candidates = [Path(p) for p in glob.glob(DATA_GLOB_TMPL.format(ticker=ticker))]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_price_data(path: Path) -> pd.DataFrame:
    """讀取與整理資料 / Load and clean price data."""
    df = pd.read_csv(path, parse_dates=["date"])
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

    windows = {"1M(21d)": 21, "3M(63d)": 63, "6M(126d)": 126, "1Y(252d)": 252}
    returns = {}
    for label, n in windows.items():
        if len(close) > n:
            returns[label] = (last_close / close.iloc[-n - 1] - 1) * 100
        else:
            returns[label] = None

    lookback = 252 if len(close) >= 252 else len(close)
    window = close.tail(lookback)
    hi_52 = window.max()
    lo_52 = window.min()
    pct_from_hi = (last_close / hi_52 - 1) * 100
    pct_from_lo = (last_close / lo_52 - 1) * 100

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


def _price_estimates(ind: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """整合 BB、ATR 與 MA 估算價位 / Estimate levels using BB, ATR, and MAs."""
    bb_upper = ind.get("bb_upper")
    bb_lower = ind.get("bb_lower")
    atr14 = ind.get("atr14")
    sma20 = ind.get("sma20")
    sma50 = ind.get("sma50")

    if None in (bb_upper, bb_lower, atr14, sma20, sma50):
        return {"buy": None, "take_profit": None, "stop_loss": None}

    ma_anchor = (sma20 + sma50) / 2
    buy = (bb_lower + 0.25 * atr14 + ma_anchor) / 2
    take_profit = (bb_upper - 0.25 * atr14 + ma_anchor) / 2
    stop_loss = min(bb_lower - 0.75 * atr14, ma_anchor - 1.0 * atr14)

    return {"buy": buy, "take_profit": take_profit, "stop_loss": stop_loss}


def _fmt(value: Optional[float], digits: int = 2) -> str:
    """格式化數值 / Format numeric output."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NA"
    return f"{value:.{digits}f}"


def _list_classic_strategies() -> List[Tuple[str, Any]]:
    """掃描可用策略 / Scan available strategies in classic module."""
    import inspect
    import backtrader as bt

    strategies = []
    for name, obj in inspect.getmembers(classic):
        if (
            inspect.isclass(obj)
            and issubclass(obj, bt.Strategy)
            and obj is not bt.Strategy
            and not name.startswith("_")
        ):
            strategies.append((name, obj))
    return sorted(strategies, key=lambda x: x[0])


def _summarize_backtest(strategy_cls, data_path: Path) -> BacktestSummary:
    """執行回測並輸出摘要 / Run backtest and summarize."""
    strategy_params = None
    if strategy_cls is CrossSMAStrategy:
        strategy_params = CROSS_SMA_PARAMS
    elif strategy_cls is RsiBollingerBandsStrategy:
        strategy_params = RSI_PARAMS

    try:
        results = run_backtest(
            strategy=strategy_cls,
            data_source=data_path,
            cash=CASH,
            commission=COMMISSION,
            sizer_type="percent",
            sizer_params=SIZER_PARAMS,
            analyzers=ANALYZERS,
            strategy_params=strategy_params,
            print_output=False,
        )
    except Exception as exc:  # 保留錯誤以利診斷 / Preserve error for diagnosis
        return BacktestSummary(
            strategy=strategy_cls.__name__,
            final_value=float("nan"),
            total_return_pct=float("nan"),
            sharpe=None,
            max_drawdown_pct=None,
            trades=None,
            win_rate_pct=None,
            error=str(exc),
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
        error=None,
    )


def _plot_chart(ticker: str, df: pd.DataFrame, ind: Dict[str, Any]) -> Path:
    """輸出技術圖表 / Export technical chart."""
    close = df["close"]
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()

    bb_mid = sma20
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    rsi14 = _calc_rsi(close, 14)

    high = df["high"]
    low = df["low"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr14 = tr.rolling(14).mean()

    # 交易點位：依 RSI+BB 規則 / Trade points using RSI+BB rule
    buy_idx = []
    sell_idx = []
    in_position = False
    for idx in df.index:
        if np.isnan(rsi14.loc[idx]) or np.isnan(bb_lower.loc[idx]) or np.isnan(bb_upper.loc[idx]):
            continue
        if not in_position and rsi14.loc[idx] < RSI_PARAMS["oversold"] and close.loc[idx] <= bb_lower.loc[idx]:
            buy_idx.append(idx)
            in_position = True
        elif in_position and (
            rsi14.loc[idx] > RSI_PARAMS["overbought"] or close.loc[idx] >= bb_upper.loc[idx]
        ):
            sell_idx.append(idx)
            in_position = False

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIG_DIR / f"{ticker}_technical.png"

    fig, (ax_price, ax_atr) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
        dpi=250,
    )

    ax_price.plot(close.index, close.values, color="#1f77b4", label="Close")
    ax_price.plot(bb_upper.index, bb_upper.values, color="#d62728", linestyle="--", label="BB Upper")
    ax_price.plot(bb_mid.index, bb_mid.values, color="#2ca02c", linestyle="-", label="BB Mid")
    ax_price.plot(bb_lower.index, bb_lower.values, color="#d62728", linestyle="--", label="BB Lower")
    ax_price.plot(sma20.index, sma20.values, color="#9467bd", label="SMA20")
    ax_price.plot(sma50.index, sma50.values, color="#8c564b", label="SMA50")

    ax_price.scatter(buy_idx, close.loc[buy_idx], marker="^", color="#2ca02c", s=30, label="Buy")
    ax_price.scatter(sell_idx, close.loc[sell_idx], marker="v", color="#d62728", s=30, label="Sell")

    ax_price.set_title(f"{ticker} Price with Bollinger Bands & Signals")
    ax_price.grid(True, alpha=0.3)
    ax_price.legend(loc="upper left", fontsize=8)

    ax_atr.plot(atr14.index, atr14.values, color="#ff7f0e", label="ATR14")
    ax_atr.set_title("ATR(14)")
    ax_atr.grid(True, alpha=0.3)
    ax_atr.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)

    return fig_path


def _render_report(results: List[TickerResult]) -> str:
    """輸出 Markdown 報告 / Render Markdown report."""
    lines: List[str] = []
    lines.append("# 全自動化深度技術分析與量化回測報告 / Automated Deep Technical Analysis & Backtest")
    lines.append("")
    lines.append(f"- 資料期間 Data Window: {START_DATE} 至 {TODAY}")
    lines.append("- 執行環境 Execution Environment: (AI_trader) conda")
    lines.append("- 多執行緒 Multi-threading: 啟用（平行回測與指標計算） / Enabled for parallel backtests and indicators")
    lines.append("")

    lines.append("## 方法 Methods (Academic Neutral Tone / 學術中性語氣)")
    lines.append(
        "- 策略掃描以 `ai_trader.backtesting.strategies.classic` 內建策略為基準。  "
        "Strategy scan is based on built-in classic strategies for single stocks."
    )
    lines.append(
        "- CrossSMAStrategy 參數調整為 Fast=10, Slow=30；RSI 策略採 RsiBollingerBandsStrategy，"
        "Oversold=30, Overbought=70。  "
        "CrossSMAStrategy uses Fast=10/Slow=30; RSI uses RsiBollingerBandsStrategy with 30/70 levels."
    )
    lines.append(
        "- 合理價估算結合布林通道、ATR 與移動平均：  "
        "Price estimation combines Bollinger Bands, ATR, and moving averages."
    )
    lines.append(
        "  Buy = (Lower Band + 0.25×ATR + MA Anchor)/2；  "
        "  Take Profit = (Upper Band − 0.25×ATR + MA Anchor)/2；  "
        "  Stop Loss = min(Lower Band − 0.75×ATR, MA Anchor − 1.0×ATR)。"
    )
    lines.append(
        "  MA Anchor = (SMA20 + SMA50)/2。  MA Anchor uses SMA20 and SMA50."
    )
    lines.append("")

    for result in results:
        lines.append(f"## 標的 {result.ticker} / Ticker {result.ticker}")
        if result.data_path is None:
            lines.append("- 資料缺失，無法分析。 / Data unavailable; analysis skipped.")
            lines.append("")
            continue

        ind = result.indicators or {}
        levels = result.price_levels or {}

        lines.append(f"- 資料來源 Data Source: `{result.data_path}`")
        lines.append(f"- 截至日期 As-of Date: {ind.get('asof').date()}")
        lines.append(f"- 收盤價 Close: {_fmt(ind.get('last_close'))}")
        if result.chart_path:
            lines.append(f"- 圖表 Chart: `{result.chart_path}`")
        lines.append("")

        lines.append("### 技術指標 Technical Indicators")
        lines.append(
            f"- SMA10/30/50/200: {_fmt(ind.get('sma10'))} / {_fmt(ind.get('sma30'))} / "
            f"{_fmt(ind.get('sma50'))} / {_fmt(ind.get('sma200'))}"
        )
        lines.append(
            f"- SMA10 vs SMA30: {ind.get('sma_signal')} (短中期 Short/Mid Trend)"
        )
        lines.append(
            f"- SMA50 vs SMA200: {ind.get('trend_50_200')} (中長期 Mid/Long Trend)"
        )
        lines.append(f"- RSI14: {_fmt(ind.get('rsi14'))}")
        lines.append(
            f"- MACD/Signal/Hist: {_fmt(ind.get('macd'))} / {_fmt(ind.get('macd_signal'))} / "
            f"{_fmt(ind.get('macd_hist'))}"
        )
        lines.append(
            f"- Bollinger(20): upper {_fmt(ind.get('bb_upper'))}, mid {_fmt(ind.get('bb_mid'))}, "
            f"lower {_fmt(ind.get('bb_lower'))}"
        )
        lines.append(f"- ATR14: {_fmt(ind.get('atr14'))}")
        lines.append(f"- 20D Avg Volume: {_fmt(ind.get('volume_20d_avg'), 0)}")
        lines.append(
            f"- 52W High/Low: {_fmt(ind.get('hi_52'))} / {_fmt(ind.get('lo_52'))}, "
            f"距高點 From High: {_fmt(ind.get('pct_from_hi'))}% , "
            f"距低點 From Low: {_fmt(ind.get('pct_from_lo'))}%"
        )
        for label, val in ind.get("returns", {}).items():
            lines.append(f"- 報酬 Return {label}: {_fmt(val)}%")
        lines.append("")

        lines.append("### 策略回測 Backtest Summary")
        lines.append("| Strategy | Final | Return% | Sharpe | MaxDD% | Trades | WinRate% | Note |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
        for s in result.backtests:
            note = s.error if s.error else "OK"
            lines.append(
                f"| {s.strategy} | {_fmt(s.final_value)} | {_fmt(s.total_return_pct)} | "
                f"{_fmt(s.sharpe)} | {_fmt(s.max_drawdown_pct)} | {s.trades or 'NA'} | "
                f"{_fmt(s.win_rate_pct)} | {note} |"
            )
        lines.append("")

        lines.append("### 合理價位診斷 Clinical Financial Diagnosis")
        lines.append(
            f"- 建議買入點 Suggested Buy: {_fmt(levels.get('buy'))}  "
            f"(BB/ATR/MA blend)"
        )
        lines.append(
            f"- 獲利了結點 Suggested Take Profit: {_fmt(levels.get('take_profit'))}  "
            f"(BB/ATR/MA blend)"
        )
        lines.append(
            f"- 風險停損位 Suggested Stop Loss: {_fmt(levels.get('stop_loss'))}  "
            f"(BB/ATR/MA blend)"
        )
        lines.append("")

        if result.chart_path:
            lines.append("### 視覺化 Visualization")
            lines.append(f"![]({result.chart_path.as_posix()})")
            lines.append("")

        if result.notes:
            lines.append("### 備註 Notes")
            for note in result.notes:
                lines.append(f"- {note}")
            lines.append("")

    lines.append("---")
    lines.append("本報告為量化指標推估，用於風險管理與情境比較，不構成投資建議。  ")
    lines.append("This report provides quantitative estimates for risk management and scenario comparison; not investment advice.")

    return "\n".join(lines)


def main() -> None:
    """主程序 / Main entry."""
    logging.getLogger().setLevel(logging.WARNING)

    strategies = _list_classic_strategies()

    results: List[TickerResult] = []
    max_workers = min(8, os.cpu_count() or 4)

    for ticker in TICKERS:
        data_path = _find_latest_file(ticker)
        if data_path is None:
            results.append(
                TickerResult(
                    ticker=ticker,
                    data_path=None,
                    indicators=None,
                    price_levels=None,
                    backtests=[],
                    chart_path=None,
                    notes=[
                        "資料檔不存在，可能為代碼無資料或抓取失敗。 "
                        "Data file not found; symbol may be unavailable or fetch failed."
                    ],
                )
            )
            continue

        df = _load_price_data(data_path)
        indicators = _indicator_report(df)
        levels = _price_estimates(indicators)
        chart_path = _plot_chart(ticker, df, indicators)

        # 多執行緒回測 / Multi-threaded backtests
        backtest_summaries: List[BacktestSummary] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(_summarize_backtest, cls, data_path): name
                for name, cls in strategies
            }
            for future in as_completed(future_map):
                backtest_summaries.append(future.result())

        # 依報酬排序 / Sort by return
        backtest_summaries.sort(key=lambda x: (x.total_return_pct if x.total_return_pct else -1e9), reverse=True)

        results.append(
            TickerResult(
                ticker=ticker,
                data_path=data_path,
                indicators=indicators,
                price_levels=levels,
                backtests=backtest_summaries,
                chart_path=chart_path,
                notes=[],
            )
        )

    report = _render_report(results)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
