from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd

from ai_trader.backtesting.strategies.classic.rsi import RSIStrategy
from ai_trader.backtesting.strategies.classic.sma import CrossSMAStrategy
from ai_trader.data.fetchers import TWStockFetcher
from ai_trader.utils.backtest import run_backtest


def fetch_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    # 下載歷史資料 / Download historical data
    fetcher = TWStockFetcher(symbol=symbol, start_date=start_date, end_date=end_date)
    df = fetcher.fetch()
    return df.loc[start_date:end_date]


def save_csv(df: pd.DataFrame, path: Path) -> None:
    # 儲存 CSV / Save CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index_label="date")


def compute_bbands_atr(df: pd.DataFrame) -> Dict[str, Any]:
    # 計算布林通道與 ATR / Compute Bollinger Bands and ATR
    close = df["close"]
    high = df["high"]
    low = df["low"]

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(14).mean()

    indicator_df = pd.DataFrame(
        {
            "close": close,
            "bb_mid": bb_mid,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "atr": atr,
        }
    ).dropna()

    last_row = indicator_df.iloc[-1]
    last_date = indicator_df.index[-1].date().isoformat()
    return {
        "date": last_date,
        "close": float(last_row["close"]),
        "bb_mid": float(last_row["bb_mid"]),
        "bb_upper": float(last_row["bb_upper"]),
        "bb_lower": float(last_row["bb_lower"]),
        "atr": float(last_row["atr"]),
        "series": indicator_df,
    }


def run_backtest_metrics(
    df: pd.DataFrame,
    strategy_cls,
    strategy_params: Dict[str, Any] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> Dict[str, Any]:
    # 多策略回測 / Multi-strategy backtest
    results = run_backtest(
        strategy=strategy_cls,
        data_source=df,
        cash=1000000,
        commission=0.001425,
        start_date=start_date,
        end_date=end_date,
        analyzers=["sharpe", "drawdown", "returns", "trades"],
        strategy_params=strategy_params,
        print_output=False,
    )
    strat = results[0]
    sharpe = strat.analyzers.sharpe.get_analysis().get("sharperatio")
    returns = strat.analyzers.returns.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    trades = strat.analyzers.trades.get_analysis()

    total_trades = trades.get("total", {}).get("total", 0)
    won = trades.get("won", {}).get("total", 0)
    lost = trades.get("lost", {}).get("total", 0)
    win_rate = (won / total_trades * 100) if total_trades else None

    return {
        "strategy": strategy_cls.__name__,
        "sharpe": float(sharpe) if sharpe is not None else None,
        "total_return_pct": float(returns.get("rtot", 0) * 100),
        "max_drawdown_pct": float(drawdown.get("max", {}).get("drawdown", 0)),
        "total_trades": int(total_trades),
        "win_rate_pct": float(win_rate) if win_rate is not None else None,
        "won_trades": int(won),
        "lost_trades": int(lost),
    }


def calculate_levels(bb_lower: float, bb_upper: float, atr: float) -> Dict[str, float]:
    # 估算合理價位 / Estimate valuation levels
    buy = bb_lower + 0.5 * atr
    take_profit = bb_upper - 0.5 * atr
    stop = bb_lower - 1.0 * atr
    return {
        "buy": float(buy),
        "take_profit": float(take_profit),
        "stop": float(stop),
    }


def plot_chart(
    symbol: str,
    indicator_df: pd.DataFrame,
    levels: Dict[str, float],
    output_path: Path,
) -> None:
    # 繪製技術圖表 / Render technical chart
    fig, (ax_price, ax_atr) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax_price.plot(indicator_df.index, indicator_df["close"], label="Close", color="#0b3d91")
    ax_price.plot(indicator_df.index, indicator_df["bb_mid"], label="BB Mid", color="#666666")
    ax_price.plot(indicator_df.index, indicator_df["bb_upper"], label="BB Upper", color="#d95f02")
    ax_price.plot(indicator_df.index, indicator_df["bb_lower"], label="BB Lower", color="#1b9e77")

    ax_price.axhline(levels["buy"], color="#2ca02c", linestyle="--", label="Buy Level")
    ax_price.axhline(levels["take_profit"], color="#ff7f0e", linestyle="--", label="Take Profit")
    ax_price.axhline(levels["stop"], color="#d62728", linestyle="--", label="Stop Loss")

    ax_price.set_title(f"{symbol} Bollinger Bands with Entry/Exit Levels")
    ax_price.set_ylabel("Price")
    ax_price.legend(loc="upper left", ncol=2, fontsize=9)
    ax_price.grid(True, alpha=0.2)

    ax_atr.plot(indicator_df.index, indicator_df["atr"], color="#9467bd", label="ATR(14)")
    ax_atr.set_ylabel("ATR")
    ax_atr.set_xlabel("Date")
    ax_atr.legend(loc="upper left", fontsize=9)
    ax_atr.grid(True, alpha=0.2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_report(
    symbol: str,
    start_date: str,
    end_date: str,
    row_count: int,
    backtests: Dict[str, Any],
    indicators: Dict[str, Any],
    levels: Dict[str, float],
    chart_path: Path,
    output_path: Path,
) -> None:
    # 產出報告 / Generate report
    report = f"""# {symbol} 深度技術分析與量化回測 / Deep Technical Analysis & Quant Backtest

## 資料與環境 / Data & Environment
- 資料區間 / Data range: {start_date} 至 {end_date}
- 樣本數 / Samples: {row_count}
- 環境 / Environment: AI_trader conda
- 計算方式 / Compute: multi-threading for backtests

## 策略回測結果 / Strategy Backtest Results
- CrossSMAStrategy (Fast=10, Slow=30)
  - Sharpe: {backtests["CrossSMAStrategy"]["sharpe"]}
  - Total Return (%): {backtests["CrossSMAStrategy"]["total_return_pct"]:.4f}
  - Max Drawdown (%): {backtests["CrossSMAStrategy"]["max_drawdown_pct"]:.4f}
  - Trades: {backtests["CrossSMAStrategy"]["total_trades"]}
  - Win Rate (%): {backtests["CrossSMAStrategy"]["win_rate_pct"]}
- RSIStrategy (Oversold=30, Overbought=70)
  - Sharpe: {backtests["RSIStrategy"]["sharpe"]}
  - Total Return (%): {backtests["RSIStrategy"]["total_return_pct"]:.4f}
  - Max Drawdown (%): {backtests["RSIStrategy"]["max_drawdown_pct"]:.4f}
  - Trades: {backtests["RSIStrategy"]["total_trades"]}
  - Win Rate (%): {backtests["RSIStrategy"]["win_rate_pct"]}

## 技術診斷（Medical Journal Style）/ Technical Diagnosis
以 20 日布林通道與 14 日 ATR 指標進行量化評估，並以審慎區間界定進出場風險。
Using 20-day Bollinger Bands and 14-day ATR, we define conservative zones for entry/exit risk.

### 今日指標 / Today’s Indicators
- Date: {indicators["date"]}
- Close: {indicators["close"]:.4f}
- BB Mid: {indicators["bb_mid"]:.4f}
- BB Upper: {indicators["bb_upper"]:.4f}
- BB Lower: {indicators["bb_lower"]:.4f}
- ATR(14): {indicators["atr"]:.4f}

### 合理價位估算 / Valuation Levels
以 {indicators["bb_lower"]:.4f} 與 {indicators["bb_upper"]:.4f} 為基準，
並以 ATR(14) {indicators["atr"]:.4f} 作為波動補正。
Based on BB lower/upper and ATR(14), the following levels are computed.

- 建議買入點 / Suggested Buy: {levels["buy"]:.4f}
- 獲利了結點 / Take Profit: {levels["take_profit"]:.4f}
- 風險停損點 / Stop Loss: {levels["stop"]:.4f}

## 圖表 / Chart
![{symbol} chart]({chart_path.as_posix()})
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")


def main() -> None:
    symbol = "2383"
    start_date = "2022-01-01"
    end_date = date.today().isoformat()

    df = fetch_data(symbol, start_date, end_date)
    save_csv(df, Path("data/tw_stock") / f"{symbol}.csv")

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            "CrossSMAStrategy": executor.submit(
                run_backtest_metrics,
                df,
                CrossSMAStrategy,
                {"fast": 10, "slow": 30},
                start_date,
                end_date,
            ),
            "RSIStrategy": executor.submit(
                run_backtest_metrics,
                df,
                RSIStrategy,
                {"rsi_period": 14, "oversold": 30, "overbought": 70},
                start_date,
                end_date,
            ),
        }
        backtest_results = {name: future.result() for name, future in futures.items()}

    indicators = compute_bbands_atr(df)
    levels = calculate_levels(indicators["bb_lower"], indicators["bb_upper"], indicators["atr"])

    indicator_df = indicators["series"]
    chart_path = Path("docs") / f"{symbol}_analysis.png"
    plot_chart(symbol, indicator_df, levels, chart_path)

    report_path = Path("docs") / f"{symbol}_analysis.md"
    build_report(
        symbol,
        start_date,
        end_date,
        len(df),
        backtest_results,
        indicators,
        levels,
        chart_path,
        report_path,
    )

    print(
        json.dumps(
            {
                "symbol": symbol,
                "start_date": start_date,
                "end_date": end_date,
                "row_count": int(len(df)),
                "report": report_path.as_posix(),
                "chart": chart_path.as_posix(),
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
