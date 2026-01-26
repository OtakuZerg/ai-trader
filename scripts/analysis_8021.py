from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from ai_trader.backtesting.strategies.classic.rsi import RSIStrategy
from ai_trader.backtesting.strategies.classic.sma import CrossSMAStrategy
from ai_trader.data.fetchers import TWStockFetcher
from ai_trader.utils.backtest import run_backtest


def fetch_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    fetcher = TWStockFetcher(symbol=symbol, start_date=start_date, end_date=end_date)
    df = fetcher.fetch()
    df = df.loc[start_date:end_date]
    return df


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index_label="date")


def compute_bbands_atr(df: pd.DataFrame) -> Dict[str, Any]:
    close = df["close"]
    high = df["high"]
    low = df["low"]

    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    upper = mid + 2 * std
    lower = mid - 2 * std

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
            "bb_mid": mid,
            "bb_upper": upper,
            "bb_lower": lower,
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
    }


def run_backtest_metrics(
    df: pd.DataFrame,
    strategy_cls,
    strategy_params: Dict[str, Any] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> Dict[str, Any]:
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


def main() -> None:
    symbol = "8021"
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
                {"fast": 5, "slow": 37},
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

    output = {
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "row_count": int(len(df)),
        "backtests": backtest_results,
        "indicators": indicators,
    }

    print(json.dumps(output, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
