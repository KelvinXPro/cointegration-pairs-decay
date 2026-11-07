"""Run the research pipeline on a Bloomberg-style wide CSV export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline import (
    BacktestConfig,
    _extract_trades,
    backtest_pair,
    load_wide_prices,
    screen_pairs,
    select_pair,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run cointegration stat-arb on a wide price CSV"
    )
    parser.add_argument(
        "prices", help="CSV with one date column and one price column per asset"
    )
    parser.add_argument("--date-col", default=None)
    parser.add_argument("--output-dir", default="csv_output")
    parser.add_argument("--min-return-corr", type=float, default=0.40)
    parser.add_argument("--min-observations", type=int, default=100)
    parser.add_argument(
        "--hedge-mode", choices=["static", "walk_forward"], default="walk_forward"
    )
    args = parser.parse_args()

    prices = load_wide_prices(
        args.prices, date_col=args.date_col, min_observations=args.min_observations
    )
    config = BacktestConfig()
    train_end = int(len(prices) * config.train_fraction)
    if train_end <= config.hedge_window:
        raise SystemExit(
            "The file needs more observations for the default 252-day hedge window."
        )
    screening = screen_pairs(
        prices.iloc[:train_end], min_return_corr=args.min_return_corr
    )
    if screening.empty:
        raise SystemExit("No candidate pair passed the screening thresholds.")
    chosen = select_pair(screening)
    result, metrics = backtest_pair(
        prices,
        chosen["dependent"],
        chosen["hedge_asset"],
        train_end,
        config=config,
        hedge_mode=args.hedge_mode,
    )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    screening.to_csv(output / "screening_results.csv", index=False)
    oos = result.loc[result["is_oos"]]
    oos.to_csv(output / "daily_backtest.csv", index_label="date")
    _extract_trades(
        oos, config.entry_z, config.exit_z, config.stop_z, config.max_holding_days
    ).to_csv(output / "trades.csv", index=False)
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
