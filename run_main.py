"""AQTE main experiment; all selection completes before OOS inspection."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from pipeline import (
    BacktestConfig,
    _extract_trades,
    backtest_pair,
    engle_granger_test,
    load_wide_prices,
    screen_pairs,
    select_pair,
)

ROOT = Path(__file__).resolve().parent
DATA_FILENAME = "SP500_Data_Sample_Prices.csv"
# Recovered from the saved AQTE experiment (144 candidates), not OOS returns.
MIN_RETURN_CORR = 0.70


def data_path() -> Path:
    directory = os.environ.get("AQTE_DATA_DIR")
    if not directory:
        raise ValueError(
            "Set AQTE_DATA_DIR to the directory containing " + DATA_FILENAME
        )
    path = Path(directory).expanduser() / DATA_FILENAME
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def add_bh_diagnostic(screening: pd.DataFrame) -> pd.DataFrame:
    """Diagnostic over every EG-tested pair; does not change selection."""
    result = screening.copy()
    if len(result):
        rejected, adjusted, _, _ = multipletests(
            result["p_value"], alpha=0.05, method="fdr_bh"
        )
        result["bh_q_value"] = adjusted
        result["bh_reject_5pct"] = rejected
    else:
        result["bh_q_value"] = pd.Series(dtype=float)
        result["bh_reject_5pct"] = pd.Series(dtype=bool)
    return result


def cost_decomposition(oos: pd.DataFrame) -> dict:
    gross = float((1 + oos["gross_return"]).prod() - 1)
    after_tc = float((1 + oos["gross_return"] - oos["transaction_cost"]).prod() - 1)
    net = float((1 + oos["net_return"]).prod() - 1)
    return {
        "gross_compound_return": gross,
        "after_transaction_compound_return": after_tc,
        "net_compound_return": net,
        "transaction_compound_drag": gross - after_tc,
        "borrow_compound_drag": after_tc - net,
        "gross_arithmetic_sum": float(oos["gross_return"].sum()),
        "transaction_cost_sum": float(oos["transaction_cost"].sum()),
        "borrow_cost_sum": float(oos["borrow_cost"].sum()),
        "net_arithmetic_sum": float(oos["net_return"].sum()),
    }


def rolling_cointegration(
    pair: pd.DataFrame, split_date: pd.Timestamp, window: int = 126
) -> pd.DataFrame:
    """Retrospective diagnostic, wholly OOS windows ending at the labelled date."""
    oos_prices = pair.loc[pair.index >= split_date].dropna()
    rows = []
    for end in range(window, len(oos_prices) + 1):
        sample = oos_prices.iloc[end - window : end]
        test = engle_granger_test(sample.iloc[:, 0], sample.iloc[:, 1])
        rows.append({"date": sample.index[-1], "window_start": sample.index[0], **test})
    return pd.DataFrame(rows)


def run(output: Path | None = None) -> dict:
    output = output or ROOT / "results"
    output.mkdir(parents=True, exist_ok=True)
    path = data_path()
    prices = load_wide_prices(path, min_observations=100)
    config = BacktestConfig()
    train_days = int(len(prices) * config.train_fraction)
    split_date = prices.index[train_days]
    print(
        f"AQTE: {prices.shape[1]} assets / {len(prices)} observations; screening training data only...",
        flush=True,
    )
    screening = add_bh_diagnostic(
        screen_pairs(
            prices.loc[prices.index < split_date], min_return_corr=MIN_RETURN_CORR
        )
    )
    chosen = select_pair(screening)
    y, x = str(chosen["dependent"]), str(chosen["hedge_asset"])
    screening.to_csv(output / "screening_results.csv", index=False)
    comparison, decompositions, oos_modes = [], [], {}
    mode_metrics = {}
    for mode in ("static", "walk_forward"):
        daily, metrics = backtest_pair(prices, y, x, split_date, config, mode)
        oos = daily.loc[daily["is_oos"]].copy()
        next_dates = pd.Series(daily.index, index=daily.index).shift(-1)
        oos["period_end"] = next_dates.loc[oos.index]
        oos["buy_hold_dependent_equity"] = (1 + oos["ret_dependent"]).cumprod()
        oos["buy_hold_hedge_equity"] = (1 + oos["ret_hedge"]).cumprod()
        oos["buy_hold_equal_initial_equity"] = 0.5 * (
            oos["buy_hold_dependent_equity"] + oos["buy_hold_hedge_equity"]
        )
        decomposition = cost_decomposition(oos)
        metrics["cost_decomposition"] = decomposition
        metrics["buy_hold_hedge_return"] = float(
            oos["buy_hold_hedge_equity"].iloc[-1] - 1
        )
        metrics["buy_hold_equal_initial_return"] = float(
            oos["buy_hold_equal_initial_equity"].iloc[-1] - 1
        )
        ledger = _extract_trades(
            oos, config.entry_z, config.exit_z, config.stop_z, config.max_holding_days
        )
        ledger_return = (
            float((1 + ledger["net_pnl"]).prod() - 1) if len(ledger) else 0.0
        )
        if not np.isclose(ledger_return, metrics["total_return"], atol=1e-12):
            raise AssertionError("Trade ledger does not reconcile to OOS equity")
        metrics["trade_ledger_compound_return"] = ledger_return
        # Price columns stay in memory for accounting; the published daily
        # tables drop them so licensed close prices are not redistributed.
        oos.drop(columns=["dependent_price", "hedge_price"]).to_csv(
            output / f"{mode}_daily.csv", index_label="date"
        )
        ledger.to_csv(output / f"{mode}_trades.csv", index=False)
        comparison.append(
            {
                "mode": mode,
                **{k: v for k, v in metrics.items() if not isinstance(v, dict)},
            }
        )
        decompositions.append({"mode": mode, **decomposition})
        mode_metrics[mode], oos_modes[mode] = metrics, oos
    wf = oos_modes["walk_forward"]
    beta, training_beta = wf["beta"], float(chosen["beta"])
    drift = beta - training_beta
    beta_stats = {
        "training_beta": training_beta,
        "min": float(beta.min()),
        "max": float(beta.max()),
        "std_ddof1": float(beta.std(ddof=1)),
        "mean": float(beta.mean()),
        "mean_drift_from_training": float(drift.mean()),
        "max_abs_drift_from_training": float(drift.abs().max()),
        "max_abs_relative_drift": float(drift.abs().max() / abs(training_beta)),
        "last_drift_from_training": float(drift.iloc[-1]),
    }
    rolling = rolling_cointegration(prices[[y, x]], split_date, window=126)
    rolling.to_csv(output / "rolling_cointegration.csv", index=False)
    beta.to_frame().assign(training_beta=training_beta, drift=drift).to_csv(
        output / "rolling_beta.csv", index_label="date"
    )
    pd.DataFrame(comparison).to_csv(output / "mode_comparison.csv", index=False)
    pd.DataFrame(decompositions).to_csv(output / "cost_decomposition.csv", index=False)
    summary = {
        "data": {
            "filename": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "cleaned_assets": prices.shape[1],
            "observations": len(prices),
            "train_days": train_days,
            "holdout_price_observations": len(prices) - train_days,
            "oos_return_periods": len(wf),
            "split_date": str(split_date.date()),
            "first_date": str(prices.index[0].date()),
            "last_date": str(prices.index[-1].date()),
            "missing_cells": int(prices.isna().sum().sum()),
            "unchanged_adjacent_price_cells": int((prices.diff() == 0).sum().sum()),
        },
        "config": asdict(config),
        "screening": {
            "possible_pairs": prices.shape[1] * (prices.shape[1] - 1) // 2,
            "tested_pairs": len(screening),
            "passed_5pct": int(screening["cointegrated"].sum()),
            "passed_bh_5pct": int(screening["bh_reject_5pct"].sum()),
            "prescreen": "training log-return correlation >= 0.70; >=100 overlapping prices",
            "selection_rule": "lowest coint statistic among raw p < .05; BH diagnostic only",
            "test": "statsmodels.coint(log y, log x, trend=c, maxlag=None, autolag=AIC)",
        },
        "selected_pair": {
            "dependent": y,
            "hedge_asset": x,
            "coint_stat": float(chosen["coint_stat"]),
            "p_value": float(chosen["p_value"]),
            "bh_q_value": float(chosen["bh_q_value"]),
        },
        "static": mode_metrics["static"],
        "walk_forward": mode_metrics["walk_forward"],
        "beta_stability": beta_stats,
        "rolling_coint_diagnostic": {
            "window": 126,
            "windows": len(rolling),
            "fraction_p_at_least_5pct": float((rolling["p_value"] >= 0.05).mean()),
            "min_p_value": float(rolling["p_value"].min()),
            "max_p_value": float(rolling["p_value"].max()),
            "last_p_value": float(rolling["p_value"].iloc[-1]),
        },
    }
    (output / "metrics.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    for mode, oos in oos_modes.items():
        ax.plot(
            [split_date, *oos["period_end"]],
            [1.0, *oos["equity"]],
            label=f"{mode}: net",
        )
    for col, label in [
        ("buy_hold_dependent_equity", f"Buy & hold {y}"),
        ("buy_hold_hedge_equity", f"Buy & hold {x}"),
        ("buy_hold_equal_initial_equity", "Buy & hold 50/50 initial capital"),
    ]:
        ax.plot(
            [split_date, *wf["period_end"]], [1.0, *wf[col]], label=label, alpha=0.65
        )
    ax.set(title=f"Holdout equity: {y} / {x}", ylabel="Growth of $1")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "oos_equity.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(wf.index, beta, label="Trailing 252-observation beta")
    axes[0].axhline(training_beta, color="black", linestyle="--", label="Training beta")
    axes[0].set(ylabel="Log-price OLS beta", title="Hedge-ratio stability diagnostic")
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    axes[1].plot(
        rolling["date"],
        rolling["p_value"],
        label="126-observation OOS Engle-Granger p-value",
    )
    axes[1].axhline(0.05, color="red", linestyle="--", label="5% threshold")
    axes[1].set(ylabel="p-value", ylim=(-0.02, 1.02))
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "hedge_stability.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    run(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
