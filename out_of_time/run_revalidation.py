"""Replay the original strategy on a new period without editing the core code."""

from __future__ import annotations
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch
import hashlib, json, sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import pipeline
from pipeline import (
    BacktestConfig,
    _extract_trades,
    backtest_pair,
    load_wide_prices,
    screen_pairs,
    select_pair,
)
from run_main import (
    add_bh_diagnostic,
    cost_decomposition,
    data_path,
    rolling_cointegration,
    MIN_RETURN_CORR,
)


def replay_frozen_initial(prices, split_date, mode, frozen):
    """Use the EXACT original backtest implementation with its 2019 initial fit.

    The original function has no initial-fit argument. A tightly scoped provider
    override supplies only the saved training regression/test result; signals,
    window fits, sizing, costs and metrics still execute the unmodified function.
    The override is removed before any new rolling cointegration diagnostics.
    """
    m = frozen["walk_forward"]
    initial = {
        "alpha": m["initial_alpha"],
        "beta": m["initial_beta"],
        "coint_stat": m["initial_coint_stat"],
        "critical_5pct": m["initial_coint_critical_5pct"],
        "cointegrated": m["initial_cointegrated"],
        "p_value": frozen["selected_pair"]["p_value"],
        "n_obs": m["train_days"],
    }
    y, x = frozen["selected_pair"]["dependent"], frozen["selected_pair"]["hedge_asset"]
    with patch.object(pipeline, "engle_granger_test", return_value=initial):
        daily, metrics = backtest_pair(prices, y, x, split_date, BacktestConfig(), mode)
    metrics["pre_evaluation_context_prices"] = metrics["train_days"]
    metrics["train_days"] = m["train_days"]
    metrics["coefficient_fit_end"] = "2019-12-11"
    return daily, metrics


def save_case(directory, prices, y, x, split_date, frozen=None):
    directory.mkdir(parents=True, exist_ok=True)
    cfg = BacktestConfig()
    modes = {}
    series = {}
    costs = []
    for mode in ["static", "walk_forward"]:
        if frozen is not None:
            daily, m = replay_frozen_initial(prices, split_date, mode, frozen)
        else:
            daily, m = backtest_pair(prices, y, x, split_date, cfg, mode)
        oos = daily.loc[daily.is_oos].copy()
        ends = pd.Series(daily.index, index=daily.index).shift(-1)
        oos["period_end"] = ends.loc[oos.index]
        for leg in ["dependent", "hedge"]:
            oos[f"buy_hold_{leg}_equity"] = (1 + oos[f"ret_{leg}"]).cumprod()
        oos["buy_hold_equal_initial_equity"] = 0.5 * (
            oos.buy_hold_dependent_equity + oos.buy_hold_hedge_equity
        )
        m.update(
            cost_decomposition=cost_decomposition(oos),
            buy_hold_hedge_return=float(oos.buy_hold_hedge_equity.iloc[-1] - 1),
            buy_hold_equal_initial_return=float(
                oos.buy_hold_equal_initial_equity.iloc[-1] - 1
            ),
        )
        ledger = _extract_trades(
            oos, cfg.entry_z, cfg.exit_z, cfg.stop_z, cfg.max_holding_days
        )
        ledger_return = float((1 + ledger.net_pnl).prod() - 1) if len(ledger) else 0.0
        if not np.isclose(ledger_return, m["total_return"], atol=1e-12):
            raise AssertionError("Trade ledger mismatch")
        m["trade_ledger_compound_return"] = ledger_return
        # Licensed close prices stay in memory for accounting only.
        oos.drop(columns=["dependent_price", "hedge_price"]).to_csv(
            directory / f"{mode}_daily.csv", index_label="date"
        )
        ledger.to_csv(directory / f"{mode}_trades.csv", index=False)
        modes[mode] = m
        series[mode] = oos
        costs.append({"mode": mode, **m["cost_decomposition"]})
    wf = series["walk_forward"]
    beta = wf.beta
    reference = modes["static"]["initial_beta"]
    beta_stats = {
        "training_reference_beta": reference,
        "min": float(beta.min()),
        "max": float(beta.max()),
        "std_ddof1": float(beta.std(ddof=1)),
        "mean_drift": float((beta - reference).mean()),
        "max_absolute_drift": float((beta - reference).abs().max()),
        "max_relative_drift": float((beta - reference).abs().max() / abs(reference)),
    }
    rolling = rolling_cointegration(prices[[y, x]], pd.Timestamp(split_date), 126)
    rolling.to_csv(directory / "rolling_cointegration.csv", index=False)
    wf[["alpha", "beta"]].to_csv(directory / "rolling_beta.csv", index_label="date")
    pd.DataFrame(costs).to_csv(directory / "cost_decomposition.csv", index=False)
    pd.DataFrame(
        [
            {"mode": mode, **{k: v for k, v in m.items() if not isinstance(v, dict)}}
            for mode, m in modes.items()
        ]
    ).to_csv(directory / "mode_comparison.csv", index=False)
    report = {
        "pair": [y, x],
        "split_date": str(pd.Timestamp(split_date).date()),
        "config": asdict(cfg),
        **modes,
        "beta_stability": beta_stats,
        "rolling_coint": {
            "window": 126,
            "windows": len(rolling),
            "reject_5pct_fraction": float((rolling.p_value < 0.05).mean()),
            "min_p": float(rolling.p_value.min()),
            "max_p": float(rolling.p_value.max()),
            "last_p": float(rolling.p_value.iloc[-1]),
        },
    }
    (directory / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    for mode, f in series.items():
        ax.plot(f.period_end, f.equity, label=mode + " net")
    ax.plot(
        wf.period_end, wf.buy_hold_dependent_equity, label="Buy & hold " + y, alpha=0.6
    )
    ax.plot(wf.period_end, wf.buy_hold_hedge_equity, label="Buy & hold " + x, alpha=0.6)
    ax.set(title=f"Out-of-time holdout: {y} / {x}", ylabel="Growth of $1")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(directory / "oos_equity.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(wf.index, beta, label="Prior 252-price beta")
    axes[0].axhline(
        reference, linestyle="--", color="black", label="Frozen training beta"
    )
    axes[1].plot(rolling.date, rolling.p_value, label="Within-OOT 126-price coint p")
    axes[1].axhline(0.05, color="red", linestyle="--")
    axes[0].set_ylabel("Beta")
    axes[1].set_ylabel("p-value")
    axes[1].set_ylim(-0.02, 1.02)
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(directory / "hedge_stability.png", dpi=160)
    plt.close(fig)
    return report


def main():
    path = ROOT / "data/aqte_universe_lseg_close.csv"
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["files"][path.name]:
        raise ValueError("New data SHA256 mismatch")
    source = load_wide_prices(data_path())
    new = load_wide_prices(path, min_observations=100)
    frozen = json.loads((ROOT.parent / "results/metrics.json").read_text())
    y, x = frozen["selected_pair"]["dependent"], frozen["selected_pair"]["hedge_asset"]
    cutoff = pd.Timestamp("2021-03-31")
    prefix = source.loc[source.index < cutoff, [y, x]]
    pair = pd.concat([prefix, new[[y, x]]]).sort_index()
    # Confirm unchanged price units when joining the old pair to the new vendor.
    anchor = pd.read_csv(
        ROOT / "data/adjusted_overlap_anchor.csv", index_col=0, parse_dates=True
    )
    mappings = pd.read_csv(ROOT / "data/identifier_mapping.csv").set_index(
        "source_column"
    )
    for col in [y, x]:
        value = float(anchor.iloc[0][mappings.loc[col, "ric"]])
        if abs(value - float(source.loc["2021-03-30", col])) > 0.011:
            raise ValueError("Fixed-pair vendor price-basis discontinuity")
    protocol = {
        "field": manifest["field"],
        "adjustments": manifest["adjustments"],
        "config": asdict(BacktestConfig()),
        "min_return_corr": MIN_RETURN_CORR,
        "pair_selection": "Most negative coint statistic among raw p < .05; BH diagnostic only.",
        "fixed_pair": [y, x],
        "static_alpha": frozen["walk_forward"]["initial_alpha"],
        "static_beta": frozen["walk_forward"]["initial_beta"],
        "coefficient_fit_period": ["2018-01-02", "2019-12-11"],
        "pre_evaluation_context_prices": len(prefix),
        "fixed_pair_start": "2021-03-31",
        "new_screen_train_fraction": 0.60,
        "core_pipeline_sha256": hashlib.sha256(
            (ROOT.parent / "pipeline.py").read_bytes()
        ).hexdigest(),
        "data_sha256": manifest["files"][path.name],
        "missing_data_policy": "Original loader and pairwise overlap; retain unavailable source columns in raw panel, never replace securities or select by OOS availability.",
    }
    (ROOT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print("Test 1: original AJG/AON fit and unmodified backtest...", flush=True)
    fixed = save_case(ROOT / "results/fixed_pair", pair, y, x, cutoff, frozen)
    split = new.index[int(len(new) * BacktestConfig().train_fraction)]
    print("Test 2: new-period training-only 0.70 screen...", flush=True)
    screening = add_bh_diagnostic(
        screen_pairs(new.loc[new.index < split], min_return_corr=MIN_RETURN_CORR)
    )
    directory = ROOT / "results/new_period_refit"
    directory.mkdir(parents=True, exist_ok=True)
    screening.to_csv(directory / "screening_results.csv", index=False)
    refit = {
        "raw_source_columns": manifest["source_columns"],
        "usable_columns": new.shape[1],
        "price_observations": len(new),
        "training_panel_prices": int(len(new) * 0.60),
        "holdout_panel_prices": len(new) - int(len(new) * 0.60),
        "split_date": str(split.date()),
        "tested_pairs": len(screening),
        "passed_raw_5pct": int(screening.cointegrated.sum()),
        "passed_bh_5pct": int(screening.bh_reject_5pct.sum()),
    }
    try:
        chosen = select_pair(screening)
    except ValueError:
        refit.update(status="NO_TRAINING_PAIR_PASSED", selected_pair=None)
    else:
        yy, xx = str(chosen.dependent), str(chosen.hedge_asset)
        refit["selected_pair"] = [yy, xx]
        refit["selected_coint_stat"] = float(chosen.coint_stat)
        refit["selected_p_value"] = float(chosen.p_value)
        try:
            refit["backtest"] = save_case(directory, new, yy, xx, split)
            refit["status"] = "COMPLETED"
        except ValueError as e:
            # Do not select a replacement pair using OOS availability.
            refit.update(status="SELECTED_PAIR_NOT_BACKTESTABLE", reason=str(e))
    (directory / "selection_summary.json").write_text(
        json.dumps(refit, indent=2, allow_nan=False) + "\n"
    )
    summary = {
        "status": "COMPLETED",
        "data_manifest": {k: v for k, v in manifest.items() if k != "files"},
        "fixed_pair": fixed,
        "new_period_refit": refit,
        "original_selected_pair_held_fixed": True,
        "main_results_overwritten": False,
    }
    (ROOT / "results/summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
