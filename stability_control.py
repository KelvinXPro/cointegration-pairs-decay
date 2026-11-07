"""Training-period control for the stability diagnostics.

Writes new diagnostic artifacts only; the main experiment is never rerun here.
"""

from __future__ import annotations
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pipeline import BacktestConfig, _walk_forward_parameters, load_wide_prices
from run_main import ROOT, data_path, rolling_cointegration

OUTPUT = ROOT / "stability_control"


def calculate(prices: pd.DataFrame) -> tuple[dict, dict, dict]:
    frozen = json.loads((ROOT / "results/metrics.json").read_text())
    y, x = frozen["selected_pair"]["dependent"], frozen["selected_pair"]["hedge_asset"]
    pair = prices[[y, x]].dropna()
    split = pd.Timestamp(frozen["data"]["split_date"])
    cfg = BacktestConfig()
    a, b = _walk_forward_parameters(np.log(pair[y]), np.log(pair[x]), cfg.hedge_window)
    betas = {
        "training": pd.DataFrame({"alpha": a, "beta": b})
        .loc[lambda f: f.index < split]
        .dropna(),
        "holdout": pd.DataFrame({"alpha": a, "beta": b})
        .loc[lambda f: (f.index >= split) & (f.index < pair.index[-1])]
        .dropna(),
    }
    train = pair.loc[pair.index < split]
    coint = {
        "training": rolling_cointegration(train, train.index[0], 126),
        "holdout": rolling_cointegration(pair, split, 126),
    }
    rows = []
    for label, segment in [
        ("training", train),
        ("holdout", pair.loc[pair.index >= split]),
    ]:
        bp, cp = betas[label], coint[label]
        rows.append(
            {
                "segment": label,
                "price_start": str(segment.index[0].date()),
                "price_end": str(segment.index[-1].date()),
                "price_observations": len(segment),
                "beta_estimates": len(bp),
                "beta_min": float(bp.beta.min()),
                "beta_max": float(bp.beta.max()),
                "beta_std_ddof1": float(bp.beta.std(ddof=1)),
                "beta_mean": float(bp.beta.mean()),
                "coint_windows": len(cp),
                "coint_reject_5pct": int((cp.p_value < 0.05).sum()),
                "coint_reject_fraction": float((cp.p_value < 0.05).mean()),
                "coint_p_min": float(cp.p_value.min()),
                "coint_p_max": float(cp.p_value.max()),
            }
        )
    return (
        {
            "pair": [y, x],
            "beta_window": 252,
            "coint_window": 126,
            "training_full_sample_p_value": frozen["selected_pair"]["p_value"],
            "training_reference_beta": frozen["walk_forward"]["initial_beta"],
            "segments": rows,
            "causal_attribution": "Overlapping-window rejection fractions do not identify statistical power or a causal percentage attributable to relationship change.",
        },
        betas,
        coint,
    )


def main():
    prices = load_wide_prices(data_path())
    summary, betas, coint = calculate(prices)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary["segments"]).to_csv(OUTPUT / "comparison.csv", index=False)
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey="row")
    for j, label in enumerate(["training", "holdout"]):
        bp, cp = betas[label], coint[label]
        bp.to_csv(OUTPUT / f"{label}_beta.csv", index_label="date")
        cp.to_csv(OUTPUT / f"{label}_rolling_coint.csv", index=False)
        axes[0, j].plot(bp.index, bp.beta, label="Prior 252-price OLS beta")
        axes[0, j].axhline(
            summary["training_reference_beta"],
            color="black",
            linestyle="--",
            label="Full training beta",
        )
        axes[0, j].set_title(
            "Training control" if label == "training" else "Original holdout"
        )
        axes[1, j].plot(cp.date, cp.p_value, label="Within-segment 126-price coint p")
        axes[1, j].axhline(0.05, color="red", linestyle="--", label="5% threshold")
        axes[1, j].set_ylim(-0.02, 1.02)
        for i in [0, 1]:
            axes[i, j].grid(alpha=0.2)
            axes[i, j].legend(fontsize=7)
            axes[i, j].tick_params(axis="x", labelrotation=25)
    axes[0, 0].set_ylabel("Log-price hedge beta")
    axes[1, 0].set_ylabel("p-value")
    fig.suptitle("Same AJG/AON diagnostics: training control vs holdout")
    fig.tight_layout()
    fig.savefig(OUTPUT / "training_vs_holdout.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
