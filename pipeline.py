"""Cointegration-based statistical arbitrage research pipeline.

This module is deliberately self-contained so it runs in a clean Python
environment with no market panel attached.  The demo uses synthetic prices;
the loaders accept a Bloomberg-style wide CSV or a generic long CSV.
Research code, not production trading code.

The backtest follows a strict information boundary:

* pair selection and the initial Engle--Granger hedge ratio use training data;
* rolling z-score statistics at date t use observations no later than t-2;
* a signal generated at t is held over t -> t+1;
* costs and an approximate borrow charge are deducted from returns.

Engle--Granger statistics, MacKinnon p-values and critical values all come
from the same statsmodels coint call (the residual step is an ADF test).
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint


@dataclass(frozen=True)
class BacktestConfig:
    train_fraction: float = 0.60
    z_window: int = 60
    hedge_window: int = 252
    entry_z: float = 2.0
    exit_z: float = 0.50
    stop_z: float = 4.0
    max_holding_days: int = 60
    transaction_cost_bps: float = 5.0
    borrow_bps: float = 50.0
    annualization: int = 252


def _as_numeric(series: pd.Series) -> pd.Series:
    """Convert Bloomberg/Excel error strings to numeric values."""

    bad = {"#N/A", "#N/A N/A", "#NAME?", "#VALUE!", ""}
    return pd.to_numeric(series.replace(list(bad), np.nan), errors="coerce")


def _read_csv_auto(path: str | Path) -> pd.DataFrame:
    """Read comma- or semicolon-delimited exports without changing the source."""

    with open(path, "r", encoding="utf-8-sig", errors="replace") as handle:
        first_line = handle.readline()
    separator = ";" if first_line.count(";") > first_line.count(",") else ","
    return pd.read_csv(path, sep=separator)


def load_wide_prices(
    path: str | Path,
    date_col: Optional[str] = None,
    min_observations: int = 100,
) -> pd.DataFrame:
    """Load a Bloomberg-style wide price export.

    Expected shape is one date column plus one column per asset, e.g.
    ``Dates, AAPL_LAST, ADBE_LAST``.  Error cells are removed and non-positive
    prices become missing.  Columns with too few observations are dropped.
    """

    frame = _read_csv_auto(path)
    if date_col is None:
        candidates = [
            c for c in frame.columns if str(c).lower() in {"date", "dates", "timestamp"}
        ]
        date_col = candidates[0] if candidates else frame.columns[0]
    dates = pd.to_datetime(frame[date_col], errors="coerce")
    frame = frame.drop(columns=[date_col]).copy()
    frame.index = dates
    frame = frame[~frame.index.isna()]
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    for col in frame.columns:
        frame[col] = _as_numeric(frame[col])
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.where(frame > 0)
    frame = frame.dropna(axis=1, thresh=min_observations)
    return frame.dropna(how="all")


def load_long_prices(
    path: str | Path,
    date_col: str = "date",
    asset_col: str = "ticker",
    price_col: str = "price",
) -> pd.DataFrame:
    """Load a generic long-format price file and pivot to date x asset."""

    frame = _read_csv_auto(path)
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame[price_col] = _as_numeric(frame[price_col])
    frame = frame.dropna(subset=[date_col, asset_col, price_col])
    frame = frame[frame[price_col] > 0]
    return frame.pivot_table(
        index=date_col, columns=asset_col, values=price_col, aggfunc="last"
    ).sort_index()


def load_two_asset_price_files(
    dependent_path: str | Path,
    hedge_path: str | Path,
    dependent_name: str = "dependent",
    hedge_name: str = "hedge",
    date_col: Optional[str] = None,
    price_col: str = "PX_LAST",
) -> pd.DataFrame:
    """Combine two Bloomberg-style single-asset exports into a price frame.

    AQTE's KO/PEP files use semicolons and contain an initial ``#NAME?`` date;
    this helper handles both details while preserving the remaining observations.
    """

    def read_one(path: str | Path, name: str) -> pd.Series:
        frame = _read_csv_auto(path)
        candidates = [
            c for c in frame.columns if str(c).lower() in {"date", "dates", "timestamp"}
        ]
        dcol = date_col or (candidates[0] if candidates else frame.columns[0])
        pcol = (
            price_col
            if price_col in frame.columns
            else next(
                c
                for c in frame.columns
                if str(c).upper() in {"PX_LAST", "PRICE", "CLOSE"}
            )
        )
        dates = pd.to_datetime(frame[dcol], errors="coerce")
        values = _as_numeric(frame[pcol])
        return pd.Series(values.to_numpy(), index=dates, name=name).dropna()

    left = read_one(dependent_path, dependent_name)
    right = read_one(hedge_path, hedge_name)
    result = pd.concat([left, right], axis=1).sort_index()
    result = result[~result.index.duplicated(keep="last")]
    result = result.where(result > 0).dropna(how="any")
    return result


def generate_synthetic_universe(
    n: int = 1_800, seed: int = 20260906
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a reproducible universe with one cointegrated pair.

    Synthetic series exist only to smoke-test the pipeline end to end when no
    market panel is available; they are not market observations.
    """

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2017-01-03", periods=n)

    common = np.cumsum(0.0002 + 0.0100 * rng.normal(size=n))
    ou = np.zeros(n)
    for i in range(1, n):
        ou[i] = 0.940 * ou[i - 1] + 0.010 * rng.normal()

    # AAPL/ADBE are the intentionally cointegrated pair.  The common factor
    # is non-stationary; the OU component creates a stationary residual.
    log_aapl = np.log(145.0) + common + 0.25 * ou + 0.0015 * rng.normal(size=n)
    log_adbe = np.log(115.0) + 1.08 * common - 0.25 * ou + 0.0015 * rng.normal(size=n)

    # Distractors have realistic market exposure but no deliberately stable
    # linear residual relationship with AAPL/ADBE.
    xom_factor = np.cumsum(0.0001 + 0.010 * rng.normal(size=n))
    jnj_factor = np.cumsum(0.00015 + 0.007 * rng.normal(size=n))
    wmt_factor = np.cumsum(0.0001 + 0.008 * rng.normal(size=n))
    log_xom = np.log(75.0) + 0.25 * common + xom_factor
    log_jnj = np.log(135.0) + 0.20 * common + jnj_factor
    log_wmt = np.log(100.0) + 0.15 * common + wmt_factor

    prices = pd.DataFrame(
        {
            "AAPL": np.exp(log_aapl),
            "ADBE": np.exp(log_adbe),
            "XOM": np.exp(log_xom),
            "JNJ": np.exp(log_jnj),
            "WMT": np.exp(log_wmt),
        },
        index=dates,
    )
    metadata = pd.DataFrame(
        {
            "sector": {
                "AAPL": "Technology",
                "ADBE": "Technology",
                "XOM": "Energy",
                "JNJ": "Health Care",
                "WMT": "Consumer Staples",
            }
        }
    )
    return prices, metadata


def _ols_with_intercept(
    y: np.ndarray, x: np.ndarray
) -> Tuple[float, float, np.ndarray]:
    """Return intercept, slope and residuals for y = alpha + beta*x."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coef
    return float(coef[0]), float(coef[1]), residual


def engle_granger_test(y_price: pd.Series, x_price: pd.Series) -> Dict[str, object]:
    """Log-price OLS and augmented Engle--Granger residual ADF via coint.

    Keep the original coint specification: trend="c", maxlag=None,
    autolag="AIC". Never rank its p-values by a separate ordinary ADF test.
    """
    pair = pd.concat([y_price, x_price], axis=1).dropna()
    if len(pair) < 50:
        raise ValueError("Not enough overlapping observations")
    y_log = np.log(pair.iloc[:, 0].to_numpy())
    x_log = np.log(pair.iloc[:, 1].to_numpy())
    alpha, beta, _ = _ols_with_intercept(y_log, x_log)
    stat, p_value, critical = coint(y_log, x_log, trend="c", maxlag=None, autolag="AIC")
    return {
        "alpha": alpha,
        "beta": beta,
        "n_obs": int(len(pair)),
        "coint_stat": float(stat),
        "critical_5pct": float(critical[1]),
        "p_value": float(p_value),
        "reject_5pct": bool(p_value < 0.05),
        "cointegrated": bool(p_value < 0.05),
    }


def select_pair(screening: pd.DataFrame) -> pd.Series:
    """Original selection rule: most negative coint statistic among p < .05."""
    passed = screening.loc[screening["cointegrated"]]
    if passed.empty:
        raise ValueError("No pair passed the 5% Engle-Granger test")
    return passed.sort_values(
        ["coint_stat", "return_corr"], ascending=[True, False], kind="stable"
    ).iloc[0]


def screen_pairs(
    prices: pd.DataFrame,
    metadata: Optional[pd.DataFrame] = None,
    min_return_corr: float = 0.40,
    require_same_sector: bool = False,
) -> pd.DataFrame:
    """Screen candidate pairs using training-only returns and EG tests."""

    clean = prices.dropna(axis=1, how="all").copy()
    returns = np.log(clean).diff()
    rows = []
    for y_name, x_name in itertools.combinations(clean.columns, 2):
        if require_same_sector and metadata is not None:
            if metadata.loc[y_name, "sector"] != metadata.loc[x_name, "sector"]:
                continue
        overlap = pd.concat([clean[y_name], clean[x_name]], axis=1).dropna()
        if len(overlap) < 100:
            continue
        return_corr = returns[[y_name, x_name]].dropna().corr().iloc[0, 1]
        if not np.isfinite(return_corr) or return_corr < min_return_corr:
            continue
        test = engle_granger_test(overlap[y_name], overlap[x_name])
        rows.append(
            {
                "dependent": y_name,
                "hedge_asset": x_name,
                "return_corr": float(return_corr),
                **test,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "dependent",
                "hedge_asset",
                "return_corr",
                "alpha",
                "beta",
                "n_obs",
                "coint_stat",
                "critical_5pct",
                "reject_5pct",
                "p_value",
                "cointegrated",
            ]
        )
    result = pd.DataFrame(rows)
    return result.sort_values(
        ["cointegrated", "coint_stat", "return_corr"], ascending=[False, True, False]
    ).reset_index(drop=True)


def _z_score(spread: pd.Series, window: int) -> pd.Series:
    """Lagged z-score: numerator and moments use information available at t."""

    observed = spread.shift(1)  # last close known before the t -> t+1 holding period
    history = observed.shift(1)  # excludes the observation used in the numerator
    mean = history.rolling(window, min_periods=window).mean()
    std = history.rolling(window, min_periods=window).std(ddof=1)
    return (observed - mean) / std.replace(0, np.nan)


def _states_from_z(
    z: pd.Series,
    entry_z: float,
    exit_z: float,
    stop_z: float,
    max_holding_days: int,
) -> pd.Series:
    """Create long-spread (+1), flat (0), short-spread (-1) states."""

    state = 0
    age = 0
    values = []
    for value in z.to_numpy(dtype=float):
        if not np.isfinite(value):
            values.append(state)
            age += int(state != 0)
            continue
        if state == 0:
            if value <= -entry_z:
                state = 1
                age = 0
            elif value >= entry_z:
                state = -1
                age = 0
        elif state == 1:
            age += 1
            if value >= -exit_z or abs(value) >= stop_z or age >= max_holding_days:
                state = 0
                age = 0
        elif state == -1:
            age += 1
            if value <= exit_z or abs(value) >= stop_z or age >= max_holding_days:
                state = 0
                age = 0
        values.append(state)
    return pd.Series(values, index=z.index, dtype=int)


def _walk_forward_parameters(
    y_log: pd.Series, x_log: pd.Series, window: int
) -> Tuple[pd.Series, pd.Series]:
    """Estimate alpha/beta at t from observations strictly before t."""

    alpha = pd.Series(np.nan, index=y_log.index, dtype=float)
    beta = pd.Series(np.nan, index=y_log.index, dtype=float)
    for i in range(window, len(y_log)):
        y_hist = y_log.iloc[i - window : i]
        x_hist = x_log.iloc[i - window : i]
        if y_hist.isna().any() or x_hist.isna().any():
            continue
        a, b, _ = _ols_with_intercept(y_hist.to_numpy(), x_hist.to_numpy())
        alpha.iloc[i] = a
        beta.iloc[i] = b
    return alpha, beta


def _extract_trades(
    daily: pd.DataFrame,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 4.0,
    max_holding_days: int = 60,
) -> pd.DataFrame:
    """Extract round-trip trades with exit-day fees attributed and open status tracked."""
    states = daily["state"]
    net_returns = daily["net_return"].fillna(0.0)
    z_scores = (
        daily["z_score"]
        if "z_score" in daily.columns
        else pd.Series(np.nan, index=daily.index)
    )

    trades = []
    in_trade = False
    start_idx = None
    direction = 0
    trade_returns = []

    for i in range(len(daily)):
        curr_state = int(states.iloc[i])
        curr_date = daily.index[i]
        if not in_trade:
            if curr_state != 0:
                in_trade = True
                start_idx = i
                direction = curr_state
                trade_returns = [net_returns.iloc[i]]
        else:
            if curr_state == direction:
                trade_returns.append(net_returns.iloc[i])
            elif curr_state == 0:
                # Exited to flat on day i: day i's net return (the exit fee included)
                # settles with this trade, so the ledger compounds to the equity curve.
                trade_returns.append(net_returns.iloc[i])
                pnl = float(np.prod([1.0 + r for r in trade_returns]) - 1.0)
                holding_days = len(trade_returns) - 1
                exit_z_val = (
                    float(z_scores.iloc[i]) if np.isfinite(z_scores.iloc[i]) else np.nan
                )
                last_z_val = (
                    float(z_scores.iloc[i - 1])
                    if np.isfinite(z_scores.iloc[i - 1])
                    else np.nan
                )

                if holding_days >= max_holding_days:
                    reason = "MAX_HOLDING_DAYS"
                elif abs(exit_z_val) >= stop_z or abs(last_z_val) >= stop_z:
                    reason = "STOP_LOSS_4SIGMA"
                elif (direction == 1 and exit_z_val >= -exit_z) or (
                    direction == -1 and exit_z_val <= exit_z
                ):
                    reason = "Z_SCORE_REVERSION"
                else:
                    reason = "STATE_CHANGE"

                trades.append(
                    {
                        "entry_date": daily.index[start_idx],
                        "exit_date": curr_date,
                        "direction": (
                            "LONG_SPREAD" if direction == 1 else "SHORT_SPREAD"
                        ),
                        "holding_days": holding_days,
                        "exit_z": exit_z_val,
                        "exit_reason": reason,
                        "net_pnl": pnl,
                        "is_win": bool(pnl > 0),
                        "is_closed": True,
                    }
                )
                in_trade = False
                start_idx = None
                direction = 0
                trade_returns = []

    # If trade was still held at the very end of sample
    if in_trade and trade_returns:
        pnl = float(np.prod([1.0 + r for r in trade_returns]) - 1.0)
        trades.append(
            {
                "entry_date": daily.index[start_idx],
                "exit_date": daily.index[-1],
                "direction": "LONG_SPREAD" if direction == 1 else "SHORT_SPREAD",
                "holding_days": len(trade_returns),
                "exit_z": (
                    float(z_scores.iloc[-1])
                    if np.isfinite(z_scores.iloc[-1])
                    else np.nan
                ),
                "exit_reason": "OPEN_AT_END",
                "net_pnl": pnl,
                "is_win": bool(pnl > 0),
                "is_closed": False,
            }
        )
    return pd.DataFrame(trades)


def _metrics(daily: pd.DataFrame, config: BacktestConfig) -> Dict[str, float | int]:
    """Calculate OOS performance and implementation diagnostics."""

    annualization = config.annualization
    r = daily["net_return"].fillna(0.0)
    equity = daily["equity"]
    active = daily["state"].abs() > 0
    valid_r = r.loc[daily.index]
    n = max(len(daily), 1)
    total_return = float(equity.iloc[-1] - 1.0) if len(equity) else 0.0
    ann_return = (
        float((1.0 + total_return) ** (annualization / n) - 1.0)
        if 1.0 + total_return > 0
        else -1.0
    )
    vol = (
        float(valid_r.std(ddof=1) * np.sqrt(annualization)) if len(valid_r) > 1 else 0.0
    )
    sharpe = (
        float(valid_r.mean() / valid_r.std(ddof=1) * np.sqrt(annualization))
        if valid_r.std(ddof=1) > 0
        else 0.0
    )
    drawdown = equity / equity.cummax() - 1.0

    # True round-trip trade statistics (evaluated strictly on completed closed trades)
    trade_df = _extract_trades(
        daily, config.entry_z, config.exit_z, config.stop_z, config.max_holding_days
    )
    closed_df = (
        trade_df[trade_df["is_closed"]]
        if ("is_closed" in trade_df.columns and not trade_df.empty)
        else trade_df
    )
    closed_count = len(closed_df)
    winning_trades = int(closed_df["is_win"].sum()) if closed_count else 0
    trade_win_rate = float(winning_trades / closed_count) if closed_count else 0.0
    avg_trade_pnl = float(closed_df["net_pnl"].mean()) if closed_count else 0.0
    avg_holding_days = float(closed_df["holding_days"].mean()) if closed_count else 0.0
    open_count = len(trade_df) - closed_count

    # Daily hit rate
    wins = int((valid_r.loc[active] > 0).sum()) if active.any() else 0
    active_count = int(active.sum())
    total_turnover = (
        float(daily["turnover"].sum()) if "turnover" in daily.columns else 0.0
    )

    return {
        "oos_days": int(len(daily)),
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "exposure": float(active.mean()) if len(active) else 0.0,
        "turnover_per_day": float(total_turnover / n),
        "total_turnover": total_turnover,
        "round_trip_trades": closed_count,
        "winning_trades": winning_trades,
        "trade_win_rate": trade_win_rate,
        "open_trades_at_end": open_count,
        "avg_trade_pnl": avg_trade_pnl,
        "avg_holding_days": avg_holding_days,
        "active_day_hit_rate": float(wins / active_count) if active_count else 0.0,
        "active_days": active_count,
    }


def backtest_pair(
    prices: pd.DataFrame,
    dependent: str,
    hedge_asset: str,
    train_end: int | str | pd.Timestamp,
    config: BacktestConfig = BacktestConfig(),
    hedge_mode: str = "walk_forward",
) -> Tuple[pd.DataFrame, Dict[str, float | int | bool | None]]:
    """Run a cost-aware OOS pair backtest.

    ``hedge_mode='static'`` uses one hedge ratio estimated on training data;
    ``hedge_mode='walk_forward'`` recalibrates the ratio using a trailing
    window ending at t-1.  In both modes the pair itself is selected from the
    training sample before this function is called.
    """

    # Resolve the panel boundary BEFORE pairwise missing-value removal.
    # An integer remains supported for old callers, but is immediately mapped
    # to a date in the original panel, never to a row in the cleaned pair.
    if isinstance(train_end, (int, np.integer)):
        if not 0 < train_end < len(prices):
            raise ValueError("Invalid panel training boundary")
        split_date = prices.index[train_end]
    else:
        split_date = pd.Timestamp(train_end)
    pair = prices[[dependent, hedge_asset]].dropna().copy()
    training = pair.loc[pair.index < split_date]
    if (
        len(training) <= config.hedge_window
        or len(pair.loc[pair.index >= split_date]) < 11
    ):
        raise ValueError("train_end leaves too little data for train/OOS split")
    y_log = np.log(pair[dependent])
    x_log = np.log(pair[hedge_asset])

    initial = engle_granger_test(training[dependent], training[hedge_asset])
    if hedge_mode not in {"static", "walk_forward"}:
        raise ValueError("hedge_mode must be 'static' or 'walk_forward'")
    if hedge_mode == "static":
        alpha = pd.Series(initial["alpha"], index=pair.index, dtype=float)
        beta = pd.Series(initial["beta"], index=pair.index, dtype=float)
    else:
        alpha, beta = _walk_forward_parameters(y_log, x_log, config.hedge_window)

    spread = y_log - alpha - beta * x_log
    z = _z_score(spread, config.z_window)
    oos_index = pair.index[pair.index >= split_date]
    states = pd.Series(0, index=pair.index, dtype=int)
    states.loc[oos_index] = _states_from_z(
        z.loc[oos_index],
        config.entry_z,
        config.exit_z,
        config.stop_z,
        config.max_holding_days,
    )

    # Return for t -> t+1, with state determined only from information through
    # t-1.  The final observation has no forward return and is excluded.
    ret_y = pair[dependent].shift(-1) / pair[dependent] - 1.0
    ret_x = pair[hedge_asset].shift(-1) / pair[hedge_asset] - 1.0

    # Dollar weights under the log-price regression ln(P_Y) = alpha + beta*ln(P_X),
    # where r_Y ≈ beta * r_X. Gross-normalized beta weights:
    # w_Y = state / (1 + |beta|), w_X = -state * beta / (1 + |beta|).
    # The spread is hedged in return sensitivity; capital is not dollar-neutral.
    gross_notional = 1.0 + beta.abs()
    weight_y = states / gross_notional
    weight_x = -states * beta / gross_notional

    gross_return = weight_y * ret_y + weight_x * ret_x

    # Full portfolio turnover at rebalancing (including daily rolling beta adjustments):
    # One-way turnover = (|w_Y,t - w_Y,t-1| + |w_X,t - w_X,t-1|) / 2.0 (or gross traded notional)
    # Here we track total traded notional: dw_y + dw_x
    dw_y = weight_y.diff().fillna(weight_y).abs()
    dw_x = weight_x.diff().fillna(weight_x).abs()
    turnover = dw_y + dw_x
    transaction_cost = turnover * config.transaction_cost_bps / 10_000.0

    short_weight = np.maximum(-weight_y, 0.0) + np.maximum(-weight_x, 0.0)
    borrow_cost = (
        pd.Series(short_weight, index=pair.index)
        * config.borrow_bps
        / 10_000.0
        / config.annualization
    )
    net_return = gross_return - transaction_cost - borrow_cost
    net_return.loc[pair.index < split_date] = 0.0
    net_return.iloc[-1] = 0.0
    equity = (1.0 + net_return.fillna(0.0)).cumprod()

    result = pd.DataFrame(
        {
            "dependent_price": pair[dependent],
            "hedge_price": pair[hedge_asset],
            "alpha": alpha,
            "beta": beta,
            "spread": spread,
            "z_score": z,
            "state": states,
            "weight_dependent": weight_y,
            "weight_hedge": weight_x,
            "ret_dependent": ret_y,
            "ret_hedge": ret_x,
            "gross_return": gross_return,
            "turnover": turnover,
            "transaction_cost": transaction_cost,
            "borrow_cost": borrow_cost,
            "net_return": net_return,
            "equity": equity,
        },
        index=pair.index,
    )
    result["is_oos"] = (result.index >= split_date) & (result.index < pair.index[-1])
    oos = result.loc[result["is_oos"]].copy()
    metrics = _metrics(oos, config)
    metrics.update(
        {
            "dependent": dependent,
            "hedge_asset": hedge_asset,
            "hedge_mode": hedge_mode,
            "train_days": int(len(training)),
            "train_start": str(pair.index[0].date()),
            "split_date": str(split_date.date()),
            "oos_start": str(oos_index[0].date()),
            "oos_end": str(pair.index[-1].date()),
            "initial_alpha": float(initial["alpha"]),
            "initial_beta": float(initial["beta"]),
            "initial_coint_stat": float(initial["coint_stat"]),
            "initial_coint_critical_5pct": float(initial["critical_5pct"]),
            "initial_cointegrated": bool(initial["cointegrated"]),
            "buy_hold_dependent_return": float(
                pair[dependent].iloc[-1] / pair.loc[oos_index[0], dependent] - 1.0
            ),
        }
    )
    return result, metrics


def run_demo(output_dir: str | Path) -> Dict[str, object]:
    """Run the full synthetic smoke test and persist its outputs."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prices, metadata = generate_synthetic_universe()
    config = BacktestConfig()
    train_end = int(len(prices) * config.train_fraction)
    training_prices = prices.iloc[:train_end]
    screening = screen_pairs(training_prices, metadata=metadata, min_return_corr=0.40)
    if screening.empty:
        raise RuntimeError("Synthetic screen produced no candidate pair")
    chosen = select_pair(screening)

    wf_result, wf_metrics = backtest_pair(
        prices,
        str(chosen["dependent"]),
        str(chosen["hedge_asset"]),
        train_end,
        config=config,
        hedge_mode="walk_forward",
    )
    static_result, static_metrics = backtest_pair(
        prices,
        str(chosen["dependent"]),
        str(chosen["hedge_asset"]),
        train_end,
        config=config,
        hedge_mode="static",
    )
    prices.to_csv(output / "synthetic_prices.csv", index_label="date")
    screening.to_csv(output / "screening_results.csv", index=False)
    wf_result.loc[wf_result["is_oos"]].to_csv(
        output / "walk_forward_daily.csv", index_label="date"
    )
    static_result.loc[static_result["is_oos"]].to_csv(
        output / "static_daily.csv", index_label="date"
    )
    _extract_trades(
        wf_result.loc[wf_result["is_oos"]],
        config.entry_z,
        config.exit_z,
        config.stop_z,
        config.max_holding_days,
    ).to_csv(output / "walk_forward_trades.csv", index=False)

    summary = {
        "data": {
            "kind": "synthetic_smoke_test",
            "n_rows": int(len(prices)),
            "n_assets": int(prices.shape[1]),
            "start": str(prices.index[0].date()),
            "end": str(prices.index[-1].date()),
            "warning": "Replace synthetic data with Bloomberg/WRDS observations before using any result externally.",
        },
        "config": asdict(config),
        "selected_pair": {
            k: (float(v) if isinstance(v, (np.floating, float)) else v)
            for k, v in chosen.to_dict().items()
        },
        "walk_forward": wf_metrics,
        "static_hedge": static_metrics,
    }
    (output / "demo_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(
            wf_result.index[train_end:],
            wf_result["equity"].iloc[train_end:],
            label="Walk-forward hedge",
        )
        ax.plot(
            static_result.index[train_end:],
            static_result["equity"].iloc[train_end:],
            label="Static training hedge",
            alpha=0.75,
        )
        ax.set_title(f"OOS equity: {chosen['dependent']} vs {chosen['hedge_asset']}")
        ax.set_ylabel("Growth of $1")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output / "oos_equity.png", dpi=160)
        plt.close(fig)
    except ImportError:
        pass
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the cointegration stat-arb synthetic smoke test"
    )
    parser.add_argument(
        "--output-dir", default=str(Path(__file__).with_name("demo_output"))
    )
    args = parser.parse_args()
    summary = run_demo(args.output_dir)
    wf = summary["walk_forward"]
    print(
        json.dumps(
            {
                "selected_pair": summary["selected_pair"],
                "oos_days": wf["oos_days"],
                "total_return": wf["total_return"],
                "sharpe": wf["sharpe"],
                "max_drawdown": wf["max_drawdown"],
                "round_trip_trades": wf["round_trip_trades"],
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
