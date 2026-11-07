"""Tests for information boundaries, date alignment and accounting identities."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.stattools import coint

from pipeline import (
    BacktestConfig,
    _extract_trades,
    _metrics,
    _states_from_z,
    _z_score,
    backtest_pair,
    engle_granger_test,
    generate_synthetic_universe,
    load_long_prices,
    load_two_asset_price_files,
    load_wide_prices,
    screen_pairs,
    select_pair,
)
from run_main import (
    add_bh_diagnostic,
    cost_decomposition,
    data_path,
    rolling_cointegration,
)


@pytest.fixture(scope="module")
def synthetic():
    return generate_synthetic_universe(n=900)[0]


@pytest.fixture(scope="module")
def config():
    return BacktestConfig(hedge_window=120)


def test_engle_granger_outputs_one_consistent_statsmodels_test(synthetic):
    pair = synthetic[["AAPL", "ADBE"]].iloc[:540]
    test = engle_granger_test(pair["AAPL"], pair["ADBE"])
    stat, p, crit = coint(
        np.log(pair["AAPL"]),
        np.log(pair["ADBE"]),
        trend="c",
        maxlag=None,
        autolag="AIC",
    )
    assert test["coint_stat"] == pytest.approx(stat, abs=1e-12)
    assert test["p_value"] == pytest.approx(p, abs=1e-12)
    assert test["critical_5pct"] == pytest.approx(crit[1], abs=1e-12)
    assert test["cointegrated"] == (p < 0.05)


def test_synthetic_cointegrated_pair_is_selected(synthetic):
    chosen = select_pair(screen_pairs(synthetic.iloc[:540]))
    assert (chosen["dependent"], chosen["hedge_asset"]) == ("AAPL", "ADBE")


def test_selects_most_negative_statistic_only_among_passed_pairs():
    screen = pd.DataFrame(
        {
            "dependent": ["failed", "passed_a", "passed_b"],
            "hedge_asset": ["x"] * 3,
            "coint_stat": [-9.0, -4.0, -5.0],
            "return_corr": [0.8, 0.9, 0.5],
            "cointegrated": [False, True, True],
        }
    )
    assert select_pair(screen)["dependent"] == "passed_b"
    with pytest.raises(ValueError, match="No pair passed"):
        select_pair(screen.loc[~screen["cointegrated"]])


def test_independent_random_walks_are_not_selected():
    rng = np.random.default_rng(5)
    prices = pd.DataFrame(
        np.exp(4 + np.cumsum(rng.normal(0, 0.01, (900, 2)), axis=0)),
        columns=["independent_1", "independent_2"],
    )
    # Disable the correlation prescreen in this test to exercise the EG gate.
    tested = screen_pairs(prices, min_return_corr=-1.0)
    assert len(tested) == 1
    assert not tested["cointegrated"].any()
    with pytest.raises(ValueError):
        select_pair(tested)


@pytest.mark.parametrize("mode", ["static", "walk_forward"])
def test_future_price_changes_leave_current_signals_unchanged(synthetic, config, mode):
    changed = synthetic.copy()
    t = 660
    changed.iloc[t + 1 :, changed.columns.get_loc("AAPL")] *= np.linspace(
        2, 4, len(changed) - t - 1
    )
    a, _ = backtest_pair(synthetic, "AAPL", "ADBE", 540, config, mode)
    b, _ = backtest_pair(changed, "AAPL", "ADBE", 540, config, mode)
    columns = [
        "alpha",
        "beta",
        "spread",
        "z_score",
        "state",
        "weight_dependent",
        "weight_hedge",
    ]
    pd.testing.assert_frame_equal(a[columns].iloc[: t + 1], b[columns].iloc[: t + 1])


def test_oos_changes_cannot_change_training_pair_selection(synthetic):
    changed = synthetic.copy()
    changed.iloc[540:] *= np.linspace(1, 8, len(changed) - 540)[:, None]
    pd.testing.assert_frame_equal(
        screen_pairs(synthetic.iloc[:540]), screen_pairs(changed.iloc[:540])
    )


def test_z_score_excludes_current_and_numerator_observation_from_moments():
    spread = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0, 200.0])
    z = _z_score(spread, 3)
    assert z.iloc[4] == pytest.approx((4.0 - 2.0) / 1.0)
    assert z.iloc[5] == pytest.approx((100.0 - 3.0) / 1.0)


def test_split_resolves_panel_date_before_pairwise_dropna(synthetic, config):
    missing = synthetic.copy()
    missing.iloc[50:75, missing.columns.get_loc("AAPL")] = np.nan
    daily, metrics = backtest_pair(missing, "AAPL", "ADBE", 540, config)
    assert metrics["train_days"] == 515
    assert metrics["oos_start"] == str(synthetic.index[540].date())
    assert metrics["oos_days"] == 359
    assert daily.loc[daily["is_oos"]].index.min() == synthetic.index[540]


def test_main_aqte_dimensions_and_date_split():
    prices = load_wide_prices(data_path())
    assert prices.shape == (846, 226)
    train = int(len(prices) * BacktestConfig().train_fraction)
    assert train == 507
    assert len(prices) - train - 1 == 338
    assert str(prices.index[train].date()) == "2019-12-12"


@pytest.mark.parametrize("mode", ["static", "walk_forward"])
def test_trade_compounding_reconciles_to_daily_equity(synthetic, config, mode):
    daily, metrics = backtest_pair(synthetic, "AAPL", "ADBE", 540, config, mode)
    oos = daily.loc[daily["is_oos"]]
    trades = _extract_trades(
        oos, config.entry_z, config.exit_z, config.stop_z, config.max_holding_days
    )
    assert len(trades) > 0
    assert (1 + trades["net_pnl"]).prod() - 1 == pytest.approx(
        metrics["total_return"], abs=1e-12
    )
    assert (1 + oos["net_return"]).prod() == pytest.approx(
        oos["equity"].iloc[-1], abs=1e-12
    )


def test_cost_difference_equals_daily_transaction_plus_borrow(synthetic, config):
    zero = replace(config, transaction_cost_bps=0, borrow_bps=0)
    d0, _ = backtest_pair(synthetic, "AAPL", "ADBE", 540, zero)
    d1, _ = backtest_pair(synthetic, "AAPL", "ADBE", 540, config)
    oos = d1.loc[d1["is_oos"]]
    diff = d0.loc[d0["is_oos"], "net_return"] - oos["net_return"]
    np.testing.assert_allclose(
        diff, oos["transaction_cost"] + oos["borrow_cost"], atol=1e-15
    )
    assert diff.sum() == pytest.approx(
        oos["transaction_cost"].sum() + oos["borrow_cost"].sum()
    )
    c = cost_decomposition(oos)
    assert c["gross_compound_return"] - c["transaction_compound_drag"] - c[
        "borrow_compound_drag"
    ] == pytest.approx(c["net_compound_return"], abs=1e-15)


def test_metrics_passes_nondefault_exit_config_to_trade_extraction():
    idx = pd.bdate_range("2020-01-01", periods=3)
    daily = pd.DataFrame(
        {
            "state": [1, 1, 0],
            "net_return": [0.01, 0.01, -0.001],
            "z_score": [-2.1, -1.9, -1.8],
            "turnover": [1.0, 0.0, 1.0],
        },
        index=idx,
    )
    daily["equity"] = (1 + daily["net_return"]).cumprod()
    cfg = BacktestConfig(max_holding_days=2)
    trade = _extract_trades(
        daily, cfg.entry_z, cfg.exit_z, cfg.stop_z, cfg.max_holding_days
    )
    assert trade.iloc[0]["exit_reason"] == "MAX_HOLDING_DAYS"
    # Spying here verifies that _metrics actually forwards the supplied config.
    from unittest.mock import patch

    with patch("pipeline._extract_trades", wraps=_extract_trades) as extractor:
        _metrics(daily, cfg)
    assert extractor.call_args.args[1:] == (
        cfg.entry_z,
        cfg.exit_z,
        cfg.stop_z,
        cfg.max_holding_days,
    )


def test_wide_loader_cleans_errors_duplicates_invalid_dates_and_prices(tmp_path):
    path = tmp_path / "wide.csv"
    path.write_text(
        "Dates;A;B;too_short\n2020-01-03;3;#N/A;1\ninvalid;99;99;99\n2020-01-01;1;10;\n2020-01-02;0;inf;\n2020-01-03;4;12;\n2020-01-04;5;-1;\n"
    )
    result = load_wide_prices(path, min_observations=2)
    assert result.columns.tolist() == ["A", "B"]
    assert result.index.is_monotonic_increasing and result.index.is_unique
    assert result.loc["2020-01-03", "A"] == 4
    assert len(result) == 3  # the all-missing 2020-01-02 row is removed
    assert np.isnan(result.loc["2020-01-04", "B"])


def test_long_loader_pivots_and_keeps_last_duplicate(tmp_path):
    path = tmp_path / "long.csv"
    path.write_text(
        "date,ticker,price\n2020-01-02,A,5\n2020-01-01,A,3\n2020-01-01,A,4\n2020-01-01,B,-1\ninvalid,B,7\n"
    )
    loaded = load_long_prices(path)
    assert loaded.shape == (2, 1)
    assert loaded.iloc[0, 0] == 4


def test_two_file_semicolon_loader_has_no_personal_path(tmp_path):
    left, right = tmp_path / "left.csv", tmp_path / "right.csv"
    left.write_text("Dates;PX_LAST\n#NAME?;99\n2020-01-01;10\n2020-01-02;11\n")
    right.write_text("Dates;PX_LAST\n2020-01-01;20\n2020-01-02;21\n")
    loaded = load_two_asset_price_files(left, right, "Y", "X")
    assert loaded.shape == (2, 2)
    assert loaded.iloc[1].tolist() == [11, 21]


def test_aqte_requires_environment_variable(monkeypatch):
    monkeypatch.delenv("AQTE_DATA_DIR", raising=False)
    with pytest.raises(ValueError, match="AQTE_DATA_DIR"):
        data_path()


def test_bh_correction_is_diagnostic_and_does_not_reorder_selection():
    screen = pd.DataFrame(
        {
            "p_value": [0.001, 0.04, 0.2],
            "cointegrated": [True, True, False],
            "coint_stat": [-6.0, -4.0, -2.0],
            "return_corr": [0.5, 0.6, 0.7],
            "dependent": ["a", "b", "c"],
            "hedge_asset": ["x"] * 3,
        }
    )
    corrected = add_bh_diagnostic(screen)
    np.testing.assert_allclose(corrected["bh_q_value"], [0.003, 0.06, 0.2])
    assert corrected["bh_reject_5pct"].tolist() == [True, False, False]
    assert select_pair(corrected)["dependent"] == select_pair(screen)["dependent"]


def test_rolling_diagnostic_uses_only_oos_observations(synthetic):
    split = synthetic.index[540]
    diag = rolling_cointegration(synthetic[["AAPL", "ADBE"]], split, 126)
    assert diag["window_start"].min() == split
    assert diag["date"].iloc[0] == synthetic.index[665]
    assert (diag["n_obs"] == 126).all()


def test_frozen_aqte_prescreen_reproduces_original_144_candidate_pairs():
    from run_main import MIN_RETURN_CORR

    assert MIN_RETURN_CORR == 0.70
    prices = load_wide_prices(data_path())
    current = screen_pairs(prices.iloc[:507], min_return_corr=MIN_RETURN_CORR)
    saved = pd.read_csv(
        Path(__file__).resolve().parent / "testdata" / "screening_results_144.csv"
    )
    assert len(current) == 144
    assert set(zip(current.dependent, current.hedge_asset)) == set(
        zip(saved.dependent, saved.hedge_asset)
    )
    assert current["cointegrated"].sum() == 9
