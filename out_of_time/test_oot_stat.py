from pathlib import Path
import importlib.util, json, sys
import numpy as np
import pandas as pd
import pytest
from pipeline import load_wide_prices, backtest_pair, BacktestConfig

PATH = Path(__file__).parent / "run_revalidation.py"
spec = importlib.util.spec_from_file_location("oot_stat_revalidation", PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_frozen_fit_adapter_reproduces_original_backtest(aqte_source_path):
    prices = load_wide_prices(aqte_source_path)
    frozen = json.loads((PATH.parents[1] / "results/metrics.json").read_text())
    for mode in ["static", "walk_forward"]:
        original, metrics = backtest_pair(
            prices, "AJG_LAST", "AON_LAST", 507, BacktestConfig(), mode
        )
        replay, replayed = m.replay_frozen_initial(
            prices, prices.index[507], mode, frozen
        )
        pd.testing.assert_frame_equal(original, replay)
        assert replayed["total_return"] == metrics["total_return"]


def test_original_source_columns_are_preserved_in_download():
    path = PATH.parent / "data/aqte_universe_lseg_close.csv"
    if not path.is_file():
        pytest.skip("LSEG download not present; re-fetch with fetch_lseg.py to run this check")
    raw = pd.read_csv(path, nrows=0)
    assert raw.shape[1] - 1 == 226
    mapping = pd.read_csv(PATH.parent / "data/identifier_mapping.csv")
    assert len(mapping) == 226
    duplicate = mapping.loc[mapping.source_column == "ALXN_LAST.1"].iloc[0]
    assert pd.isna(duplicate.ric)
