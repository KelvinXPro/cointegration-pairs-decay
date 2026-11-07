import numpy as np
import pandas as pd
from pipeline import load_wide_prices
from stability_control import calculate


def test_training_control_uses_same_windows_and_stays_inside_training(aqte_source_path):
    summary, betas, coint = calculate(load_wide_prices(aqte_source_path))
    train, holdout = summary["segments"]
    assert train["price_start"] == "2018-01-02" and train["price_end"] == "2019-12-11"
    assert train["price_observations"] == 507
    assert train["coint_windows"] == 507 - 126 + 1
    assert (coint["training"].n_obs == 126).all()
    assert coint["training"].date.max() <= pd.Timestamp("2019-12-11")
    assert len(betas["training"]) == 507 - 252
    frozen = pd.read_csv(
        __import__("pathlib").Path(__file__).parent / "results/rolling_beta.csv"
    )
    np.testing.assert_allclose(betas["holdout"].beta, frozen.beta, atol=1e-12)
