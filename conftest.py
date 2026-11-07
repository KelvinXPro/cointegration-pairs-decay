"""Skip the AQTE-data tests when the external price panel is not available.

The main experiment needs a course price panel that cannot be redistributed.
Without AQTE_DATA_DIR those tests skip with an explicit reason; everything
else in the suite runs on synthetic data or shipped fixtures.
"""
import os
from pathlib import Path

import pytest

AQTE_TESTS = {
    "test_main_aqte_dimensions_and_date_split",
    "test_frozen_aqte_prescreen_reproduces_original_144_candidate_pairs",
    "test_training_control_uses_same_windows_and_stays_inside_training",
    "test_frozen_fit_adapter_reproduces_original_backtest",
}


def aqte_skip_reason():
    directory = os.environ.get("AQTE_DATA_DIR")
    if not directory:
        return "AQTE_DATA_DIR is not set; this test requires the external AQTE price panel."
    path = Path(directory).expanduser() / "SP500_Data_Sample_Prices.csv"
    if not path.is_file():
        return f"AQTE source price file does not exist: {path}"
    return None


def pytest_runtest_setup(item):
    if item.originalname in AQTE_TESTS:
        reason = aqte_skip_reason()
        if reason:
            pytest.skip(reason)


@pytest.fixture
def aqte_source_path():
    reason = aqte_skip_reason()
    if reason:
        pytest.skip(reason)
    return Path(os.environ["AQTE_DATA_DIR"]).expanduser() / "SP500_Data_Sample_Prices.csv"
