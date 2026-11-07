# Synthetic smoke test only

All five assets in this directory are generated artificial series, including those named AAPL and ADBE. They are not market observations and do not support the 226-equity / 846-day headline numbers. Reproduce the workflow with `python cointegration_stat_arb/pipeline.py` from the project root. The full verification command also runs this entry point to guard against the former `round_trip_entries` KeyError. AQTE main evidence is in `../results/`.
