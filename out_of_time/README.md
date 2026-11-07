# 2026 out-of-time revalidation

**Bottom line: the original AJG/AON pair keeps failing; the original selection process, rerun on this new holdout, still finds positive returns.** Both findings stand on their own. The second is not a reason to keep trading AJG/AON, and the first is not proof that the screening process is dead for good. No mode or parameter was chosen using holdout returns.

Data were downloaded through a locally authenticated LSEG Workspace Python API session and cover **2021-03-31 → 2026-10-28, 1,402 trading days**. Field **`TRDPRC_1`** with capital-change adjustments explicitly set to `CCH/CRE/RPO/RTS/exchangeCorrection/manualCorrection` — a price-close series, not dividend-reinvested total returns. This matches the [LSEG adjustment guide](https://developers.lseg.com/en/article-catalog/article/workspace-corporate-actions-content-set-guide) and the original panel's price definition (split-adjusted, not total-return). The source file has no full vendor metadata, so security identity was additionally checked against unadjusted closes on 2021-03-26/29/30, and the AJG/AON splice-point price units were checked separately. Cross-vendor historical revisions remain a limitation.

The original universe of **226 source columns / 225 unique ticker strings** is fully retained: 221 columns have new prices and the original cleaning rules keep 219. Five source codes could not be verified; twenty now-inactive quotes keep history up to their stop dates. The duplicate ALXN column disagrees on price, SPRT does not match its true quote, and FBSH/BCA/HTL have no US common-equity match — none were guessed or replaced. The full list with reasons is in [missing_or_inactive.csv](data/missing_or_inactive.csv); rename mappings, raw responses and all SHA256 checksums are in [data/manifest.json](data/manifest.json). The raw LSEG files themselves are not redistributable — re-fetch them with `fetch_lseg.py` against your own session.

| Check | Mode | Gross | Net | Sharpe | Max drawdown |
| --- | --- | --- | --- | --- | --- |
| Original AJG/AON | static | −7.1497% | −10.5535% | −0.2624 | −14.9254% |
| Original AJG/AON | walk_forward | −0.3822% | −4.3348% | −0.0642 | −17.1570% |
| New pick CMS/DTE | static | +9.0604% | +7.1281% | 1.0413 | −3.0910% |
| New pick CMS/DTE | walk_forward | +3.5268% | +1.9957% | 0.2922 | −3.2064% |

The original pair keeps its **2018-01-02–2019-12-11** α = 0.0171732638794202, β = 0.8503639747493976. Earlier history serves only as warm-up for signals and rolling estimates; positions reset to flat on 2021-03-31 before the original functions take over. That's 1,401 return periods. Full static/walk-forward metrics, cost decomposition, daily equity, trade tables and rolling diagnostics: [fixed_pair/](results/fixed_pair/). Rolling β spans **−0.34 to 2.85**; only **4.39%** of 1,277 rolling 126-day windows reject no-cointegration (last p = 0.8928).

The new window keeps the same 60/40 split: 841 training prices, 561 holdout prices / 560 return periods, boundary 2024-08-05. Training-only 0.70 prescreen and EG tests: **185 pairs tested, 14 raw 5% passes, 1 BH pass**. The original t-stat rule selects CMS/DTE. Both hedge modes are reported; the higher-returning one was not picked after seeing results. See [new_period_refit/](results/new_period_refit/).

**Evidence:** the original pair's static gross is already −7.1497% — dead before costs. Walk-forward gross −0.3822% becomes net −4.3348% once costs take another ~3.95 points. The β swings support an unstable relationship; weak gross and costs together explain the failure. The training-period control rejects in 25.92% of 126-day windows versus 0% out-of-sample and 4.39% this round. Short windows often lack power, so non-rejections cannot all be read as relationship breakdown, and these overlapping windows cannot split "low power" from "relationship change" into percentages.

**Speculation:** market or sector regimes may have shifted the relationship, but this round adds no factors and no causal test, so no specific market cause is identified. The new pair's positive net comes with a walk-forward Sharpe of only 0.2922 and is still one holdout. Inactive securities are not stitched to acquirer stock and no delisting cash flows are invented. The original cost and price-backtest limitations carry over.

Re-run offline from the project root: `python out_of_time/run_revalidation.py` (requires `AQTE_DATA_DIR`). Re-download with `fetch_lseg.py` in the same folder — the desktop session must be logged in, and the download environment installs `requirements-fetch.txt` separately. Checks: `python -m pytest out_of_time/`. Main code, main results and existing tests are unchanged.
