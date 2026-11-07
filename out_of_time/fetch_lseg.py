"""Resolve original AQTE identifiers and fetch matching LSEG daily price history.

Local desktop only. No ChatGPT connector, substitute securities, dividend-total-
return prices or forward filling. Every mapping is checked against three raw
2021 closing prices before it can enter the panel.
"""

from __future__ import annotations
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import hashlib, json, os, re, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(ROOT.parent))
from pipeline import load_wide_prices
from run_main import data_path

ADJUSTMENTS = ["exchangeCorrection", "manualCorrection", "CCH", "CRE", "RPO", "RTS"]
SELECT = "RIC,DocumentTitle,TickerSymbol,USStockTicker,OldQuoteTickers,AssetState,IsPrimaryRIC,ExchangeName,IssuerOAPermID"


def normal_columns(frame, universe=None):
    f = frame.copy()
    if isinstance(f.columns, pd.MultiIndex):
        f.columns = [next(str(v) for v in c if str(v) != "TRDPRC_1") for c in f.columns]
    if universe is not None and len(universe) == 1 and list(f.columns) == ["TRDPRC_1"]:
        f.columns = list(universe)
    f.index = pd.to_datetime(f.index).normalize()
    return f


def candidate_rank(row):
    exchange = row.get("ExchangeName", "")
    if exchange == "New York Stock Exchange":
        score = 0
    elif "Consolidated" in exchange and "NASDAQ" in exchange:
        score = 1
    elif exchange == "NYSE Consolidated":
        score = 2
    elif "NASDAQ Stock Exchange" in exchange:
        score = 3
    else:
        score = 10
    return (
        row.get("AssetState") != "AC",
        score,
        not bool(row.get("IsPrimaryRIC")),
        str(row["RIC"]),
    )


def main():
    import lseg.data as ld
    import importlib.metadata

    data = ROOT / "data"
    data.mkdir(parents=True, exist_ok=True)
    lookups = data / "lookups"
    lookups.mkdir(exist_ok=True)
    source = load_wide_prices(data_path())
    symbols = list(dict.fromkeys(c.split("_LAST")[0] for c in source.columns))
    now = datetime.now(ZoneInfo("America/New_York"))
    cutoff = now.date() if now.hour >= 16 else now.date() - timedelta(days=1)
    ld.open_session(name="desktop.workspace")
    try:
        candidates = {}
        for i, symbol in enumerate(symbols):
            cache = lookups / (symbol.replace(".", "_") + ".json")
            if cache.exists():
                rows = json.loads(cache.read_text())
            else:
                filt = f"(TickerSymbol eq '{symbol}' or OldQuoteTickers eq '{symbol}' or USStockTicker eq '{symbol}') and ExchangeCountry eq 'USA'"
                try:
                    f = ld.discovery.search(
                        view=ld.discovery.Views.EQUITY_QUOTES,
                        filter=filt,
                        select=SELECT,
                        top=200,
                    )
                    rows = (
                        json.loads(f.to_json(orient="records"))
                        if f is not None and not f.empty
                        else []
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Identifier lookup failed for {symbol}: {type(e).__name__}: {e}"
                    ) from e
                cache.write_text(json.dumps(rows, indent=2) + "\n")
                time.sleep(0.12)

            def linked_to_original(r):
                normalize = (
                    lambda v: str(v or "").replace(".", "").replace(" ", "").upper()
                )
                wanted = normalize(symbol)
                direct = [r.get("TickerSymbol"), r.get("USStockTicker")]
                historical = " ".join(r.get("OldQuoteTickers") or []).split()
                return any(normalize(v) == wanted for v in direct + historical)

            eligible = [
                r
                for r in rows
                if "Ordinary Share" in r.get("DocumentTitle", "")
                and linked_to_original(r)
            ]
            # At most three primary-listing candidates; no universe replacements.
            candidates[symbol] = sorted(eligible, key=candidate_rank)[:3]
            if (i + 1) % 20 == 0:
                print(f"Identifier lookup {i+1}/{len(symbols)}", flush=True)
        rics = list(
            dict.fromkeys(r["RIC"] for rows in candidates.values() for r in rows)
        )
        anchor_file = data / "identity_anchor_raw_close.csv"
        if anchor_file.exists():
            anchors = pd.read_csv(anchor_file, index_col=0, parse_dates=True)
        else:
            frames = []
            for start in range(0, len(rics), 30):
                batch = rics[start : start + 30]
                try:
                    f = normal_columns(
                        ld.get_history(
                            batch,
                            fields=["TRDPRC_1"],
                            interval="1D",
                            start="2021-03-26",
                            end="2021-03-30",
                            adjustments=["unadjusted"],
                        ),
                        batch,
                    )
                    frames.append(f)
                except Exception as e:
                    raise RuntimeError(f"Identity anchor request failed: {e}") from e
            anchors = pd.concat(frames, axis=1)
            anchors = anchors.loc[:, ~anchors.columns.duplicated()]
            anchors.to_csv(anchor_file, index_label="date")
        missing_rics = [ric for ric in rics if ric not in anchors.columns]
        for start in range(0, len(missing_rics), 30):
            f = normal_columns(
                ld.get_history(
                    missing_rics[start : start + 30],
                    fields=["TRDPRC_1"],
                    interval="1D",
                    start="2021-03-26",
                    end="2021-03-30",
                    adjustments=["unadjusted"],
                ),
                missing_rics[start : start + 30],
            )
            anchors = pd.concat([anchors, f], axis=1)
        anchors = anchors.loc[:, ~anchors.columns.duplicated()]
        anchors.to_csv(anchor_file, index_label="date")
        mappings = []
        for column in source.columns:
            symbol = column.split("_LAST")[0]
            record = {
                "source_column": column,
                "original_ticker": symbol,
                "ric": None,
                "reason": None,
            }
            for c in candidates[symbol]:
                if c["RIC"] not in anchors:
                    continue
                match = pd.concat(
                    [source[column], anchors[c["RIC"]]], axis=1, join="inner"
                ).dropna()
                if len(match) >= 2 and np.allclose(
                    match.iloc[:, 0].to_numpy(dtype=float),
                    match.iloc[:, 1].to_numpy(dtype=float),
                    atol=0.011,
                    rtol=0,
                ):
                    record.update(
                        ric=c["RIC"],
                        issuer=c.get("IssuerOAPermID"),
                        company=c["DocumentTitle"],
                        asset_state=c.get("AssetState"),
                        current_ticker=c.get("TickerSymbol"),
                        anchor_observations=len(match),
                        anchor_max_abs_error=float(
                            (match.iloc[:, 0] - match.iloc[:, 1]).abs().max()
                        ),
                        reason="same_security_verified_by_historical_raw_closes",
                    )
                    break
            if record["ric"] is None:
                record["reason"] = (
                    "no_US_ordinary_equity_identifier_match"
                    if not candidates[symbol]
                    else "identity_not_verified_against_AQTE_closes"
                )
                if ".1" in column:
                    record["reason"] = (
                        "duplicate_ALXN_header_contains_different_prices_no_substitution"
                    )
            mappings.append(record)
        pd.DataFrame(mappings).to_csv(data / "identifier_mapping.csv", index=False)
        (data / "identifier_mapping.json").write_text(
            json.dumps(mappings, indent=2) + "\n"
        )
        resolved = list(dict.fromkeys(r["ric"] for r in mappings if r["ric"]))
        rawdir = data / "raw_batches"
        rawdir.mkdir(exist_ok=True)
        histories = []
        for start in range(0, len(resolved), 10):
            batch = resolved[start : start + 10]
            signature = hashlib.sha256("|".join(batch).encode()).hexdigest()[:12]
            cache = rawdir / f"batch_{start//10:02d}_{signature}_{cutoff}.csv"
            legacy = rawdir / f"batch_{start//10:02d}_{cutoff}.csv"
            if not cache.exists() and legacy.exists():
                probe = pd.read_csv(legacy, index_col=0, parse_dates=True)
                if set(batch) == set(probe.columns):
                    legacy.rename(cache)
            if cache.exists():
                f = pd.read_csv(cache, index_col=0, parse_dates=True)
            else:
                f = normal_columns(
                    ld.get_history(
                        batch,
                        fields=["TRDPRC_1"],
                        interval="1D",
                        start="2021-03-30",
                        end=str(cutoff),
                        adjustments=ADJUSTMENTS,
                    ),
                    batch,
                )
                f.to_csv(cache, index_label="date")
            if len(batch) == 1 and list(f.columns) == ["TRDPRC_1"]:
                f.columns = batch
                f.to_csv(cache, index_label="date")
            histories.append(f)
            print(
                f"Price history {min(start+10,len(resolved))}/{len(resolved)} securities",
                flush=True,
            )
        history = pd.concat(histories, axis=1)
        history = history.loc[:, ~history.columns.duplicated()].sort_index()
        # Keep all 226 source columns, even those that remain entirely missing.
        panel = pd.DataFrame(index=history.index)
        for record in mappings:
            panel[record["source_column"]] = (
                history[record["ric"]] if record["ric"] in history else np.nan
            )
        panel = panel.loc[panel.index >= pd.Timestamp("2021-03-31")].dropna(how="all")
        panel.to_csv(data / "aqte_universe_lseg_close.csv", index_label="date")
        coverage = []
        last = panel.index[-1]
        for record in mappings:
            series = panel[record["source_column"]].dropna()
            row = {
                **record,
                "observations": len(series),
                "first_price_date": (
                    str(series.index[0].date()) if len(series) else None
                ),
                "last_price_date": (
                    str(series.index[-1].date()) if len(series) else None
                ),
            }
            if len(series) and series.index[-1] < last:
                row["coverage_reason"] = (
                    "inactive_or_delisted_historical_quote"
                    if row.get("asset_state") == "DC"
                    else "history_ends_before_latest_session"
                )
            elif not len(series):
                row["coverage_reason"] = (
                    record["reason"]
                    if record["ric"] is None
                    else "no_history_returned_for_original_security"
                )
            else:
                row["coverage_reason"] = "available_through_latest_session"
            coverage.append(row)
        pd.DataFrame(coverage).to_csv(data / "coverage.csv", index=False)
        pd.DataFrame(
            [
                r
                for r in coverage
                if r["coverage_reason"] != "available_through_latest_session"
            ]
        ).to_csv(data / "missing_or_inactive.csv", index=False)
        history.loc[history.index == pd.Timestamp("2021-03-30")].to_csv(
            data / "adjusted_overlap_anchor.csv", index_label="date"
        )
        manifest = {
            "provider": "LSEG Workspace local desktop Python API",
            "sdk_version": importlib.metadata.version("lseg-data"),
            "session": "desktop.workspace",
            "field": "TRDPRC_1",
            "adjustments": ADJUSTMENTS,
            "price_definition": "Daily closing prices adjusted for capital changes, not dividend-reinvested total returns; raw overlap closes verify security identity.",
            "start": "2021-03-31",
            "requested_end": str(cutoff),
            "last_available_session": str(last.date()),
            "retrieved_at": now.isoformat(),
            "source_columns": len(source.columns),
            "unique_source_ticker_strings": len(symbols),
            "identity_verified_columns": sum(r["ric"] is not None for r in mappings),
            "columns_with_prices": int(panel.notna().any().sum()),
            "price_observations": len(panel),
            "files": {
                str(p.relative_to(data)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in data.rglob("*")
                if p.is_file() and p.name != "manifest.json"
            },
        }
        (data / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(
            json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2),
            flush=True,
        )
    finally:
        ld.close_session()


if __name__ == "__main__":
    main()
