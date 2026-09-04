"""
Market data loading.

Daily bars come from Yahoo Finance via yfinance. It is free and good enough
for daily swing trading research. It is not good enough for intraday work and
it will occasionally hand you a bad print, which is why load_bars sanity
checks everything before returning it.

Every download is cached to data/cache as CSV so repeated backtests do not
hammer the API and so you can work offline.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_COLS = ["open", "high", "low", "close", "volume"]

# How far cached prices may sit from a fresh download of the same dates before
# the cache is treated as being on a different basis. Rounding moves prices by
# a hundredth of a percent. A split moves them by half or more.
ADJUST_TOLERANCE = 0.002


class DataError(RuntimeError):
    pass


def _cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol.upper()}.csv"


def _read_cache(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return _normalize(pd.read_csv(path, index_col=0, parse_dates=True))
    except Exception:
        return None    # an unreadable cache is not worth failing the run over


def _adjustment_drift(old: pd.DataFrame, fresh: pd.DataFrame):
    """How far apart two frames are on the dates they both cover.

    Returns (largest relative difference, number of shared bars). Zero shared
    bars means the question cannot be answered, which is itself a reason not
    to merge the two halves together.
    """
    shared = old.index.intersection(fresh.index)
    if len(shared) == 0:
        return 0.0, 0
    a = old.loc[shared, "close"].astype(float)
    b = fresh.loc[shared, "close"].astype(float)
    denom = b.abs().clip(lower=1e-9)
    return float(((a - b).abs() / denom).max()), int(len(shared))


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Force whatever the source hands back into a clean lowercase OHLCV frame."""
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance returns a MultiIndex when given a list of tickers.
        df = df.droplevel(1, axis=1)

    df = df.rename(columns={c: str(c).strip().lower().replace(" ", "_") for c in df.columns})

    if "adj_close" in df.columns and "close" not in df.columns:
        df["close"] = df["adj_close"]

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise DataError(f"data source is missing columns: {missing}")

    df = df[REQUIRED_COLS].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _sanity_check(symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that cannot be real. Bad ticks make fake backtest profits."""
    n_before = len(df)

    df = df.dropna(subset=REQUIRED_COLS)
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    df = df[df["volume"] >= 0]
    # High must be the highest and low the lowest of the bar.
    df = df[df["high"] >= df[["open", "close", "low"]].max(axis=1) - 1e-9]
    df = df[df["low"] <= df[["open", "close", "high"]].min(axis=1) + 1e-9]

    dropped = n_before - len(df)
    if dropped > 0:
        print(f"  [{symbol}] dropped {dropped} malformed bar(s)")
    return df


def download_bars(symbol: str, start: str, end: Optional[str] = None,
                  retries: int = 3) -> pd.DataFrame:
    """Fetch daily bars from Yahoo. Requires network access."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise DataError(
            "yfinance is not installed. Run: pip install -r requirements.txt"
        ) from exc

    last_err = None
    for attempt in range(retries):
        try:
            raw = yf.download(
                symbol,
                start=start,
                end=end,
                progress=False,
                auto_adjust=True,   # split and dividend adjusted, which is what
                                    # you want for a multi-year backtest
                threads=False,
            )
            if raw is not None and len(raw) > 0:
                return _normalize(raw)
            last_err = DataError(f"empty response for {symbol}")
        except Exception as exc:  # network flake, rate limit, schema change
            last_err = exc
        time.sleep(1.5 * (attempt + 1))

    raise DataError(f"could not download {symbol}: {last_err}")


def download_many(symbols: List[str], start: str, end: Optional[str] = None,
                  chunk: int = 40) -> Dict[str, pd.DataFrame]:
    """Fetch several symbols per request instead of one at a time.

    A watchlist of 200 names is 200 separate HTTP requests the naive way, which
    is slow and, from a shared address like a CI runner, a good way to get rate
    limited into a half-empty result. Yahoo will return a batch in one call, so
    this asks in chunks and splits the frame afterwards.

    Anything the batch does not return is retried on its own, because one bad
    ticker in a chunk should not cost you the other thirty-nine.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise DataError(
            "yfinance is not installed. Run: pip install -r requirements.txt"
        ) from exc

    out: Dict[str, pd.DataFrame] = {}
    todo = [s.upper() for s in symbols]

    for i in range(0, len(todo), chunk):
        group = todo[i:i + chunk]
        try:
            raw = yf.download(group, start=start, end=end, progress=False,
                              auto_adjust=True, threads=False, group_by="column")
        except Exception:
            continue
        if raw is None or len(raw) == 0:
            continue
        if not isinstance(raw.columns, pd.MultiIndex):
            # Yahoo flattens the columns when only one symbol comes back.
            if len(group) == 1:
                try:
                    out[group[0]] = _normalize(raw)
                except DataError:
                    pass
            continue
        for sym in group:
            try:
                one = raw.xs(sym, axis=1, level=1).dropna(how="all")
            except (KeyError, IndexError):
                continue
            if len(one) == 0:
                continue
            try:
                out[sym] = _normalize(one)
            except DataError:
                pass

    # Whatever the batch missed, ask for individually before giving up on it.
    for sym in todo:
        if sym in out:
            continue
        try:
            out[sym] = download_bars(sym, start=start, end=end, retries=2)
        except Exception:
            pass
    return out


def load_bars(symbol: str, start: str = "2015-01-01", end: Optional[str] = None,
              use_cache: bool = True, refresh: bool = False,
              fresh: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Load daily bars, preferring the local cache.

    Set refresh=True to force a re-download (do this before live scanning so
    you are not trading off stale bars).
    """
    symbol = symbol.upper()
    path = _cache_path(symbol)

    if use_cache and path.exists() and not refresh and fresh is None:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df = _normalize(df)
    else:
        # `fresh` lets a caller hand over bars it already fetched in a batch.
        # Everything below, including the re-adjustment check, is identical
        # either way, so batching cannot quietly skip a safety step.
        if fresh is None:
            fresh = download_bars(symbol, start=start, end=end)

        # Merge into whatever is already cached instead of replacing it.
        # A daily run only asks for a couple of years of bars, and without
        # this merge each run would quietly delete the older history, so a
        # backtest run tomorrow would silently cover a shorter period than
        # the same command covered today. Data disappearing without an error
        # is the worst kind of bug: nothing breaks, the answers just change.
        #
        # Merging is only safe while both halves are quoted on the same basis.
        # These bars are split and dividend adjusted, so every split and every
        # dividend rewrites the entire history at the source. Gluing yesterday's
        # cache onto today's download after a 10-for-1 split would leave a 90%
        # cliff in the middle of the series, and nothing downstream would call
        # that an error. The moving averages would simply be wrong, the trend
        # filter would read a crash, and the agent would act on it. So the two
        # halves are compared where they overlap before they are joined.
        old = _read_cache(path)
        if old is not None and len(old):
            drift, shared = _adjustment_drift(old, fresh)
            if shared == 0:
                reason = "the cache and the download share no dates"
            elif drift > ADJUST_TOLERANCE:
                reason = (f"prices differ by {drift * 100:.1f}% on {shared} "
                          f"shared bar(s), the source has re-adjusted")
            else:
                reason = ""

            if reason:
                # Do not repair the seam, refuse to create one. Re-download the
                # whole history so every bar comes from one adjustment basis.
                print(f"  [{symbol}] rebuilding cache: {reason}")
                floor = str(old.index.min().date())
                fresh = download_bars(symbol,
                                      start=min(floor, start or floor), end=end)
            else:
                fresh = pd.concat([old, fresh])
                fresh = fresh[~fresh.index.duplicated(keep="last")].sort_index()

        df = fresh
        df.to_csv(path)

    df = _sanity_check(symbol, df)

    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end)]

    if df.empty:
        raise DataError(f"no usable bars for {symbol} in the requested range")
    return df


def load_universe(symbols: List[str], start: str = "2015-01-01",
                  end: Optional[str] = None, refresh: bool = False
                  ) -> Dict[str, pd.DataFrame]:
    """Load bars for a whole watchlist. Symbols that fail are skipped loudly."""
    out: Dict[str, pd.DataFrame] = {}

    prefetched: Dict[str, pd.DataFrame] = {}
    if refresh and len(symbols) > 1:
        prefetched = download_many(symbols, start=start, end=end)
        missed = [s for s in symbols if s.upper() not in prefetched]
        if missed:
            print(f"  {len(missed)} symbol(s) returned nothing: "
                  f"{', '.join(missed[:10])}")

    for sym in symbols:
        try:
            out[sym] = load_bars(sym, start=start, end=end, refresh=refresh,
                                 fresh=prefetched.get(sym.upper()))
        except Exception as exc:
            print(f"  [{sym}] SKIPPED: {exc}")
    if not out:
        raise DataError(
            "no symbols loaded. If every symbol failed, you have no network "
            "access to Yahoo Finance from this machine."
        )
    return out


def cache_status() -> pd.DataFrame:
    """What is in the cache and how stale it is."""
    rows = []
    for path in sorted(CACHE_DIR.glob("*.csv")):
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            rows.append({
                "symbol": path.stem,
                "bars": len(df),
                "first": df.index.min().date() if len(df) else None,
                "last": df.index.max().date() if len(df) else None,
                "size_kb": round(os.path.getsize(path) / 1024, 1),
            })
        except Exception:
            rows.append({"symbol": path.stem, "bars": 0, "first": None,
                         "last": None, "size_kb": 0})
    return pd.DataFrame(rows)
