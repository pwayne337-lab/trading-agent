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


class DataError(RuntimeError):
    pass


def _cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol.upper()}.csv"


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


def load_bars(symbol: str, start: str = "2015-01-01", end: Optional[str] = None,
              use_cache: bool = True, refresh: bool = False) -> pd.DataFrame:
    """Load daily bars, preferring the local cache.

    Set refresh=True to force a re-download (do this before live scanning so
    you are not trading off stale bars).
    """
    symbol = symbol.upper()
    path = _cache_path(symbol)

    if use_cache and path.exists() and not refresh:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df = _normalize(df)
    else:
        fresh = download_bars(symbol, start=start, end=end)

        # Merge into whatever is already cached instead of replacing it.
        # A daily run only asks for a couple of years of bars, and without
        # this merge each run would quietly delete the older history, so a
        # backtest run tomorrow would silently cover a shorter period than
        # the same command covered today. Data disappearing without an error
        # is the worst kind of bug: nothing breaks, the answers just change.
        if path.exists():
            try:
                old = _normalize(pd.read_csv(path, index_col=0, parse_dates=True))
                fresh = pd.concat([old, fresh])
                fresh = fresh[~fresh.index.duplicated(keep="last")].sort_index()
            except Exception:
                pass   # unreadable cache is not worth failing the run over

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
    for sym in symbols:
        try:
            out[sym] = load_bars(sym, start=start, end=end, refresh=refresh)
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
