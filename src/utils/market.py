import yfinance as yf
import polars as pl


def get_prices(tickers: list[str], period: str) -> pl.DataFrame:
    """OHLCV data for one or more tickers. Returns long-format Polars DataFrame."""
    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)

    frames: list[pl.DataFrame] = []
    for ticker in tickers:
        sub = (
            raw.xs(ticker, axis=1, level=1)[["Open", "High", "Low", "Close", "Volume"]]
            if len(tickers) > 1
            else raw[["Open", "High", "Low", "Close", "Volume"]]
        )
        frames.append(
            pl.from_pandas(sub.reset_index())
            .rename({"Date": "date", "Open": "open", "High": "high",
                     "Low": "low", "Close": "close", "Volume": "volume"})
            .with_columns(pl.lit(ticker).alias("ticker"))
        )

    return pl.concat(frames)


def get_current_price(ticker: str) -> float:
    """Last traded price for a single ticker."""
    return float(yf.Ticker(ticker).fast_info["lastPrice"])


def get_sp500_return(start_date: str) -> float:
    """SPY total return from start_date (YYYY-MM-DD) to today."""
    raw = yf.download("SPY", start=start_date, auto_adjust=True, progress=False)
    close = raw["Close"]
    return float((close.iloc[-1] - close.iloc[0]) / close.iloc[0])
