import polars as pl

from src.utils.market import get_prices


def rsi(close: pl.Series, window: int = 14) -> pl.Series:
    delta = close.diff()
    gain = delta.clip(lower_bound=0.0)
    loss = (-delta).clip(lower_bound=0.0)
    avg_gain = gain.rolling_mean(window)
    avg_loss = loss.rolling_mean(window)
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def momentum(close: pl.Series, window: int = 30) -> pl.Series:
    shifted = close.shift(window)
    return (close - shifted) / shifted


def volume_zscore(volume: pl.Series, window: int = 20) -> pl.Series:
    mean = volume.rolling_mean(window)
    std = volume.rolling_std(window)
    return (volume - mean) / std


def macd_signal(close: pl.Series) -> pl.Series:
    macd = close.ewm_mean(span=12) - close.ewm_mean(span=26)
    return macd.ewm_mean(span=9)


def bollinger_width(close: pl.Series, window: int = 20) -> pl.Series:
    mean = close.rolling_mean(window)
    std = close.rolling_std(window)
    return (4 * std) / mean


def build_feature_matrix(tickers: list[str]) -> pl.DataFrame:
    """One row per ticker with the most recent feature values. Drops tickers with insufficient history."""
    prices = get_prices(tickers, period="6mo")

    rows: list[dict] = []
    for ticker in tickers:
        sub = prices.filter(pl.col("ticker") == ticker).sort("date")
        close = sub["close"]
        vol = sub["volume"]

        rows.append({
            "ticker": ticker,
            "rsi_14": rsi(close)[-1],
            "momentum_30": momentum(close)[-1],
            "volume_zscore_20": volume_zscore(vol)[-1],
            "macd_signal": macd_signal(close)[-1],
            "bollinger_width_20": bollinger_width(close)[-1],
        })

    return pl.DataFrame(rows).drop_nulls()
