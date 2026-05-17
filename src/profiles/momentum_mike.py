import polars as pl

from src.profiles.base import BaseProfile
from src.utils.market import get_current_price, get_prices


class MomentumMike(BaseProfile):
    @property
    def name(self) -> str:
        return "Momentum Mike"

    @property
    def strategy(self) -> str:
        return "momentum"

    def pick(self, universe: list[str]) -> pl.DataFrame:
        prices = get_prices(universe, period="1mo")

        top3 = (
            prices.group_by("ticker")
            .agg([
                pl.col("close").sort_by(pl.col("date")).first().alias("start_price"),
                pl.col("close").sort_by(pl.col("date")).last().alias("end_price"),
            ])
            .with_columns(
                ((pl.col("end_price") - pl.col("start_price")) / pl.col("start_price"))
                .alias("return_30d")
            )
            .sort("return_30d", descending=True)
            .head(3)
        )

        tickers = top3["ticker"].to_list()
        returns = top3["return_30d"].to_list()
        return pl.DataFrame({
            "ticker": tickers,
            "price_entry": [get_current_price(t) for t in tickers],
            "reason": [f"30d return {r:+.1%}" for r in returns],
        })
