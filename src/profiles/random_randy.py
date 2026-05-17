import random

import polars as pl

from src.profiles.base import BaseProfile
from src.utils.market import get_current_price


class RandomRandy(BaseProfile):
    @property
    def name(self) -> str:
        return "Random Randy"

    @property
    def strategy(self) -> str:
        return "random"

    def pick(self, universe: list[str]) -> pl.DataFrame:
        tickers = random.sample(universe, 3)
        return pl.DataFrame({
            "ticker": tickers,
            "price_entry": [get_current_price(t) for t in tickers],
            "reason": ["random pick"] * 3,
        })
