import polars as pl

from src.profiles.base import BaseProfile
from src.utils.market import get_current_price

_PICKS = ["MSFT", "JNJ", "KO"]


class BoomerBob(BaseProfile):
    @property
    def name(self) -> str:
        return "Boomer Bob"

    @property
    def strategy(self) -> str:
        return "blue_chip"

    def pick(self, universe: list[str]) -> pl.DataFrame:
        return pl.DataFrame({
            "ticker": _PICKS,
            "price_entry": [get_current_price(t) for t in _PICKS],
            "reason": ["blue-chip staple"] * 3,
        })
