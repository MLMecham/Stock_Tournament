from abc import ABC, abstractmethod

import polars as pl


class BaseProfile(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def strategy(self) -> str: ...

    @abstractmethod
    def pick(self, universe: list[str]) -> pl.DataFrame:
        """Return DataFrame with columns: ticker (str), price_entry (float), reason (str)."""
        ...
