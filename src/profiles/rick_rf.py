import random
from pathlib import Path

import polars as pl

from src.profiles.base import BaseProfile
from src.utils.features import build_feature_matrix
from src.utils.market import get_current_price
from src.utils.universe import UNIVERSE

_MODEL_PATH = Path("model/random_forest.pkl")
_SCALER_PATH = Path("model/scaler.pkl")


class RickRF(BaseProfile):
    @property
    def name(self) -> str:
        return "Rick RF"

    @property
    def strategy(self) -> str:
        return "random_forest"

    def pick(self, universe: list[str]) -> pl.DataFrame:
        if not _MODEL_PATH.exists():
            tickers = random.sample(universe, 3)
            return pl.DataFrame({
                "ticker": tickers,
                "price_entry": [get_current_price(t) for t in tickers],
                "reason": ["cold start — no model yet"] * 3,
            })

        import joblib
        model = joblib.load(_MODEL_PATH)
        scaler = joblib.load(_SCALER_PATH)

        features = build_feature_matrix(universe).drop_nulls()
        feature_cols = [c for c in features.columns if c != "ticker"]

        X = scaler.transform(features.select(feature_cols).to_numpy())
        probs = model.predict_proba(X)[:, 1]

        top3 = (
            features.with_columns(pl.Series("prob", probs))
            .sort("prob", descending=True)
            .head(3)
        )

        tickers = top3["ticker"].to_list()
        return pl.DataFrame({
            "ticker": tickers,
            "price_entry": [get_current_price(t) for t in tickers],
            "reason": [f"RF confidence: {p:.1%}" for p in top3["prob"].to_list()],
        })
