import json
import random

import anthropic
import polars as pl

from src.profiles.base import BaseProfile
from src.utils.headlines import get_headlines
from src.utils.market import get_current_price

_MODEL = "claude-haiku-4-5-20251001"


class ClaudeProfile(BaseProfile):
    @property
    def name(self) -> str:
        return "Claude"

    @property
    def strategy(self) -> str:
        return "llm_news"

    def pick(self, universe: list[str]) -> pl.DataFrame:
        tickers, reasons = self._claude_picks(universe)
        return pl.DataFrame({
            "ticker": tickers,
            "price_entry": [get_current_price(t) for t in tickers],
            "reason": reasons,
        })

    def _claude_picks(self, universe: list[str]) -> tuple[list[str], list[str]]:
        try:
            headlines = get_headlines()
            client = anthropic.Anthropic()

            response = client.messages.create(
                model=_MODEL,
                max_tokens=256,
                system=(
                    f"You are a stock picker. Your universe is: {', '.join(universe)}. "
                    "Only pick tickers from this list. "
                    "Respond with ONLY a JSON array, no other text: "
                    '[{"ticker": "X", "reason": "..."}]'
                ),
                messages=[{
                    "role": "user",
                    "content": (
                        f"Today's financial headlines:\n{headlines}\n\n"
                        "Pick exactly 3 stocks for the next week."
                    ),
                }],
            )

            picks = json.loads(response.content[0].text)

            universe_set = set(universe)
            picks = [p for p in picks if p.get("ticker") in universe_set][:3]

            if len(picks) < 3:
                raise ValueError("fewer than 3 valid tickers returned")

            return [p["ticker"] for p in picks], [p["reason"] for p in picks]

        except Exception as e:
            print(f"[ClaudeProfile] falling back to random: {type(e).__name__}: {e}")
            return random.sample(universe, 3), [f"fallback: {type(e).__name__}"] * 3
