import os
import random
import re
from collections import Counter

import polars as pl

from src.profiles.base import BaseProfile
from src.utils.market import get_current_price


class RedditRita(BaseProfile):
    @property
    def name(self) -> str:
        return "Reddit Rita"

    @property
    def strategy(self) -> str:
        return "wsb_sentiment"

    def pick(self, universe: list[str]) -> pl.DataFrame:
        tickers, reasons = self._wsb_picks(universe)
        return pl.DataFrame({
            "ticker": tickers,
            "price_entry": [get_current_price(t) for t in tickers],
            "reason": reasons,
        })

    def _wsb_picks(self, universe: list[str]) -> tuple[list[str], list[str]]:
        client_id = os.getenv("REDDIT_CLIENT_ID")
        client_secret = os.getenv("REDDIT_CLIENT_SECRET")

        if not client_id or not client_secret:
            return random.sample(universe, 3), ["fallback: no Reddit creds"] * 3

        try:
            import praw

            reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent="stock-tournament/1.0",
            )

            universe_set = set(universe)
            counts: Counter = Counter()

            for post in reddit.subreddit("wallstreetbets").hot(limit=50):
                text = f"{post.title} {post.selftext}"
                for candidate in re.findall(r"\b[A-Z]{1,5}\b", text):
                    if candidate in universe_set:
                        counts[candidate] += 1

            top = counts.most_common(3)
            tickers = [t for t, _ in top]
            reasons = [f"WSB mentions: {c}" for _, c in top]

            # fill with random if fewer than 3 tickers found
            if len(tickers) < 3:
                remaining = [t for t in universe if t not in tickers]
                fill = random.sample(remaining, 3 - len(tickers))
                tickers += fill
                reasons += ["fallback: low WSB mentions"] * len(fill)

            return tickers, reasons

        except Exception as e:
            print(f"[RedditRita] falling back to random: {type(e).__name__}: {e}")
            return random.sample(universe, 3), [f"fallback: {type(e).__name__}"] * 3
