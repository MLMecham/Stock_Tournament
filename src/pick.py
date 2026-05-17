import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from src.profiles.boomer_bob import BoomerBob
from src.profiles.claude_profile import ClaudeProfile
from src.profiles.momentum_mike import MomentumMike
from src.profiles.random_randy import RandomRandy
from src.profiles.reddit_rita import RedditRita
from src.profiles.rick_rf import RickRF
from src.utils.universe import UNIVERSE

PROFILES = [
    RandomRandy(),
    BoomerBob(),
    MomentumMike(),
    RedditRita(),
    ClaudeProfile(),
    RickRF(),
]


def main() -> None:
    today = datetime.now(timezone.utc).date()
    iso = today.isocalendar()
    year, week = iso.year, iso.week

    frames: list[pl.DataFrame] = []
    for profile in PROFILES:
        try:
            df = profile.pick(UNIVERSE).with_columns([
                pl.lit(profile.name).alias("profile_name"),
                pl.lit(str(today)).alias("pick_date"),
                pl.lit(week).alias("week"),
                pl.lit(year).alias("year"),
            ])
            frames.append(df)
            print(f"[ok] {profile.name}")
        except Exception as e:
            print(f"[!!] {profile.name}: {type(e).__name__}: {e}")

    if not frames:
        print("all profiles failed")
        sys.exit(1)

    combined = pl.concat(frames)

    out_path = Path(f"data/picks/year={year}/week={week}.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(out_path, compression="snappy")

    print(f"\nPicks — {today}  (year={year} week={week})")
    print(combined.select(["profile_name", "ticker", "price_entry", "reason"]))


if __name__ == "__main__":
    main()
