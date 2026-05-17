import json
from datetime import date, datetime, timezone
from pathlib import Path

import duckdb
import polars as pl

from src.utils.features import build_feature_matrix
from src.utils.market import get_current_price, get_sp500_return
from src.utils.universe import UNIVERSE

PICKS_DIR = Path("data/picks")
OUTCOMES_DIR = Path("data/outcomes")
TRAINING_DIR = Path("data/training")
RESULTS_PATH = Path("results/performance.json")
SETTLE_AFTER_DAYS = 7


def get_unresolved_weeks(con: duckdb.DuckDBPyConnection) -> list[tuple[int, int]]:
    if not any(PICKS_DIR.glob("year=*/week=*.parquet")):
        return []

    pick_weeks = {
        (r[0], r[1])
        for r in con.execute(
            "SELECT DISTINCT year, week "
            "FROM read_parquet('data/picks/year=*/week=*.parquet', hive_partitioning=true)"
        ).fetchall()
    }

    if not any(OUTCOMES_DIR.glob("year=*/week=*.parquet")):
        return list(pick_weeks)

    outcome_weeks = {
        (r[0], r[1])
        for r in con.execute(
            "SELECT DISTINCT year, week "
            "FROM read_parquet('data/outcomes/year=*/week=*.parquet', hive_partitioning=true)"
        ).fetchall()
    }

    return list(pick_weeks - outcome_weeks)


def settle_week(year: int, week: int, today: date) -> None:
    picks = pl.read_parquet(PICKS_DIR / f"year={year}" / f"week={week}.parquet")

    pick_date = date.fromisoformat(picks["pick_date"][0])
    if (today - pick_date).days < SETTLE_AFTER_DAYS:
        print(f"  {year}W{week:02d}: only {(today - pick_date).days}d old, skipping")
        return

    exit_prices = {t: get_current_price(t) for t in picks["ticker"].unique().to_list()}
    exit_df = pl.DataFrame({"ticker": list(exit_prices), "price_exit": list(exit_prices.values())})

    outcomes = picks.join(exit_df, on="ticker", how="left").with_columns([
        pl.lit(str(today)).alias("exit_date"),
        ((pl.col("price_exit") - pl.col("price_entry")) / pl.col("price_entry") * 100)
        .alias("return_pct"),
    ])

    out_path = OUTCOMES_DIR / f"year={year}" / f"week={week}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    outcomes.write_parquet(out_path, compression="snappy")
    print(f"  {year}W{week:02d}: settled {len(outcomes)} picks → {out_path}")

    features = build_feature_matrix(UNIVERSE)
    training = outcomes.join(features, on="ticker", how="left").drop_nulls()

    train_path = TRAINING_DIR / f"year={year}" / f"week={week}.parquet"
    train_path.parent.mkdir(parents=True, exist_ok=True)
    training.write_parquet(train_path, compression="snappy")
    print(f"  {year}W{week:02d}: training rows → {train_path}")


def build_performance(con: duckdb.DuckDBPyConnection) -> dict:
    if not any(OUTCOMES_DIR.glob("year=*/week=*.parquet")):
        return {}

    all_outcomes = con.execute(
        "SELECT * FROM read_parquet('data/outcomes/year=*/week=*.parquet', hive_partitioning=true)"
    ).pl()

    try:
        spy_return_pct = get_sp500_return(str(all_outcomes["pick_date"].min())) * 100
    except Exception:
        spy_return_pct = 0.0

    perf = {}
    for name in sorted(all_outcomes["profile_name"].unique().to_list()):
        data = all_outcomes.filter(pl.col("profile_name") == name)
        avg_return = float(data["return_pct"].mean())
        best = data.sort("return_pct", descending=True).row(0, named=True)
        worst = data.sort("return_pct").row(0, named=True)

        perf[name] = {
            "return_pct": round(avg_return, 2),
            "vs_sp500": round(avg_return - spy_return_pct, 2),
            "wins": int((data["return_pct"] > 0).sum()),
            "losses": int((data["return_pct"] <= 0).sum()),
            "best_pick": f"{best['ticker']} ({best['return_pct']:.1f}%)",
            "worst_pick": f"{worst['ticker']} ({worst['return_pct']:.1f}%)",
        }

    return perf


def main() -> None:
    today = datetime.now(timezone.utc).date()
    con = duckdb.connect()

    unresolved = get_unresolved_weeks(con)
    if unresolved:
        print(f"Settling {len(unresolved)} unresolved week(s)...")
        for year, week in sorted(unresolved):
            settle_week(year, week, today)
    else:
        print("No unresolved weeks.")

    perf = build_performance(con)
    if perf:
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(json.dumps(perf, indent=2))
        print(f"\nperformance.json — {len(perf)} profiles")
        for name, s in perf.items():
            print(f"  {name:20s} avg={s['return_pct']:+.1f}%  vs_spy={s['vs_sp500']:+.1f}%  W{s['wins']}/L{s['losses']}")


if __name__ == "__main__":
    main()
