import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import joblib
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler

TRAINING_DIR = Path("data/training")
MODEL_PATH = Path("model/random_forest.pkl")
SCALER_PATH = Path("model/scaler.pkl")
METADATA_PATH = Path("model/metadata.json")
MIN_ROWS = 30
FEATURE_COLS = ["rsi_14", "momentum_30", "volume_zscore_20", "macd_signal", "bollinger_width_20"]


def load_training_data() -> pl.DataFrame:
    parquet_files = list(TRAINING_DIR.glob("**/*.parquet"))
    if not parquet_files:
        return pl.DataFrame()

    paths = [str(p) for p in parquet_files]
    query = f"SELECT * FROM read_parquet({paths})"
    return pl.from_arrow(duckdb.execute(query).arrow())


def main() -> None:
    df = load_training_data()

    if len(df) < MIN_ROWS:
        print(f"insufficient data ({len(df)} rows), skipping")
        sys.exit(0)

    feature_cols = FEATURE_COLS
    df = df.with_columns(
        (pl.col("return_pct") > 0).cast(pl.Int8).alias("label")
    ).drop_nulls()

    X = df.select(feature_cols).to_numpy()
    y = df["label"].to_numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
    cv = cross_validate(model, X_scaled, y, cv=5, scoring=["accuracy", "f1"], return_train_score=False)

    val_accuracy = float(cv["test_accuracy"].mean())
    val_f1 = float(cv["test_f1"].mean())

    if METADATA_PATH.exists():
        current = json.loads(METADATA_PATH.read_text())
        if val_f1 <= current.get("val_f1", 0):
            print(f"no improvement (new f1={val_f1:.4f} <= current={current['val_f1']:.4f}), skipping save")
            return

    model.fit(X_scaled, y)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "val_accuracy": val_accuracy,
        "val_f1": val_f1,
        "n_training_rows": len(df),
        "feature_names": feature_cols,
        "deployed": True,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    print(f"model saved — val_accuracy={val_accuracy:.4f}, val_f1={val_f1:.4f}, rows={len(df)}")


if __name__ == "__main__":
    main()
