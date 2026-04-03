from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data_utils import KAGGLE_DATASET_URL, load_dataset

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_PATH = DATA_DIR / "Indian_Climate_Dataset_2024_2025.csv"
MODEL_PATH = ARTIFACTS_DIR / "climate_forecast_model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
FEATURES_PATH = ARTIFACTS_DIR / "feature_importance.csv"
FORECAST_BASE_PATH = ARTIFACTS_DIR / "forecast_base.csv"
TEST_PREDICTIONS_PATH = ARTIFACTS_DIR / "test_predictions.csv"
CITY_SUMMARY_PATH = ARTIFACTS_DIR / "city_summary.csv"
MONTHLY_SUMMARY_PATH = ARTIFACTS_DIR / "monthly_summary.csv"
CORRELATION_PATH = ARTIFACTS_DIR / "correlation_matrix.csv"

FEATURE_COLUMNS = [
    "City",
    "State",
    "month",
    "day_of_year",
    "day_of_week",
    "is_weekend",
    "temp_lag_1",
    "temp_lag_2",
    "temp_lag_3",
    "temp_lag_7",
    "temp_roll_3",
    "temp_roll_7",
    "humidity_lag_1",
    "rainfall_lag_1",
    "aqi_lag_1",
    "pressure_lag_1",
    "cloud_lag_1",
]
TARGET_COLUMN = "Temperature_Avg_C"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["month"] = enriched["Date"].dt.month
    enriched["day_of_year"] = enriched["Date"].dt.dayofyear
    enriched["day_of_week"] = enriched["Date"].dt.dayofweek
    enriched["is_weekend"] = enriched["day_of_week"].isin([5, 6]).astype(int)

    city_group = enriched.groupby("City", group_keys=False)
    enriched["temp_lag_1"] = city_group["Temperature_Avg_C"].shift(1)
    enriched["temp_lag_2"] = city_group["Temperature_Avg_C"].shift(2)
    enriched["temp_lag_3"] = city_group["Temperature_Avg_C"].shift(3)
    enriched["temp_lag_7"] = city_group["Temperature_Avg_C"].shift(7)
    enriched["temp_roll_3"] = city_group["Temperature_Avg_C"].shift(1).rolling(3).mean()
    enriched["temp_roll_7"] = city_group["Temperature_Avg_C"].shift(1).rolling(7).mean()
    enriched["humidity_lag_1"] = city_group["Humidity_pct"].shift(1)
    enriched["rainfall_lag_1"] = city_group["Rainfall_mm"].shift(1)
    enriched["aqi_lag_1"] = city_group["AQI"].shift(1)
    enriched["pressure_lag_1"] = city_group["Pressure_hPa"].shift(1)
    enriched["cloud_lag_1"] = city_group["Cloud_Cover_pct"].shift(1)

    return enriched.dropna(subset=["temp_lag_7"]).reset_index(drop=True)


def build_pipeline() -> Pipeline:
    numeric_columns = [column for column in FEATURE_COLUMNS if column not in {"City", "State"}]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                ["City", "State"],
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_columns,
            ),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=220,
        max_depth=14,
        min_samples_leaf=3,
        n_jobs=1,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def train_and_save(force: bool = False) -> dict:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_df = load_dataset(DATA_PATH)
    df = build_features(raw_df)

    split_date = df["Date"].max() - pd.Timedelta(days=45)
    train_df = df[df["Date"] <= split_date].copy()
    test_df = df[df["Date"] > split_date].copy()

    pipeline = build_pipeline()
    pipeline.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN])

    train_predictions = pipeline.predict(train_df[FEATURE_COLUMNS])
    test_predictions = pipeline.predict(test_df[FEATURE_COLUMNS])
    baseline_predictions = test_df["temp_lag_1"].to_numpy()

    train_r2 = r2_score(train_df[TARGET_COLUMN], train_predictions)
    test_r2 = r2_score(test_df[TARGET_COLUMN], test_predictions)
    mae = mean_absolute_error(test_df[TARGET_COLUMN], test_predictions)
    rmse = mean_squared_error(test_df[TARGET_COLUMN], test_predictions) ** 0.5
    baseline_mae = mean_absolute_error(test_df[TARGET_COLUMN], baseline_predictions)
    baseline_rmse = mean_squared_error(test_df[TARGET_COLUMN], baseline_predictions) ** 0.5

    prediction_frame = test_df[["Date", "City", "State", TARGET_COLUMN]].copy()
    prediction_frame["predicted_temp"] = test_predictions
    prediction_frame["residual"] = prediction_frame[TARGET_COLUMN] - prediction_frame["predicted_temp"]
    prediction_frame.to_csv(TEST_PREDICTIONS_PATH, index=False)

    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    importances = pipeline.named_steps["model"].feature_importances_
    feature_table = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    feature_table.to_csv(FEATURES_PATH, index=False)

    latest_rows = raw_df.sort_values("Date").groupby("City", as_index=False).tail(7)
    latest_rows.to_csv(FORECAST_BASE_PATH, index=False)

    city_summary = (
        raw_df.groupby(["City", "State"], as_index=False)
        .agg(
            avg_temperature=("Temperature_Avg_C", "mean"),
            max_temperature=("Temperature_Max_C", "max"),
            total_rainfall=("Rainfall_mm", "sum"),
            avg_humidity=("Humidity_pct", "mean"),
            avg_aqi=("AQI", "mean"),
        )
        .sort_values("avg_temperature", ascending=False)
    )
    city_summary.to_csv(CITY_SUMMARY_PATH, index=False)

    monthly_summary = (
        raw_df.assign(
            month_number=raw_df["Date"].dt.month,
            month_name=raw_df["Date"].dt.strftime("%b"),
        )
        .groupby(["month_number", "month_name"], as_index=False)
        .agg(
            avg_temperature=("Temperature_Avg_C", "mean"),
            avg_rainfall=("Rainfall_mm", "mean"),
            avg_humidity=("Humidity_pct", "mean"),
            avg_aqi=("AQI", "mean"),
        )
        .sort_values("month_number")
    )
    monthly_summary.to_csv(MONTHLY_SUMMARY_PATH, index=False)

    correlation_matrix = raw_df[
        [
            "Temperature_Avg_C",
            "Humidity_pct",
            "Rainfall_mm",
            "Wind_Speed_kmh",
            "AQI",
            "Pressure_hPa",
            "Cloud_Cover_pct",
        ]
    ].corr(numeric_only=True)
    correlation_matrix.to_csv(CORRELATION_PATH)

    joblib.dump(pipeline, MODEL_PATH)

    metrics = {
        "dataset_source": KAGGLE_DATASET_URL,
        "dataset_rows": int(len(raw_df)),
        "dataset_columns": int(len(raw_df.columns)),
        "cities": int(raw_df["City"].nunique()),
        "date_start": raw_df["Date"].min().strftime("%Y-%m-%d"),
        "date_end": raw_df["Date"].max().strftime("%Y-%m-%d"),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_r2": round(float(train_r2), 4),
        "test_r2": round(float(test_r2), 4),
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "baseline_mae": round(float(baseline_mae), 4),
        "baseline_rmse": round(float(baseline_rmse), 4),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def ensure_artifacts(force: bool = False) -> dict:
    needed = [
        MODEL_PATH,
        METRICS_PATH,
        FEATURES_PATH,
        FORECAST_BASE_PATH,
        TEST_PREDICTIONS_PATH,
        CITY_SUMMARY_PATH,
        MONTHLY_SUMMARY_PATH,
        CORRELATION_PATH,
    ]
    if force or any(not path.exists() for path in needed):
        return train_and_save(force=True)
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    print(json.dumps(train_and_save(force=True), indent=2))
