from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.data_utils import KAGGLE_DATASET_URL, load_dataset
from src.train_model import (
    DATA_PATH,
    FEATURES_PATH,
    FORECAST_BASE_PATH,
    METRICS_PATH,
    MODEL_PATH,
    TEST_PREDICTIONS_PATH,
    ensure_artifacts,
)

st.set_page_config(page_title="India Climate Forecast", layout="wide")


def build_feature_row(history: pd.DataFrame, future_date: pd.Timestamp) -> dict:
    ordered = history.sort_values("Date").reset_index(drop=True)
    return {
        "City": ordered.iloc[-1]["City"],
        "State": ordered.iloc[-1]["State"],
        "month": future_date.month,
        "day_of_year": future_date.dayofyear,
        "day_of_week": future_date.dayofweek,
        "is_weekend": int(future_date.dayofweek in [5, 6]),
        "temp_lag_1": float(ordered.iloc[-1]["Temperature_Avg_C"]),
        "temp_lag_2": float(ordered.iloc[-2]["Temperature_Avg_C"]),
        "temp_lag_3": float(ordered.iloc[-3]["Temperature_Avg_C"]),
        "temp_lag_7": float(ordered.iloc[-7]["Temperature_Avg_C"]),
        "temp_roll_3": float(ordered["Temperature_Avg_C"].tail(3).mean()),
        "temp_roll_7": float(ordered["Temperature_Avg_C"].tail(7).mean()),
        "humidity_lag_1": float(ordered.iloc[-1]["Humidity_pct"]),
        "rainfall_lag_1": float(ordered.iloc[-1]["Rainfall_mm"]),
        "aqi_lag_1": float(ordered.iloc[-1]["AQI"]),
        "pressure_lag_1": float(ordered.iloc[-1]["Pressure_hPa"]),
        "cloud_lag_1": float(ordered.iloc[-1]["Cloud_Cover_pct"]),
    }


@st.cache_resource
def load_dashboard_data():
    ensure_artifacts()
    dataset = load_dataset(DATA_PATH)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    model = joblib.load(MODEL_PATH)
    recent_history = pd.read_csv(FORECAST_BASE_PATH, parse_dates=["Date"])
    predictions = pd.read_csv(TEST_PREDICTIONS_PATH, parse_dates=["Date"])
    feature_table = pd.read_csv(FEATURES_PATH)

    return {
        "dataset": dataset,
        "metrics": metrics,
        "model": model,
        "recent_history": recent_history,
        "predictions": predictions,
        "feature_table": feature_table,
    }


def forecast_city(model, history: pd.DataFrame, horizon: int) -> pd.DataFrame:
    working = history.sort_values("Date").copy()
    rows: list[dict[str, object]] = []

    for _ in range(horizon):
        next_date = working["Date"].max() + pd.Timedelta(days=1)
        feature_row = build_feature_row(working, next_date)
        feature_frame = pd.DataFrame([feature_row])
        predicted_temp = float(model.predict(feature_frame)[0])

        next_row = working.iloc[-1].copy()
        next_row["Date"] = next_date
        next_row["Temperature_Avg_C"] = predicted_temp
        working = pd.concat([working, pd.DataFrame([next_row])], ignore_index=True)

        rows.append(
            {
                "Date": next_date,
                "Predicted_Temperature_C": round(predicted_temp, 2),
                "City": feature_row["City"],
                "State": feature_row["State"],
            }
        )

    return pd.DataFrame(rows)


def draw_residual_chart(predictions: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(predictions["Temperature_Avg_C"], predictions["predicted_temp"], alpha=0.65, color="#1f77b4")
    line_min = min(predictions["Temperature_Avg_C"].min(), predictions["predicted_temp"].min())
    line_max = max(predictions["Temperature_Avg_C"].max(), predictions["predicted_temp"].max())
    ax.plot([line_min, line_max], [line_min, line_max], linestyle="--", color="gray")
    ax.set_xlabel("Actual Average Temperature")
    ax.set_ylabel("Predicted Average Temperature")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)


def main():
    resources = load_dashboard_data()
    dataset = resources["dataset"]
    metrics = resources["metrics"]
    model = resources["model"]
    recent_history = resources["recent_history"]
    predictions = resources["predictions"]

    st.title("India Climate Forecast Dashboard")
    st.write(
        "A city-wise climate dashboard built on recent Indian weather and air-quality data."
    )
    st.link_button("Open Kaggle Dataset", KAGGLE_DATASET_URL)

    top_a, top_b, top_c, top_d = st.columns(4)
    top_a.metric("Rows", metrics["dataset_rows"])
    top_b.metric("Cities", metrics["cities"])
    top_c.metric("Test MAE", f'{metrics["mae"]:.2f} C')
    top_d.metric("Test RMSE", f'{metrics["rmse"]:.2f} C')
    st.caption(f'Dataset range: {metrics["date_start"]} to {metrics["date_end"]}')

    cities = sorted(dataset["City"].dropna().unique().tolist())
    selected_city = st.selectbox("Select a city", cities, index=0 if cities else None)

    city_data = dataset[dataset["City"] == selected_city].sort_values("Date")
    city_history = recent_history[recent_history["City"] == selected_city].sort_values("Date")

    tab1, tab2, tab3 = st.tabs(["Forecast", "Climate Trends", "Model Snapshot"])

    with tab1:
        st.subheader(f"Forecast for {selected_city}")
        horizon = st.slider("Forecast days", 3, 30, 7, 1)
        if st.button("Generate Forecast", use_container_width=True):
            forecast_df = forecast_city(model, city_history, horizon)
            combined = pd.concat(
                [
                    city_data[["Date", "Temperature_Avg_C"]].tail(30).assign(series="Historical"),
                    forecast_df.rename(columns={"Predicted_Temperature_C": "Temperature_Avg_C"}).assign(series="Forecast"),
                ],
                ignore_index=True,
            )
            st.dataframe(forecast_df, use_container_width=True)
            st.line_chart(combined, x="Date", y="Temperature_Avg_C", color="series")
        else:
            st.caption("Forecast appears here after you pick a city and horizon.")

    with tab2:
        st.subheader("Climate trends")
        left, right = st.columns(2)
        with left:
            st.write("Average temperature over time")
            st.line_chart(city_data, x="Date", y="Temperature_Avg_C")

            st.write("Rainfall over time")
            st.bar_chart(city_data.set_index("Date")["Rainfall_mm"])

        with right:
            st.write("Humidity and cloud cover")
            humidity_cloud = city_data[["Date", "Humidity_pct", "Cloud_Cover_pct"]].set_index("Date")
            st.line_chart(humidity_cloud)

            st.write("AQI trend")
            st.line_chart(city_data, x="Date", y="AQI")

        monthly_summary = (
            city_data.assign(Month=city_data["Date"].dt.to_period("M").astype(str))
            .groupby("Month", as_index=False)[["Temperature_Avg_C", "Rainfall_mm", "AQI"]]
            .mean()
        )
        st.write("Recent monthly averages")
        st.dataframe(monthly_summary.tail(12), use_container_width=True)

    with tab3:
        st.subheader("Model snapshot")
        metric_table = pd.DataFrame(
            {
                "Metric": ["Train R2", "Test R2", "MAE", "RMSE", "Train Rows", "Test Rows"],
                "Value": [
                    metrics["train_r2"],
                    metrics["test_r2"],
                    metrics["mae"],
                    metrics["rmse"],
                    metrics["train_rows"],
                    metrics["test_rows"],
                ],
            }
        )
        st.dataframe(metric_table, use_container_width=True)
        draw_residual_chart(predictions)
        st.write("Model features")
        st.dataframe(resources["feature_table"], use_container_width=True)


if __name__ == "__main__":
    main()
