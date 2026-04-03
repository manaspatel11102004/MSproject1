from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.data_utils import KAGGLE_DATASET_URL, load_dataset
from src.train_model import (
    CITY_SUMMARY_PATH,
    CORRELATION_PATH,
    DATA_PATH,
    FEATURES_PATH,
    FORECAST_BASE_PATH,
    METRICS_PATH,
    MODEL_PATH,
    MONTHLY_SUMMARY_PATH,
    TEST_PREDICTIONS_PATH,
    ensure_artifacts,
)

st.set_page_config(page_title="Climate Change Analytics", layout="wide")


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
    return {
        "dataset": load_dataset(DATA_PATH),
        "metrics": json.loads(METRICS_PATH.read_text(encoding="utf-8")),
        "model": joblib.load(MODEL_PATH),
        "recent_history": pd.read_csv(FORECAST_BASE_PATH, parse_dates=["Date"]),
        "predictions": pd.read_csv(TEST_PREDICTIONS_PATH, parse_dates=["Date"]),
        "feature_table": pd.read_csv(FEATURES_PATH),
        "city_summary": pd.read_csv(CITY_SUMMARY_PATH),
        "monthly_summary": pd.read_csv(MONTHLY_SUMMARY_PATH),
        "correlation_matrix": pd.read_csv(CORRELATION_PATH, index_col=0),
    }


def forecast_city(model, history: pd.DataFrame, horizon: int) -> pd.DataFrame:
    working = history.sort_values("Date").copy()
    rows: list[dict[str, object]] = []

    for _ in range(horizon):
        next_date = working["Date"].max() + pd.Timedelta(days=1)
        feature_row = build_feature_row(working, next_date)
        predicted_temp = float(model.predict(pd.DataFrame([feature_row]))[0])

        next_row = working.iloc[-1].copy()
        next_row["Date"] = next_date
        next_row["Temperature_Avg_C"] = predicted_temp
        working = pd.concat([working, pd.DataFrame([next_row])], ignore_index=True)

        rows.append(
            {
                "Date": next_date,
                "City": feature_row["City"],
                "State": feature_row["State"],
                "Predicted_Temperature_C": round(predicted_temp, 2),
            }
        )

    return pd.DataFrame(rows)


def make_scatter_plot(predictions: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.scatter(predictions["Temperature_Avg_C"], predictions["predicted_temp"], alpha=0.7, color="#0f766e")
    diagonal_min = min(predictions["Temperature_Avg_C"].min(), predictions["predicted_temp"].min())
    diagonal_max = max(predictions["Temperature_Avg_C"].max(), predictions["predicted_temp"].max())
    ax.plot([diagonal_min, diagonal_max], [diagonal_min, diagonal_max], linestyle="--", color="#475569")
    ax.set_xlabel("Actual temperature")
    ax.set_ylabel("Predicted temperature")
    ax.set_title("Model fit on held-out data")
    st.pyplot(fig)


def make_residual_plot(predictions: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.hist(predictions["residual"], bins=20, color="#f97316", edgecolor="white")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    ax.set_title("Residual distribution")
    st.pyplot(fig)


def show_overview_tab(dataset: pd.DataFrame, city_summary: pd.DataFrame, monthly_summary: pd.DataFrame, metrics: dict):
    hottest_city = city_summary.iloc[0]
    worst_air_city = city_summary.sort_values("avg_aqi", ascending=False).iloc[0]
    wettest_city = city_summary.sort_values("total_rainfall", ascending=False).iloc[0]

    top_a, top_b, top_c, top_d = st.columns(4)
    top_a.metric("Observations", metrics["dataset_rows"])
    top_b.metric("Cities Covered", metrics["cities"])
    top_c.metric("Date Range", f'{metrics["date_start"]} to {metrics["date_end"]}')
    top_d.metric("Model RMSE", f'{metrics["rmse"]:.2f} C')

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Why this project matters")
        st.write(
            "Climate data changes across time, geography, and environmental conditions. "
            "This dashboard uses a Kaggle weather dataset to explore warming patterns, rainfall shifts, "
            "air-quality stress, and short-term temperature forecasting across major Indian cities."
        )
        st.write("Monthly climate profile")
        st.line_chart(
            monthly_summary.set_index("month_name")[["avg_temperature", "avg_rainfall", "avg_aqi"]]
        )

    with right:
        st.subheader("Quick findings")
        st.write(
            f"- Warmest average city in this dataset: **{hottest_city['City']}** ({hottest_city['avg_temperature']:.1f} C)"
        )
        st.write(
            f"- Highest average AQI: **{worst_air_city['City']}** ({worst_air_city['avg_aqi']:.0f})"
        )
        st.write(
            f"- Highest total rainfall: **{wettest_city['City']}** ({wettest_city['total_rainfall']:.1f} mm)"
        )
        st.write("City comparison")
        st.dataframe(city_summary.round(2), use_container_width=True)


def show_trends_tab(city_data: pd.DataFrame, monthly_summary: pd.DataFrame, correlation_matrix: pd.DataFrame):
    st.subheader("Long-term trends and seasonality")
    trend_left, trend_right = st.columns(2)

    with trend_left:
        st.write("Temperature trend")
        st.line_chart(city_data, x="Date", y="Temperature_Avg_C")
        st.write("Rainfall trend")
        st.bar_chart(city_data.set_index("Date")["Rainfall_mm"])

    with trend_right:
        st.write("Humidity and cloud cover")
        st.line_chart(city_data.set_index("Date")[["Humidity_pct", "Cloud_Cover_pct"]])
        st.write("AQI trend")
        st.line_chart(city_data, x="Date", y="AQI")

    st.write("Seasonal monthly averages across the Kaggle dataset")
    st.dataframe(monthly_summary.round(2), use_container_width=True)

    st.write("Climate variable correlation matrix")
    st.dataframe(correlation_matrix.round(2), use_container_width=True)


def show_forecast_tab(model, city_data: pd.DataFrame, city_history: pd.DataFrame):
    st.subheader("Temperature forecasting lab")
    st.write(
        "The forecasting model uses lagged temperature, humidity, rainfall, AQI, pressure, and cloud-cover features "
        "to predict the next few days of average temperature."
    )

    horizon = st.slider("Forecast horizon (days)", min_value=3, max_value=30, value=10, step=1)
    if st.button("Run forecast", use_container_width=True):
        forecast_df = forecast_city(model, city_history, horizon)
        chart_data = pd.concat(
            [
                city_data[["Date", "Temperature_Avg_C"]].tail(45).assign(series="Historical"),
                forecast_df.rename(columns={"Predicted_Temperature_C": "Temperature_Avg_C"}).assign(series="Forecast"),
            ],
            ignore_index=True,
        )
        st.dataframe(forecast_df, use_container_width=True)
        st.line_chart(chart_data, x="Date", y="Temperature_Avg_C", color="series")
    else:
        st.caption("Choose a city and forecast horizon to generate the projection.")


def show_model_tab(predictions: pd.DataFrame, feature_table: pd.DataFrame, metrics: dict):
    st.subheader("Model evaluation")
    metric_table = pd.DataFrame(
        {
            "Metric": [
                "Train R2",
                "Test R2",
                "Model MAE",
                "Model RMSE",
                "Baseline MAE",
                "Baseline RMSE",
            ],
            "Value": [
                metrics["train_r2"],
                metrics["test_r2"],
                metrics["mae"],
                metrics["rmse"],
                metrics["baseline_mae"],
                metrics["baseline_rmse"],
            ],
        }
    )
    st.dataframe(metric_table, use_container_width=True)

    plot_left, plot_right = st.columns(2)
    with plot_left:
        make_scatter_plot(predictions)
    with plot_right:
        make_residual_plot(predictions)

    st.write("Top feature signals")
    st.dataframe(feature_table.head(12).round(4), use_container_width=True)


def main():
    resources = load_dashboard_data()
    dataset = resources["dataset"]
    metrics = resources["metrics"]
    model = resources["model"]
    recent_history = resources["recent_history"]
    predictions = resources["predictions"]
    city_summary = resources["city_summary"]
    monthly_summary = resources["monthly_summary"]
    correlation_matrix = resources["correlation_matrix"]

    st.title("Climate Change Analytics Dashboard")
    st.write(
        "Exploratory analysis and forecasting on a Kaggle climate dataset covering Indian cities in 2024-2025."
    )
    st.link_button("View Kaggle Dataset", KAGGLE_DATASET_URL)

    cities = sorted(dataset["City"].dropna().unique().tolist())
    selected_city = st.selectbox("Select a city", cities, index=0 if cities else None)
    city_data = dataset[dataset["City"] == selected_city].sort_values("Date")
    city_history = recent_history[recent_history["City"] == selected_city].sort_values("Date")

    overview_tab, trends_tab, forecast_tab, model_tab = st.tabs(
        ["Overview", "Trends", "Forecast", "Model Insight"]
    )

    with overview_tab:
        show_overview_tab(dataset, city_summary, monthly_summary, metrics)
    with trends_tab:
        show_trends_tab(city_data, monthly_summary, correlation_matrix)
    with forecast_tab:
        show_forecast_tab(model, city_data, city_history)
    with model_tab:
        show_model_tab(predictions, resources["feature_table"], metrics)


if __name__ == "__main__":
    main()
