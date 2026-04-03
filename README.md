# Climate Change Analytics Dashboard

This project analyzes climate change indicators with a Kaggle dataset covering Indian cities across 2024-2025. It combines exploratory data analysis, seasonal trend tracking, air-quality and rainfall analysis, and a machine-learning model for short-term temperature forecasting.

## What the project includes

- Exploratory analysis of temperature, rainfall, humidity, AQI, pressure, and cloud cover
- Monthly seasonal summaries and city-wise climate comparisons
- Short-term temperature forecasting using lag-based climate features
- Model evaluation with baseline comparison, residual analysis, and feature importance
- An interactive Streamlit dashboard for presenting results

## Dataset

Kaggle source:
[Indian Climate Dataset (2024-2025)](https://www.kaggle.com/datasets/ankushnarwade/indian-climate-dataset-20242025)

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
