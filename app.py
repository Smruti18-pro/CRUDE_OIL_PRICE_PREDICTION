# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings("ignore")

# ------------------------
# Load Data
# ------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("crude_oil_multicountry_2020_2025.xlsx")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

df = load_data()

# ------------------------
# Features & Target
# ------------------------
target_col = "Country_Spot_Price_USD"
features = [c for c in df.columns if c not in ["Date", target_col]]

# ------------------------
# Sidebar Controls
# ------------------------
st.sidebar.header("⚙ Model Controls")
country = st.sidebar.selectbox("Select Country", df["Country"].unique())
model_type = st.sidebar.radio("Select Model", ["Regression (fundamentals)", "Time-series"])
forecast_months = st.sidebar.slider("Forecast Horizon (Months)", min_value=6, max_value=36, value=24, step=6)

# ------------------------
# Train Models
# ------------------------
def train_regression(data):
    X = data[features]
    y = data[target_col]

    categorical = [col for col in features if df[col].dtype == "object"]
    transformer = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical)],
        remainder="passthrough"
    )
    model = Pipeline(steps=[
        ("transform", transformer),
        ("model", RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    model.fit(X, y)
    return model

def make_lag_features(data, lags=3):
    df_lag = data.copy()
    for lag in range(1, lags+1):
        df_lag[f"lag_{lag}"] = df_lag[target_col].shift(lag)
    return df_lag

def train_timeseries(data):
    df_lag = make_lag_features(data)
    df_lag = df_lag.dropna()
    X = df_lag[[f"lag_{i}" for i in range(1, 4)]]
    y = df_lag[target_col]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, df_lag

def forecast_future(data, model, months=24):
    future = []
    last_row = data.iloc[-1].copy()
    lags = [last_row[target_col]] * 3

    for m in range(months):
        X_pred = np.array(lags[-3:]).reshape(1, -1)
        y_pred = model.predict(X_pred)[0]
        next_date = last_row["Date"] + pd.DateOffset(months=m+1)
        future.append({"Date": next_date, "Forecast": y_pred})
        lags.append(y_pred)

    return pd.DataFrame(future)

# ------------------------
# Fit Model
# ------------------------
if model_type == "Regression (fundamentals)":
    model = train_regression(df[df["Country"] == country])
else:
    model, df_lag = train_timeseries(df[df["Country"] == country])

# ------------------------
# Show Results
# ------------------------
st.title("🌍 Crude Oil Price Prediction Dashboard")

country_df = df[df["Country"] == country].copy()

# Historical Data
st.subheader(f"Historical Spot Price for {country}")
st.line_chart(country_df.set_index("Date")[target_col])

# Prediction Section
st.subheader(f"{model_type} Prediction for {country}")

if model_type == "Regression (fundamentals)":
    yhat = model.predict(country_df[features])
    country_df["Predicted"] = yhat
    st.line_chart(country_df.set_index("Date")[["Country_Spot_Price_USD","Predicted"]])

else:  # Time-series
    df_ts = df_lag.copy()
    yhat = model.predict(df_ts[[f"lag_{i}" for i in range(1, 4)]])
    df_ts["Predicted"] = yhat

    # Show backtest predictions
    st.line_chart(df_ts.set_index("Date")[["Country_Spot_Price_USD","Predicted"]])

    # Future Forecast
    st.subheader(f"🔮 Next {forecast_months} Months Forecast")
    forecast_df = forecast_future(country_df, model, months=forecast_months)
    st.table(forecast_df)

    # Plot with shading for forecast
    combined = pd.concat([
        country_df[["Date", target_col]].set_index("Date"),
        forecast_df.set_index("Date").rename(columns={"Forecast": target_col})
    ])

    fig, ax = plt.subplots(figsize=(10, 5))
    # Historical line
    country_df.plot(x="Date", y=target_col, ax=ax, label="Actual", color="blue")
    # Forecast line
    forecast_df.plot(x="Date", y="Forecast", ax=ax, label="Forecast", color="orange")
    # Shaded forecast region
    ax.axvspan(forecast_df["Date"].min(), forecast_df["Date"].max(),
               color="orange", alpha=0.15, label="Forecast Period")

    ax.set_title(f"Crude Oil Price Forecast ({country})")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    st.pyplot(fig)
