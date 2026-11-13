import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# 1. Page Configuration
st.set_page_config(page_title="Air Quality Analytics Dashboard", layout="wide")
st.title("Air Quality & Seasonal Time Series Dashboard (2016–2023)")
st.markdown("Interactive exploration of CPCB monitoring station data (`AP001.csv`), seasonal monsoon washout effects, and Holt-Winters forecasting.")

# 2. Load Cleaned Data (cached so filters respond instantly)
@st.cache_data
def load_data():
    df = pd.read_csv("data/daily_cleaned.csv", index_col=0, parse_dates=True)
    return df

daily_df = load_data()

# 3. Sidebar Filters
st.sidebar.header("Dashboard Filters")
pollutants = [
    "PM2.5 (ug/m3)", "PM10 (ug/m3)", "NO2 (ug/m3)",
    "SO2 (ug/m3)", "CO (mg/m3)", "Ozone (ug/m3)"
]
selected_pollutant = st.sidebar.selectbox("Select Pollutant", pollutants)

years = sorted(daily_df["Year"].dropna().unique().astype(int))
selected_years = st.sidebar.multiselect("Filter Years", years, default=years)

filtered_df = daily_df[daily_df["Year"].isin(selected_years)].copy()

# 4. Top KPI Summary Cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("Average Level", f"{filtered_df[selected_pollutant].mean():.2f}")
col2.metric("Peak Daily Level", f"{filtered_df[selected_pollutant].max():.2f}")
col3.metric("Monsoon Avg", f"{filtered_df[filtered_df['Season'] == 'Monsoon'][selected_pollutant].mean():.2f}")
col4.metric("Winter Avg", f"{filtered_df[filtered_df['Season'] == 'Winter'][selected_pollutant].mean():.2f}")

st.divider()

# 5. Interactive Time Series Trend Chart
st.subheader(f"Daily Trend & 30-Day Rolling Average — {selected_pollutant}")
filtered_df["Rolling_30d"] = filtered_df[selected_pollutant].rolling(window=30, min_periods=7).mean()

fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(
    x=filtered_df.index, y=filtered_df[selected_pollutant],
    mode="lines", name="Daily Reading", line=dict(color="lightgray", width=1)
))
fig_trend.add_trace(go.Scatter(
    x=filtered_df.index, y=filtered_df["Rolling_30d"],
    mode="lines", name="30-Day Rolling Avg", line=dict(color="#168653", width=2.5)
))
fig_trend.update_layout(height=400, hovermode="x unified", yaxis_title=selected_pollutant)
st.plotly_chart(fig_trend, use_container_width=True)

# 6. Two-Column Section: Seasonal Boxplot & Yearly Comparison
left_col, right_col = st.columns(2)

with left_col:
    st.subheader("Seasonal Distribution (Monsoon Washout)")
    season_order = ["Winter", "Summer", "Monsoon", "Post-Monsoon"]
    fig_box = px.box(
        filtered_df, x="Season", y=selected_pollutant,
        category_orders={"Season": season_order}, color="Season"
    )
    fig_box.update_layout(showlegend=False, height=380)
    st.plotly_chart(fig_box, use_container_width=True)

with right_col:
    st.subheader("Yearly Average Comparison")
    yearly_df = filtered_df.groupby("Year")[selected_pollutant].mean().reset_index()
    fig_bar = px.bar(yearly_df, x="Year", y=selected_pollutant, text_auto=".1f", color_discrete_sequence=["#2b5c8f"])
    fig_bar.update_layout(height=380)
    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# 7. Live 12-Month Holt-Winters Forecast Section
st.subheader(f"12-Month Time Series Forecast — {selected_pollutant}")
monthly_series = daily_df[selected_pollutant].resample("ME").mean().interpolate()

hw_model = ExponentialSmoothing(
    monthly_series, trend="add", seasonal="add", seasonal_periods=12
).fit()
future_forecast = hw_model.forecast(12)

fig_fc = go.Figure()
fig_fc.add_trace(go.Scatter(
    x=monthly_series.index, y=monthly_series,
    mode="lines+markers", name="Historical Monthly Avg", line=dict(color="#2b5c8f")
))
fig_fc.add_trace(go.Scatter(
    x=future_forecast.index, y=future_forecast,
    mode="lines+markers", name="12-Month Forecast", line=dict(color="#d90429", dash="dot", width=3)
))
fig_fc.update_layout(height=400, hovermode="x unified", yaxis_title=selected_pollutant)
st.plotly_chart(fig_fc, use_container_width=True)