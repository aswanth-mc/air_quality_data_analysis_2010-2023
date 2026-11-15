import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# 1. Page Configuration (collapsed sidebar avoids hiding filters off-screen)
st.set_page_config(
    page_title="Air Quality Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for mobile spacing, typography, and 2x2 KPI grid on small screens
st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    @media (max-width: 768px) {
        h1 { font-size: 1.6rem !important; }
        h2, h3 { font-size: 1.2rem !important; }
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
        }
        [data-testid="column"] {
            min-width: 46% !important;
            flex: 1 1 46% !important;
        }
    }
</style>
""", unsafe_allow_html=True)

st.title("Air Quality & Seasonal Time Series (2016–2023)")
st.caption("CPCB monitoring station data (`AP001`), seasonal monsoon washout effects, and Holt-Winters forecasting.")

# 2. Load Cleaned Data
@st.cache_data
def load_data():
    df = pd.read_csv("data/daily_cleaned.csv", index_col=0, parse_dates=True)
    return df

daily_df = load_data()

# 3. Mobile-Friendly Top Filter Bar (visible on both mobile & desktop)
with st.expander("Filter Pollutant & Years", expanded=True):
    filter_col1, filter_col2 = st.columns([1, 2])
    pollutants = [
        "PM2.5 (ug/m3)", "PM10 (ug/m3)", "NO2 (ug/m3)",
        "SO2 (ug/m3)", "CO (mg/m3)", "Ozone (ug/m3)"
    ]
    with filter_col1:
        selected_pollutant = st.selectbox("Select Pollutant", pollutants)

    years = sorted(daily_df["Year"].dropna().unique().astype(int))
    with filter_col2:
        selected_years = st.multiselect("Filter Years", years, default=years)

if not selected_years:
    st.warning("Please select at least one year to display data.")
    st.stop()

filtered_df = daily_df[daily_df["Year"].isin(selected_years)].copy()

# 4. Top KPI Summary Cards (2x2 grid on mobile, 4-across on desktop)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Average Level", f"{filtered_df[selected_pollutant].mean():.2f}")
col2.metric("Peak Daily Level", f"{filtered_df[selected_pollutant].max():.2f}")
col3.metric("Monsoon Avg", f"{filtered_df[filtered_df['Season'] == 'Monsoon'][selected_pollutant].mean():.2f}")
col4.metric("Winter Avg", f"{filtered_df[filtered_df['Season'] == 'Winter'][selected_pollutant].mean():.2f}")

st.divider()

# Shared mobile-safe Plotly layout helper
def style_mobile_chart(fig, height=360):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        dragmode=False  # Allows smooth page scrolling on touchscreens
    )
    return fig

# 5. Interactive Time Series Trend Chart
st.subheader(f"Daily Trend & 30-Day Rolling Avg — {selected_pollutant}")
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
fig_trend.update_layout(hovermode="x unified", yaxis_title=selected_pollutant)
style_mobile_chart(fig_trend, height=380)
st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar": False})

# 6. Two-Column Section (auto-stacks vertically on mobile)
left_col, right_col = st.columns(2)

with left_col:
    st.subheader("Seasonal Distribution")
    season_order = ["Winter", "Summer", "Monsoon", "Post-Monsoon"]
    fig_box = px.box(
        filtered_df, x="Season", y=selected_pollutant,
        category_orders={"Season": season_order}, color="Season"
    )
    fig_box.update_layout(showlegend=False)
    style_mobile_chart(fig_box, height=350)
    st.plotly_chart(fig_box, use_container_width=True, config={"displayModeBar": False})

with right_col:
    st.subheader("Yearly Average")
    yearly_df = filtered_df.groupby("Year")[selected_pollutant].mean().reset_index()
    fig_bar = px.bar(
        yearly_df, x="Year", y=selected_pollutant,
        text_auto=".1f", color_discrete_sequence=["#2b5c8f"]
    )
    style_mobile_chart(fig_bar, height=350)
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

st.divider()

# 7. Live 12-Month Holt-Winters Forecast Section
st.subheader(f"12-Month Forecast — {selected_pollutant}")
monthly_series = daily_df[selected_pollutant].resample("ME").mean().interpolate()

hw_model = ExponentialSmoothing(
    monthly_series, trend="add", seasonal="add", seasonal_periods=12
).fit()
future_forecast = hw_model.forecast(12)

fig_fc = go.Figure()
fig_fc.add_trace(go.Scatter(
    x=monthly_series.index, y=monthly_series,
    mode="lines+markers", name="Historical Monthly", line=dict(color="#2b5c8f")
))
fig_fc.add_trace(go.Scatter(
    x=future_forecast.index, y=future_forecast,
    mode="lines+markers", name="12-Month Forecast", line=dict(color="#d90429", dash="dot", width=3)
))
fig_fc.update_layout(hovermode="x unified", yaxis_title=selected_pollutant)
style_mobile_chart(fig_fc, height=380)
st.plotly_chart(fig_fc, use_container_width=True, config={"displayModeBar": False})