# Air Quality & Seasonal Time Series Analytics Dashboard (2016–2023)

End-to-end data analytics and time series forecasting project analyzing 59,150 hourly CPCB air quality and meteorological observations (`AP001.csv`, July 2016 – March 2023).

## Key Insights
- **Monsoon Washout Effect:** Mean `PM2.5` drops by 67% from Winter (`45.02 µg/m³`) to Monsoon (`14.89 µg/m³`).
- **COVID-19 Lockdown Impact:** Traffic-driven `NO2` plunged to an annual low of `17.28 µg/m³` in 2020 (down from `35.72 µg/m³` in 2019).
- **Seasonal Forecasting:** Built an additive Holt-Winters Exponential Smoothing model (`MAE: 8.12 µg/m³`, `RMSE: 9.34 µg/m³` on the 2022–2023 holdout set) with a 12-month forward projection.

## Tech Stack
- **Data Cleaning & EDA:** Python, Pandas, NumPy, Matplotlib, Seaborn
- **Time Series Modeling:** Statsmodels (STL Decomposition, Holt-Winters)
- **Interactive Dashboard:** Streamlit, Plotly

- ## [Live Demo] [https://airqualitydataanalysis2010-2023-4aokcbera4wcnhtcxvsrnu.streamlit.app/]
