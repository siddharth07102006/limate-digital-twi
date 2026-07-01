
%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="India Climate Digital Twin", layout="wide")
st.title("🌍 AI-Powered Digital Twin of India's Climate")
st.caption("Google Colab Web Prototype Deployment")

# --- DATA GENERATION (Simulated IMD/ISRO Grid) ---
@st.cache_data
def generate_regional_grid():
    lats = np.arange(15.0, 22.0, 1.0)
    lons = np.arange(73.0, 80.0, 1.0)
    dates = pd.date_range(start="2026-06-01", periods=30)
    grid_data = []
    for d in dates:
        for lat in lats:
            for lon in lons:
                base_temp = 32.0 - (lat - 15) * 0.4
                base_rain = 5.0 + (lon - 73) * 1.2
                grid_data.append({
                    "Date": d, "Latitude": lat, "Longitude": lon,
                    "Temperature": round(base_temp + np.random.normal(0, 1.5), 2),
                    "Rainfall": round(max(0, base_rain + np.random.normal(0, 3)), 2)
                })
    return pd.DataFrame(grid_data)

df_grid = generate_regional_grid()

# --- TRAIN ML MODEL ---
@st.cache_resource
def train_climate_ai(df):
    df['Month'] = pd.to_datetime(df['Date']).dt.month
    df['Day'] = pd.to_datetime(df['Date']).dt.day
    X = df[['Latitude', 'Longitude', 'Month', 'Day', 'Temperature']]
    Y = df['Rainfall']
    model = RandomForestRegressor(n_estimators=30, random_state=42)
    model.fit(X, Y)
    return model

ai_model = train_climate_ai(df_grid)

# --- SIDEBAR INTERFACE ---
st.sidebar.header("🎛️ Digital Twin Controls")
user_temp_anomaly = st.sidebar.slider("Simulate Temperature Shift (°C)", -5.0, 5.0, 0.0, 0.5)

df_twin = df_grid.copy()
df_twin['Temperature'] += user_temp_anomaly
df_twin['Month'] = pd.to_datetime(df_twin['Date']).dt.month
df_twin['Day'] = pd.to_datetime(df_twin['Date']).dt.day
df_twin['Rainfall'] = ai_model.predict(df_twin[['Latitude', 'Longitude', 'Month', 'Day', 'Temperature']]).round(2)

# --- DASHBOARD MAP ---
st.subheader("🗺️ Live Geospatial Visualization Dashboard")
map_layer = st.radio("Select Layer", ["Temperature", "Rainfall"])
latest_date = df_twin['Date'].max()
map_df = df_twin[df_twin['Date'] == latest_date]

fig_map = px.scatter_mapbox(
    map_df, lat="Latitude", lon="Longitude", color=map_layer, size=map_layer,
    color_continuous_scale="Plasma" if map_layer == "Temperature" else "Blues",
    zoom=4.5, mapbox_style="carto-positron"
)
fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
st.plotly_chart(fig_map, use_container_width=True)
