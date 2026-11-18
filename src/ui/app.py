"""
PathSnap Evaluator User Interface (Laptop 2)
--------------------------------------------
Streamlit Web Interface that communicates with the Laptop 1 FastAPI server,
displays evaluation metrics, and renders interactive Folium map layers.
"""

from __future__ import annotations
import streamlit as st
import requests
import folium
from streamlit_folium import st_folium
import numpy as np
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="PathSnap — Map-Matching Evaluator", layout="wide", page_icon="🗺️")

st.title("🗺️ PathSnap: Distributed Map-Matching & Route Reconstruction")
st.markdown("**PES University | UE24CS352A Machine Learning Hackathon** — *Two-Laptop Deployment System*")

# Sidebar Configuration
st.sidebar.header("📡 Backend Server Configuration (Laptop 1)")
api_host = st.sidebar.text_input("FastAPI Server URL", value="http://localhost:8000")

# Fetch available trips from API
trip_options = [
    "00000000", "00000007", "00000009", "00000040", "00000052",
    "00000056", "00000065", "00000076", "00000077", "00000091"
]
try:
    resp = requests.get(f"{api_host}/trips", timeout=3)
    if resp.status_code == 200:
        fetched_trips = resp.json().get("available_trips", [])
        if fetched_trips:
            trip_options = fetched_trips
except Exception:
    st.sidebar.warning("⚠️ Could not reach Laptop 1 API. Using default evaluation trip IDs.")

selected_trip = st.sidebar.selectbox("Select Evaluation Trip (1 of 10 Reserved Trips)", options=trip_options)

st.sidebar.markdown("---")
evaluate_btn = st.sidebar.button("🚀 Run Map-Matching & Route Reconstruction", type="primary")

# Main View
if evaluate_btn:
    with st.spinner(f"Sending evaluation request for Trip {selected_trip} to {api_host}..."):
        try:
            api_url = f"{api_host}/evaluate_trip/{selected_trip}"
            res = requests.post(api_url, timeout=60)
            if res.status_code == 200:
                data = res.json()
                st.session_state["eval_result"] = data
                st.success(f"✅ Successfully evaluated Trip {selected_trip}!")
            else:
                st.error(f"Server Error {res.status_code}: {res.text}")
        except Exception as e:
            st.error(f"Failed to communicate with Laptop 1 API: {e}")

if "eval_result" in st.session_state:
    data = st.session_state["eval_result"]

    # Display Metric KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Map-Matching Accuracy", f"{data['top1_accuracy_percent']:.1f}%")
    col2.metric("Classifier Top-1", f"{data.get('classifier_top1_percent', 0.0):.1f}%")
    col3.metric("Route Continuity Score", f"{data['route_continuity_percent']:.1f}%")
    col4.metric("Candidate Miss Rate", f"{data['candidate_miss_rate_percent']:.2f}%")
    col5.metric("Inference Latency", f"{data['inference_time_seconds']:.2f} s")

    # Trajectory Information
    st.subheader(f"📊 Trajectory & Route Summary: {data['trip_id']}")
    st.write(f"- Total GPS Observations: **{data['total_observations']}**")
    st.write(f"- Reconstructed Road Arcs Count: **{data['reconstructed_arc_count']}**")
    st.write(f"- Sample Reconstructed Arcs: `{data['reconstructed_route_sample']}`")

# Map Visualization Section
st.subheader("📍 Interactive Folium Geographical Map")
st.markdown("Displays **Raw GPS points (Blue)** and **Reconstructed Trajectory (Green)**.")

if "eval_result" in st.session_state and st.session_state["eval_result"].get("gps_coordinates"):
    coords = st.session_state["eval_result"]["gps_coordinates"]
    mid_lat = float(np.mean([c[0] for c in coords]))
    mid_lon = float(np.mean([c[1] for c in coords]))
    m = folium.Map(location=[mid_lat, mid_lon], zoom_start=14, tiles="cartodbpositron")

    # Add Raw GPS points line
    folium.PolyLine(coords, color="#2980b9", weight=4, opacity=0.7, tooltip="Raw GPS Observations").add_to(m)

    # Add Reconstructed Route
    route_coords = st.session_state["eval_result"].get("route_coordinates", [])
    if route_coords:
        folium.PolyLine(route_coords, color="#27ae60", weight=5, opacity=0.9, tooltip="Reconstructed Trajectory").add_to(m)

    # Markers for Start and End
    folium.Marker(coords[0], popup="Trip Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker(coords[-1], popup="Trip Destination", icon=folium.Icon(color="red", icon="stop")).add_to(m)
else:
    # Default map centered on Darmstadt
    m = folium.Map(location=[49.858, 8.582], zoom_start=14, tiles="cartodbpositron")
    folium.Marker([49.858, 8.582], tooltip="Darmstadt Test Network").add_to(m)

try:
    st_folium(m, width=1100, height=550)
except Exception:
    st.info("Interactive map ready. Run `pip install streamlit-folium` for full inline rendering.")
