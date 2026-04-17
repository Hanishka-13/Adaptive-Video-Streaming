import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="RL Adaptive Video Streaming")

st.title("📺 Adaptive Video Streaming Dashboard")
st.markdown("This dashboard tracks the performance of our RL models adapting video bitrate to dynamic network conditions.")

API_URL = "http://localhost:8000"

if "history" not in st.session_state:
    st.session_state.history = []

sidebar = st.sidebar
sidebar.header("Controls")

model_choice = sidebar.radio("Select Scenario:", ["live", "dash"], format_func=lambda x: "Live Streaming (SAC - Model 2)" if x == "live" else "DASH Normal Video (LSTM - Model 1)")

def reset_sim():
    try:
        requests.get(f"{API_URL}/reset/{model_choice}")
        st.session_state.history = []
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend.")

def run_step():
    try:
        res = requests.get(f"{API_URL}/step/{model_choice}")
        if res.status_code == 200:
            data = res.json()
            st.session_state.history.append(data)
            return data["done"]
        return False
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend.")
        return False

# ----- VIDEO SOURCE SECTION -----
st.subheader("1. Video Stream Source")

source_type = st.radio("Choose Video Source:", ["Simulate YouTube/Live Stream URL", "Upload Local Video File"], horizontal=True)

col_vid, col_desc = st.columns([1.5, 1])

with col_vid:
    if source_type == "Simulate YouTube/Live Stream URL":
        yt_url = st.text_input("Paste a YouTube or Live Stream URL:", "https://www.youtube.com/watch?v=jfKfPfyJRdk")  # Lofi girl live stream as default
        if yt_url:
            st.video(yt_url)
            
    else:
        uploaded_video = st.file_uploader("Choose a video file", type=['mp4', 'mov', 'avi'])
        if uploaded_video is not None:
            st.video(uploaded_video)

with col_desc:
    if source_type == "Simulate YouTube/Live Stream URL" and yt_url:
        st.success("✅ Connected to Live Stream")
        st.info("In a real-world integration (like Instagram Live or YouTube Live), the platform breaks the live video into small 1-second chunks, encoded at different qualities (e.g., 480p, 720p, 1080p). Our RL Model sits inside the client player. It observes the network, buffer, and latency, and constantly dictates which quality chunk to download next, and adjusts playback speed (e.g., 1.05x to catch up to real-time).")
    elif source_type == "Upload Local Video File" and uploaded_video is not None:
        st.success("✅ Connected to VOD File")
        st.info("The RL agent is predicting optimal streaming bits as if this file were being streamed over DASH.")

# Sidebar controls
if sidebar.button("Reset Environment", type="secondary"):
    reset_sim()

num_steps = sidebar.slider("Steps to simulate", 1, 100, 10)
if sidebar.button("Run Simulation Step(s)", type="primary"):
    for _ in range(num_steps):
        done = run_step()
        if done:
            st.toast("Episode terminated early (limits hit).")
            break

# ----- SIMULATION CHARTS -----
st.subheader("2. Real-Time Telemetry & Agent Metrics")
if len(st.session_state.history) > 0:
    df = pd.DataFrame(st.session_state.history)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Network Throughput vs. Chosen Bitrate**")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(y=df["throughput"], mode='lines', name='Available Bandwidth (kbps)', line=dict(color='#3b82f6', width=2)))
        fig1.add_trace(go.Scatter(y=df["chosen_bitrate"], mode='lines', name='Chosen Bitrate (kbps)', line=dict(color='#10b981', width=3, dash='dot')))
        st.plotly_chart(fig1, use_container_width=True)

        if model_choice == "live":
            st.markdown("**Latency**")
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(y=df["latency"], mode='lines', name='Latency (s)', line=dict(color='#ef4444', width=2)))
            fig3.add_hline(y=3.0, line_dash="dash", line_color="#f59e0b", annotation_text="Target Latency")
            st.plotly_chart(fig3, use_container_width=True)

    with c2:
        st.markdown("**Player Buffer Level**")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(y=df["buffer"], mode='lines', name='Buffer (s)', fill='tozeroy', line=dict(color='#8b5cf6')))
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("**Agent Reward Strategy**")
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(y=df["reward"], mode='lines', name='Reward', line=dict(color='#f59e0b')))
        st.plotly_chart(fig4, use_container_width=True)
else:
    st.info("No network data yet. Start simulation from the sidebar.")
