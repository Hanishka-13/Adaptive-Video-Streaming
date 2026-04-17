# Adaptive Video Streaming using Reinforcement Learning 📺🚀

## 📌 Project Overview
Traditional video streaming systems use fixed rules, often leading to buffering or poor video quality. This project aims to revolutionize this by using **Deep Reinforcement Learning (RL)** to dynamically optimize video streaming quality in real-time based on network conditions.

The system natively acts as an intelligent client. Given network bandwidth, player buffer level, and latency, it mathematically predicts the optimal video bitrate chunks to download.

## 🏗 System Architecture

We implemented two separated RL models to target different real-world streaming constraints:

1. **Model 1: DASH (Normal Video / VoD)** 
   - **Algorithm:** Long Short-Term Memory (LSTM)
   - **Goal:** Focuses on smooth playback and stable, high quality without strict latency conditions.
   - **Status:** Integrated via `model1_dash/`

2. **Model 2: Live Streaming**
   - **Algorithm:** Soft Actor-Critic (SAC)
   - **Goal:** Targets absolute low latency, prioritizing real-time chunk synchronization while keeping buffering to a minimum.
   - **Status:** Integrated via `model2_live/`

## 💻 Tech Stack
- **AI/ML:** PyTorch (LSTM & SAC algorithm implementation)
- **Backend Environment:** FastAPI & Python
- **Frontend / Telemetry UI:** Streamlit & Plotly (Real-time dynamic telemetry)

## ⚙️ How to Run the Project locally

### 1. Start the Backend API
The FastAPI backend serves the `.pkl` ML models and exposes the streaming environment to the dashboard.
```bash
uvicorn backend:app --port 8000
```
*Your backend will be running at `http://localhost:8000`*

### 2. Start the Frontend Dashboard
The Streamlit frontend provides the interactive video player and chart visualizations.
```bash
streamlit run frontend.py --server.port 8501
```
*Your UI will open automatically at `http://localhost:8501`*

## 🧑‍🏫 How it works (for Viva / Presentation)
If evaluating this project for acadamia, note that the platform relies on **PyTorch Mathematical Simulations**. 

Because proprietary platforms like YouTube and Instagram do not provide external APIs to control their underlying streaming algorithms, researchers test ABR (Adaptive Bitrate) models via simulation. The provided Streamlit dashboard illustrates exactly how the Reinforcement Learning agent manages the **Bitrate**, **Buffer**, and **Latency** in real-time, side by side with a video simulation, proving the algorithm's validity.
