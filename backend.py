import torch
import numpy as np
from fastapi import FastAPI
import uvicorn

from model2_live.live_stream_env import LiveStreamingEnv
from model2_live.sac.actor import Actor as LiveActor

from model1_dash.dash_env import DashEnv
from model1_dash.lstm_agent import LSTMAgent

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model 2 (Live SAC) Setup ---
live_env = LiveStreamingEnv()
live_actor = LiveActor(6, 2).to(device)
live_actor.load_state_dict(torch.load("model2_live/live_model.pkl", map_location=device))
live_actor.eval()

def normalize_live_state(state):
    state = state.copy()
    state[0] /= 5000.0   # throughput
    state[1] /= 5000.0   # avg throughput
    state[2] /= 10.0     # buffer
    state[3] /= 20.0     # latency
    state[4] /= 5000.0   # prev bitrate
    state[5] /= 5.0      # download time
    return state

def scale_live_action(action):
    bitrate = 300 + (action[0] + 1) / 2 * (5000 - 300)
    playback_speed = 1.0 + 0.05 * action[1]
    return float(bitrate), float(playback_speed)

# --- Model 1 (DASH LSTM) Setup ---
dash_env = DashEnv()
dash_actor = LSTMAgent(3, 1).to(device)
try:
    dash_actor.load_state_dict(torch.load("model1_dash/dash_model.pkl", map_location=device))
except FileNotFoundError:
    pass
dash_actor.eval()

def scale_dash_action(action):
    bitrate = 300 + (action[0] + 1) / 2 * (5000 - 300)
    return float(bitrate)

# --- Globals ---
current_live_state = live_env.reset()
current_dash_state = dash_env.reset()
dash_hidden_state = None

@app.get("/reset/{model}")
def reset_env(model: str):
    global current_live_state, current_dash_state, dash_hidden_state
    if model == "live":
        current_live_state = live_env.reset()
        return {"message": "Live Environment reset successful", "state": current_live_state.tolist()}
    elif model == "dash":
        current_dash_state = dash_env.reset()
        dash_hidden_state = None
        return {"message": "DASH Environment reset successful", "state": current_dash_state.tolist()}
    return {"error": "Invalid model"}

@app.get("/step/{model}")
def step_env(model: str):
    global current_live_state, current_dash_state, dash_hidden_state
    
    if model == "live":
        state = normalize_live_state(current_live_state)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            mean, _ = live_actor(state_tensor)
        action = torch.tanh(mean).cpu().numpy()[0]
        
        bitrate, speed = scale_live_action(action)
        raw_next_state, reward, done, _ = live_env.step([bitrate, speed])
        current_live_state = raw_next_state

        return {
            "throughput": float(raw_next_state[0]),
            "buffer": float(raw_next_state[2]),
            "latency": float(raw_next_state[3]),
            "chosen_bitrate": float(bitrate),
            "playback_speed": float(speed),
            "reward": float(reward),
            "done": bool(done)
        }
        
    elif model == "dash":
        state = current_dash_state 
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device) 
        
        with torch.no_grad():
            action_tensor, dash_hidden_state = dash_actor(state_tensor, dash_hidden_state)
        action = action_tensor.cpu().numpy()[0]
        
        bitrate = scale_dash_action(action)
        raw_next_state, reward, done, _ = dash_env.step(action)
        current_dash_state = raw_next_state

        return {
            "throughput": float(raw_next_state[0] * 5000.0),
            "buffer": float(raw_next_state[1] * 30.0),
            "latency": 0.0, 
            "chosen_bitrate": float(bitrate),
            "playback_speed": 1.0,
            "reward": float(reward),
            "done": bool(done)
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
