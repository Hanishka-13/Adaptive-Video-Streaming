import torch
import numpy as np

from live_stream_env import LiveStreamingEnv
from sac.actor import Actor


# ---------------------------------
# Device
# ---------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------
# Load Environment
# ---------------------------------
env = LiveStreamingEnv()

state_dim = 6
action_dim = 2


# ---------------------------------
# Load Trained Actor
# ---------------------------------
actor = Actor(state_dim, action_dim).to(device)
actor.load_state_dict(torch.load("ll_gabr_actor.pt", map_location=device))
actor.eval()


# ---------------------------------
# State Normalization (must match train.py!)
# ---------------------------------
def normalize_state(state):
    state = state.copy()
    state[0] /= 5000.0   # throughput
    state[1] /= 5000.0   # avg throughput
    state[2] /= 10.0     # buffer
    state[3] /= 20.0     # latency
    state[4] /= 5000.0   # prev bitrate
    state[5] /= 5.0      # download time
    return state


# ---------------------------------
# Action Scaling
# ---------------------------------
def scale_action(action):
    bitrate = 300 + (action[0] + 1) / 2 * (5000 - 300)
    playback_speed = 1.0 + 0.05 * action[1]
    return bitrate, playback_speed


# ---------------------------------
# Deterministic Action (Mean Only)
# ---------------------------------
def select_action(state):
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        mean, _ = actor(state_tensor)
    action = torch.tanh(mean).cpu().numpy()[0]
    return action


# ---------------------------------
# Evaluation Loop
# ---------------------------------
raw_state = env.reset()
state = normalize_state(raw_state)

print("\n--- Evaluation Start ---\n")

total_reward = 0

for step in range(50):

    action = select_action(state)
    bitrate, speed = scale_action(action)

    raw_next_state, reward, done, _ = env.step([bitrate, speed])
    next_state = normalize_state(raw_next_state)

    total_reward += reward

    print(f"Step {step+1}")
    print(f"  Throughput:     {raw_next_state[0]:.2f} kbps")
    print(f"  Buffer:         {raw_next_state[2]:.2f} s")
    print(f"  Latency:        {raw_next_state[3]:.2f} s")
    print(f"  Chosen Bitrate: {bitrate:.2f} kbps")
    print(f"  Playback Speed: {speed:.3f}x")
    print(f"  Reward:         {reward:.3f}")
    print("-" * 40)

    if done:
        print("Episode ended early (latency/rebuffer limit hit).")
        break

    state = next_state

print(f"\nTotal Reward: {total_reward:.2f}")