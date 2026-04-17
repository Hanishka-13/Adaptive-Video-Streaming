import torch
import numpy as np
import matplotlib.pyplot as plt

from live_stream_env import LiveStreamingEnv
from sac.replay_buffer import ReplayBuffer
from sac.sac_agent import SACAgent


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

env = LiveStreamingEnv()

state_dim = 6
action_dim = 2

agent = SACAgent(state_dim, action_dim, device)
replay_buffer = ReplayBuffer(100000, state_dim, action_dim)

episodes = 300
steps_per_episode = 150
batch_size = 64

reward_history = []


# ---------------------------------
# State Normalization
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


def scale_action(action):
    bitrate = 300 + (action[0] + 1) / 2 * (5000 - 300)
    playback_speed = 1.0 + 0.05 * action[1]
    return np.array([bitrate, playback_speed])


for episode in range(episodes):

    state = env.reset()
    state = normalize_state(state)

    episode_reward = 0

    for step in range(steps_per_episode):

        action = agent.select_action(state)
        scaled_action = scale_action(action)

        next_state, reward, done, _ = env.step(scaled_action)
        next_state = normalize_state(next_state)

        replay_buffer.add(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        if replay_buffer.size > batch_size:
            agent.update(replay_buffer, batch_size)

        if done:
            break

    reward_history.append(episode_reward)

    if (episode + 1) % 10 == 0:
        print(f"Episode {episode+1}/{episodes} | Reward: {episode_reward:.2f} | Alpha: {agent.alpha:.4f}")


torch.save(agent.actor.state_dict(), "ll_gabr_actor.pt")
print("Model saved.")

plt.figure(figsize=(10, 5))
plt.plot(reward_history, alpha=0.4, label="Raw")

window = 20
if len(reward_history) >= window:
    smoothed = np.convolve(reward_history, np.ones(window)/window, mode='valid')
    plt.plot(range(window-1, len(reward_history)), smoothed, linewidth=2, label="Smoothed")

plt.title("Training Reward (LL-GABR SAC)")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()
plt.tight_layout()
plt.savefig("training_curve.png")
plt.show()
print("Training curve saved.")