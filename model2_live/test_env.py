from live_stream_env import LiveStreamingEnv

env = LiveStreamingEnv()
state = env.reset()

print("Initial State:", state)

for i in range(5):
    action = [2000, 1.0]  # bitrate, playback speed
    state, reward, done, _ = env.step(action)
    
    print(f"\nStep {i+1}")
    print("State:", state)
    print("Reward:", reward)