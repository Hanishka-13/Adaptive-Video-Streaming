import numpy as np


class LiveStreamingEnv:

    def __init__(self):
        self.max_buffer = 10.0
        self.min_bitrate = 300.0
        self.max_bitrate = 5000.0
        self.target_latency = 3.0
        self.reset()

    def reset(self):
        self.buffer = 5.0
        self.latency = 2.0
        self.prev_bitrate = 1000.0

        self.base_bandwidth = np.random.uniform(1500, 3000)

        self.current_throughput = self.base_bandwidth
        self.avg_throughput = self.base_bandwidth
        self.last_download_time = 0.0

        self.throughput_history = []
        self.time = 0

        return self._get_state()

    def step(self, action):
        bitrate, playback_speed = action

        bitrate = np.clip(bitrate, self.min_bitrate, self.max_bitrate)
        playback_speed = np.clip(playback_speed, 0.95, 1.05)

        # -----------------------------
        # Smooth Bandwidth Model
        # -----------------------------
        self.base_bandwidth += np.random.normal(0, 80)

        if np.random.rand() < 0.05:
            self.base_bandwidth *= np.random.uniform(0.6, 0.85)

        self.base_bandwidth = np.clip(self.base_bandwidth, 500, 5000)
        throughput = self.base_bandwidth
        self.current_throughput = throughput

        self.throughput_history.append(throughput)
        if len(self.throughput_history) > 5:
            self.throughput_history.pop(0)

        self.avg_throughput = np.mean(self.throughput_history)

        # -----------------------------
        # Download
        # -----------------------------
        segment_duration = 1.0
        download_time = bitrate / throughput
        self.last_download_time = download_time

        # -----------------------------
        # Buffer Update
        # -----------------------------
        self.buffer += segment_duration - download_time

        rebuffer = 0.0
        if self.buffer < 0:
            rebuffer = abs(self.buffer)
            self.buffer = 0.0

        self.buffer = min(self.buffer, self.max_buffer)

        # -----------------------------
        # Latency Update
        # -----------------------------
        self.latency += download_time - playback_speed
        self.latency = max(0.1, self.latency)

        # -----------------------------
        # Reward Design (Balanced)
        # -----------------------------

        # 1️⃣ Quality reward (log scale prevents max bias)
        quality_reward = np.log(1 + bitrate / 1000)

        # 2️⃣ Buffer safety penalty
        buffer_penalty = 2.5 * (1 / (self.buffer + 0.5))

        # 3️⃣ Latency penalty
        latency_penalty = 2.0 * abs(self.latency - self.target_latency)

        # 4️⃣ Download overflow penalty
        overflow_penalty = 4.0 * max(0, download_time - segment_duration)

        # 5️⃣ Rebuffer penalty (strong)
        rebuffer_penalty = 6.0 * rebuffer

        reward = (
            quality_reward
            - buffer_penalty
            - latency_penalty
            - overflow_penalty
            - rebuffer_penalty
        )

        self.prev_bitrate = bitrate
        self.time += 1

        # -----------------------------
        # Termination
        # -----------------------------
        if self.latency > 15 or rebuffer > 2:
            done = True
        else:
            done = False

        return self._get_state(), reward, done, {}

    def _get_state(self):
        return np.array([
            self.current_throughput,
            self.avg_throughput,
            self.buffer,
            self.latency,
            self.prev_bitrate,
            self.last_download_time
        ], dtype=np.float32)