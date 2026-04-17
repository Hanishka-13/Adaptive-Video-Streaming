import numpy as np

class DashEnv:
    def __init__(self):
        self.max_buffer = 30.0 # VOD has higher buffer limit
        self.min_bitrate = 300.0
        self.max_bitrate = 5000.0
        self.reset()
        
    def reset(self):
        self.buffer = 15.0
        self.prev_bitrate = 1000.0
        self.base_bandwidth = np.random.uniform(1500, 4000)
        self.time = 0
        return self._get_state()
        
    def step(self, action):
        # Action is normalized between -1 and 1
        bitrate = 300 + (action[0] + 1) / 2 * (5000 - 300)
        bitrate = np.clip(bitrate, self.min_bitrate, self.max_bitrate)
        
        # Bandwidth variation
        self.base_bandwidth += np.random.normal(0, 100)
        self.base_bandwidth = np.clip(self.base_bandwidth, 500, 5000)
        throughput = self.base_bandwidth
        
        segment_duration = 4.0 # DASH usually has longer segments
        download_time = (bitrate * segment_duration) / throughput
        
        self.buffer += segment_duration - download_time
        rebuffer = 0.0
        if self.buffer < 0:
            rebuffer = abs(self.buffer)
            self.buffer = 0.0
        self.buffer = min(self.buffer, self.max_buffer)
        
        # Reward calculation for VOD (DASH)
        qoe = np.log(1 + bitrate / 1000.0)
        smoothness_penalty = abs(bitrate - self.prev_bitrate) / 1000.0
        rebuffer_penalty = 10.0 * rebuffer
        reward = qoe - 0.5 * smoothness_penalty - rebuffer_penalty
        
        self.prev_bitrate = bitrate
        self.time += 1
        
        done = False
        if rebuffer > 10:
            done = True
            
        return self._get_state(), reward, done, {}
        
    def _get_state(self):
        # State: [bandwidth_ratio, buffer_ratio, prev_bitrate_ratio]
        return np.array([
            self.base_bandwidth / 5000.0,
            self.buffer / 30.0,
            self.prev_bitrate / 5000.0
        ], dtype=np.float32)
