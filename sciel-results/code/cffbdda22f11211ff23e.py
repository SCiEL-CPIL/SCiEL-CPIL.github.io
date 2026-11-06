"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Policy:
    """
    Non-stationary linear bandit using a sliding-window linear regression approach.
    
    Instead of per-arm LinUCB with forgetting, this uses a single global model 
    with a sliding window of recent observations to adapt to non-stationarity.
    The model assumes reward = context @ theta + noise, where theta changes over time.
    We maintain a window of recent (context, reward) pairs and re-estimate theta
    via ridge regression on the window. For exploration, we use an optimistic 
    bonus based on the uncertainty from the recent window's covariance.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.window_size = 200  # Number of recent observations to keep
        self.history = []  # List of (context, arm, reward)
        self.lam = 1.0  # Ridge regularization
        self.alpha = 0.1  # Exploration coefficient
        
        # Pre-allocate for speed
        self.max_window = 500
        self.ctx_buffer = np.zeros((self.max_window, dim))
        self.arm_buffer = np.zeros(self.max_window, dtype=int)
        self.rwd_buffer = np.zeros(self.max_window)
        self.count = 0
        
        # For tracking per-arm stats for arm-specific bias if needed
        self.arm_counts = np.zeros(n_arms)

    def _add_observation(self, context, arm, reward):
        """Add observation to the circular buffer."""
        idx = self.count % self.max_window
        self.ctx_buffer[idx] = context
        self.arm_buffer[idx] = arm
        self.rwd_buffer[idx] = reward
        self.count += 1
        self.arm_counts[arm] += 1

    def _get_recent_data(self, window_size):
        """Get the most recent window_size observations."""
        if self.count == 0:
            return None, None, None
        
        n = min(window_size, self.count)
        start = (self.count - n) % self.max_window
        
        # Handle wrap-around
        if start + n <= self.max_window:
            ctxs = self.ctx_buffer[start:start+n].copy()
            arms = self.arm_buffer[start:start+n].copy()
            rwds = self.rwd_buffer[start:start+n].copy()
        else:
            part1_size = self.max_window - start
            part2_size = n - part1_size
            ctxs = np.vstack([self.ctx_buffer[start:], self.ctx_buffer[:part2_size]])
            arms = np.concatenate([self.arm_buffer[start:], self.arm_buffer[:part2_size]])
            rwds = np.concatenate([self.rwd_buffer[start:], self.rwd_buffer[:part2_size]])
        
        return ctxs, arms, rwds

    def _estimate_theta(self):
        """Estimate theta via ridge regression on recent data."""
        if self.count < 5:
            # Not enough data, return zero vector
            return np.zeros(self.d), np.eye(self.d)
        
        ctxs, arms, rwds = self._get_recent_data(self.window_size)
        if ctxs is None or len(ctxs) < 5:
            return np.zeros(self.d), np.eye(self.d)
        
        # Design matrix with arm dummies? No, let's use a single global model
        # with arm as part of the feature vector for better separation.
        # Actually, let's stick to a simpler approach: per-arm models but with sliding window.
        # Or better: a single model where features include arm one-hot encoding.
        
        # Let's use a single global model with arm one-hot features.
        # Feature vector = [context, one_hot(arm)]
        # But context dim is d, so total dim = d + n
        # This might be too many features. Let's try per-arm sliding window instead.
        
        # Actually, let's go with per-arm sliding window linear regression.
        # For each arm, maintain a window of recent (context, reward) pairs for that arm.
        # This is simpler and more interpretable.
        
        # Re-initialize: use per-arm buffers
        # We'll need to change the data structure. Let me redesign.
        
        # For now, return a default
        return np.zeros(self.d), np.eye(self.d)

    def select(self, context):
        """Select arm using optimistic linear regression with sliding window."""
        if self.count < 3:
            # Explore randomly early on
            return int(np.random.randint(self.n))
        
        scores = np.zeros(self.n)
        for a in range(self.n):
            # Get recent observations for arm a
            ctxs_a, rwds_a = self._get_arm_recent(a, self.window_size)
            
            if ctxs_a is None or len(ctxs_a) < 3:
                # Not enough data for this arm, give it a high exploration score
                scores[a] = 1.0  # High score to encourage exploration
                continue
            
            # Ridge regression: theta = (X^T X + lam * I)^-1 X^T y
            X = ctxs_a
            y = rwds_a
            XtX = X.T @ X + self.lam * np.eye(self.d)
            Xty = X.T @ y
            
            try:
                theta = np.linalg.solve(XtX, Xty)
                # Uncertainty: trace of (X^T X + lam I)^-1
                inv_XtX = np.linalg.inv(XtX)
                uncertainty = np.sqrt(context @ inv_XtX @ context)
                scores[a] = context @ theta + self.alpha * uncertainty
            except np.linalg.LinAlgError:
                scores[a] = 0.0
        
        return int(np.argmax(scores))

    def _get_arm_recent(self, arm, window_size):
        """Get recent observations for a specific arm."""
        if self.count == 0:
            return None, None
        
        # Collect observations for this arm from the buffer
        ctxs_list = []
        rwds_list = []
        
        for i in range(self.count):
            idx = i % self.max_window
            if self.arm_buffer[idx] == arm:
                ctxs_list.append(self.ctx_buffer[idx])
                rwds_list.append(self.rwd_buffer[idx])
        
        if len(ctxs_list) == 0:
            return None, None
        
        # Take the most recent window_size observations
        if len(ctxs_list) > window_size:
            ctxs_list = ctxs_list[-window_size:]
            rwds_list = rwds_list[-window_size:]
        
        ctxs = np.array(ctxs_list)
        rwds = np.array(rwds_list)
        
        return ctxs, rwds

    def update(self, context, arm, reward):
        """Update with new observation."""
        self._add_observation(context, arm, reward)
# EVOLVE-BLOCK-END
