# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Non-stationary Linear Contextual Bandit using Thompson Sampling
    with a sliding window of recent observations per arm.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0
        
        # Sliding window size for Thompson Sampling
        self.window_size = 60
        
        # Per-arm storage of recent (context, reward) pairs
        self.history = {a: [] for a in range(n_arms)}
        
        # Pre-allocated buffers for speed
        self._X_buf = np.zeros((self.window_size, dim))
        self._y_buf = np.zeros(self.window_size)
        
        # Regularization
        self.reg = 1.0
        
        # Exploration scale (std multiplier for Thompson Sampling)
        self.exploration_scale = 1.0
        
        # Tracking for deadline and best arm
        self.best_arm = 0
        self.arm_recent_rewards = {a: [] for a in range(n_arms)}
        self.arm_history_len = 30
        
        # Noise variance estimate
        self.noise_var = 1.0
        self.recent_rewards = []
        self.recent_window = 50

    def _within_deadline(self):
        return time.time() < self.deadline

    def _get_theta_and_cov(self, arm):
        """Compute posterior mean and covariance for a given arm using sliding window."""
        hist = self.history[arm]
        n_obs = len(hist)
        
        if n_obs == 0:
            # Prior: zero mean, identity covariance (scaled)
            mean = np.zeros(self.d)
            cov = np.eye(self.d) * 10.0
            return mean, cov
        
        # Fill buffers
        start = max(0, n_obs - self.window_size)
        count = n_obs - start
        
        for i in range(count):
            ctx, rew = hist[start + i]
            self._X_buf[i] = ctx
            self._y_buf[i] = rew
        
        X = self._X_buf[:count]
        y = self._y_buf[:count]
        
        # Bayesian Linear Regression:
        # Posterior mean = (X^T X + reg*I)^{-1} X^T y
        # Posterior cov = (X^T X + reg*I)^{-1} * noise_var
        # For Thompson Sampling, we sample from N(mean, cov)
        
        XtX = X.T @ X
        Xty = X.T @ y
        
        # Add regularization
        XtX_reg = XtX + self.reg * np.eye(self.d)
        
        try:
            # Compute inverse once for both mean and covariance
            XtX_inv = np.linalg.inv(XtX_reg)
            mean = XtX_inv @ Xty
            cov = XtX_inv * self.noise_var
        except np.linalg.LinAlgError:
            # Fallback
            mean = np.zeros(self.d)
            cov = np.eye(self.d) * 10.0
            
        return mean, cov

    def select(self, context):
        self.t += 1
        
        # Deadline guard
        if not self._within_deadline():
            return self.best_arm
        
        # Pure exploration in the very first few rounds
        if self.t <= 3:
            return int(np.random.randint(self.n))
        
        # Thompson Sampling: sample a theta for each arm and pick the best
        best_arm = 0
        best_val = -np.inf
        
        for a in range(self.n):
            mean, cov = self._get_theta_and_cov(a)
            
            # Sample from N(mean, cov)
            # Use Cholesky decomposition for efficiency if possible, else eig
            try:
                L = np.linalg.cholesky(cov)
                z = np.random.randn(self.d)
                theta_sample = mean + L @ z
            except np.linalg.LinAlgError:
                # Fallback: use mean only (greedy)
                theta_sample = mean
            
            val = np.dot(theta_sample, context)
            if val > best_val:
                best_val = val
                best_arm = a
        
        return best_arm

    def update(self, context, arm, reward):
        # Deadline guard
        if not self._within_deadline():
            return
        
        # Add to history
        self.history[arm].append((context, reward))
        if len(self.history[arm]) > self.window_size * 2:
            # Trim to keep memory bounded, keep most recent window_size * 2
            self.history[arm] = self.history[arm][-self.window_size * 2:]
        
        # Update best arm tracking
        self.arm_recent_rewards[arm].append(reward)
        if len(self.arm_recent_rewards[arm]) > self.arm_history_len:
            self.arm_recent_rewards[arm].pop(0)
        
        arm_avgs = {}
        for a in range(self.n):
            if self.arm_recent_rewards[a]:
                arm_avgs[a] = np.mean(self.arm_recent_rewards[a])
            else:
                arm_avgs[a] = -np.inf
        if arm_avgs:
            self.best_arm = max(arm_avgs, key=arm_avgs.get)
        
        # Update noise variance estimate
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > self.recent_window:
            self.recent_rewards.pop(0)
        
        if len(self.recent_rewards) >= 10:
            self.noise_var = max(0.1, np.var(self.recent_rewards))
# EVOLVE-BLOCK-END
