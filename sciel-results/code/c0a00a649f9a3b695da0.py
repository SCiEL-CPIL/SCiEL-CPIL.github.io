"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

class Policy:
    """
    Non-stationary linear contextual bandit using Sliding Window Linear Regression 
    with Thompson Sampling and Variance-Adaptive Exploration.
    
    Strategy:
    1. Sliding Window: Strictly maintains only the last W samples for each arm. 
       This provides a natural, robust forgetting mechanism that adapts to 
       non-stationarity without heuristic thresholds.
    2. Thompson Sampling: Samples from the posterior to balance exploration/exploitation.
    3. Variance-Adaptive Exploration: Adjusts exploration noise based on the 
       variance of recent rewards for the selected arm. High variance (indicating 
       potential non-stationarity or high uncertainty) increases exploration.
    4. Deadline Guard: Ensures completion within the 150s limit.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        
        # Sliding window size: 100 is a common heuristic for non-stationary bandits.
        # It balances responsiveness to change vs. noise in estimation.
        self.window_size = 100
        
        # Store (context, reward) pairs for each arm
        self.windows = [deque(maxlen=self.window_size) for _ in range(n_arms)]
        
        # Pre-allocate matrices for speed (avoid repeated allocation in loop)
        self.A = [np.eye(dim) * 0.1 for _ in range(n_arms)]  # Small regularization
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # EWMA of reward variance for adaptive exploration
        self.reward_variances = [1.0] * n_arms
        self.last_rewards = [0.0] * n_arms
        self.var_ewma_alpha = 0.1  # Smoothing factor for variance EWMA
        
        # Time management
        self.start_time = time.time()
        self.deadline = self.start_time + 145.0  # 5s safety margin

    def _update_matrices(self, arm):
        """Recompute A and b from the sliding window. 
        For dim <= 10 and window <= 100, this is fast enough.
        A = X^T X + lambda * I
        b = X^T y
        """
        if len(self.windows[arm]) == 0:
            return
            
        # Extract data
        data = list(self.windows[arm])
        X = np.array([d[0] for d in data])  # (N, d)
        y = np.array([d[1] for d in data])  # (N,)
        
        # Compute normal equations
        # Add small regularization for numerical stability
        self.A[arm] = X.T @ X + 0.1 * np.eye(self.d)
        self.b[arm] = X.T @ y

    def _get_exploration_scale(self, arm):
        """
        Determine exploration scale based on recent reward variance.
        Higher variance -> more exploration.
        """
        # Base exploration is 1.0
        # Scale by sqrt(variance) but cap it to prevent extreme exploration
        # If variance is low, explore less; if high, explore more.
        var = self.reward_variances[arm]
        # Heuristic: scale between 0.5 and 2.0 based on variance
        # Normalize variance roughly: if var > 1, scale up; if var < 1, scale down
        # Using a soft clip
        scale = np.sqrt(var)
        scale = np.clip(scale, 0.5, 2.0)
        return scale

    def select(self, context):
        # Deadline check: if we are running out of time, fall back to a simple greedy
        if time.time() > self.deadline:
            # Simple greedy: pick arm with highest mean reward so far
            best_arm = 0
            best_mean = -np.inf
            for a in range(self.n):
                if len(self.windows[a]) > 0:
                    rewards = [d[1] for d in self.windows[a]]
                    mean_r = np.mean(rewards)
                    if mean_r > best_mean:
                        best_mean = mean_r
                        best_arm = a
            return best_arm

        context = np.asarray(context, dtype=float)
        scores = []
        
        for a in range(self.n):
            # Update matrices if new data was added (this is called in update, 
            # but we ensure A/b are current. For efficiency, we could cache, 
            # but recomputing from deque is O(W*d^2) which is fine for W=100, d~5-10)
            # Actually, updating in `update` is better. Here we just use current A/b.
            
            try:
                # Compute posterior mean and variance
                # theta_hat = A^{-1} b
                # Var(theta) = A^{-1} (for standard TS with unit reward variance)
                
                # Use solve for better numerical stability and speed
                theta_hat = np.linalg.solve(self.A[a], self.b[a])
                
                # Cholesky decomposition for sampling
                # Add small jitter if needed
                cov = self.A[a].copy()
                try:
                    L = np.linalg.cholesky(cov)
                except np.linalg.LinAlgError:
                    cov += 1e-6 * np.eye(self.d)
                    L = np.linalg.cholesky(cov)
                
                # Sample from N(theta_hat, A^{-1})
                # Note: In standard LinUCB/TS, the covariance is A^{-1}.
                # We need the inverse of A. We can get it from Cholesky of A^{-1} or solve.
                # Easier: sample z ~ N(0, I), then theta_sample = theta_hat + A^{-1/2} z
                # Since L is Cholesky of A, A = L L^T. A^{-1} = (L L^T)^{-1} = L^{-T} L^{-1}.
                # So A^{-1/2} = L^{-T} (symmetric square root).
                # We can solve L y = z, then A^{-1/2} z = y? No.
                # Let's just compute the inverse of A for simplicity, as d is small.
                # Or better: sample using the fact that if u ~ N(0, I), then A^{-1} u is not correct.
                # Correct: theta_sample = theta_hat + L_inv_T @ z, where L_inv_T is such that L_inv_T @ L_inv_T^T = A^{-1}.
                # Since A = L L^T, A^{-1} = L^{-T} L^{-1}.
                # So we need to solve L^T v = z, then L w = v? 
                # Actually, standard way: 
                # 1. Compute inv_A = np.linalg.inv(A)
                # 2. L_inv = np.linalg.cholesky(inv_A)
                # 3. theta_sample = theta_hat + L_inv @ z
                
                # Let's do it the robust way:
                inv_A = np.linalg.inv(self.A[a])
                try:
                    L_inv = np.linalg.cholesky(inv_A)
                except np.linalg.LinAlgError:
                    inv_A += 1e-6 * np.eye(self.d)
                    L_inv = np.linalg.cholesky(inv_A)
                
                z = np.random.randn(self.d)
                base_sample = theta_hat + L_inv @ z
                
                # Adaptive exploration scale
                scale = self._get_exploration_scale(a)
                
                # Scale the deviation from the mean
                # theta_sample = theta_hat + scale * (base_sample - theta_hat)
                theta_sample = theta_hat + scale * (base_sample - theta_hat)
                
                scores.append(theta_sample @ context)
                
            except np.linalg.LinAlgError:
                # Fallback: use mean only
                scores.append(self.b[a] @ context) # Crude fallback
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=float)
        
        # Add to sliding window
        self.windows[arm].append((context, reward))
        
        # Update variance EWMA
        # Calculate instantaneous variance proxy: (reward - last_reward)^2
        # Or better: maintain a running mean and variance? 
        # For simplicity, use the difference from the mean of the last few rewards.
        # Let's use a simple EWMA of squared differences from the moving average.
        # For now, a simple proxy:
        diff = reward - self.last_rewards[arm]
        sq_diff = diff * diff
        self.reward_variances[arm] = (1 - self.var_ewma_alpha) * self.reward_variances[arm] + self.var_ewma_alpha * sq_diff
        self.last_rewards[arm] = reward
        
        # Update matrices
        # For efficiency, we can update A and b incrementally:
        # A = A - beta * (old_context @ old_context^T) + new_context @ new_context^T
        # But since we use a deque, we don't easily know the "old" context if the window is full.
        # Given small d and W, recomputing from scratch is O(W*d^2) which is ~100*25=2500 ops.
        # This is fast enough.
        self._update_matrices(arm)
        
        self.t += 1
# EVOLVE-BLOCK-END
