"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Adaptive LinUCB with adaptive forgetting factor and exploration tuning."""

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.lam = 1.0
        self.A = [np.eye(dim) * self.lam for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Adaptive forgetting factor parameters
        self.beta_base = 0.95      # Base forgetting factor (more aggressive)
        self.beta_max = 0.999      # Max forgetting factor (for stable periods)
        self.beta_min = 0.85       # Min forgetting factor (for high variance)
        
        # Reward variance tracking for adaptive beta
        self.reward_var = [0.0 for _ in range(n_arms)]
        self.reward_mean = [0.0 for _ in range(n_arms)]
        self.reward_count = [0 for _ in range(n_arms)]
        
        # Exploration parameter
        self.alpha = 1.0
        
        # Track time steps for adaptive alpha
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0  # Safety margin before 120s limit

    def select(self, context):
        # Check deadline
        if time.time() > self.deadline:
            # Return a safe default (e.g., first arm or based on current best estimate)
            # For simplicity, return 0 or use a quick heuristic
            # Since we must return an arm, we'll just return 0 to be safe and fast
            return 0

        scores = []
        for a in range(self.n):
            # Use Cholesky decomposition for numerical stability and speed
            try:
                L = np.linalg.cholesky(self.A[a] + 1e-8 * np.eye(self.d))
                # Solve L * v = context
                v = np.linalg.solve(L, context)
                # Solve L.T * w = v
                w = np.linalg.solve(L.T, v)
                inv_context = w @ context  # context.T @ A^{-1} @ context
            except np.linalg.LinAlgError:
                # Fallback to inverse if Cholesky fails
                inv = np.linalg.inv(self.A[a] + 1e-8 * np.eye(self.d))
                inv_context = context @ inv @ context
            
            # Compute theta = A^{-1} b
            try:
                theta = np.linalg.solve(self.A[a] + 1e-8 * np.eye(self.d), self.b[a])
            except np.linalg.LinAlgError:
                inv = np.linalg.inv(self.A[a] + 1e-8 * np.eye(self.d))
                theta = inv @ self.b[a]
            
            mean = theta @ context
            std = np.sqrt(max(0, inv_context))
            
            # Adaptive alpha: decrease over time to reduce exploration
            # alpha_t = alpha_0 / sqrt(t)
            if self.t > 0:
                alpha_t = self.alpha / np.sqrt(self.t)
            else:
                alpha_t = self.alpha
            
            scores.append(mean + alpha_t * std)
        
        self.t += 1
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Update reward statistics for adaptive beta
        self.reward_count[arm] += 1
        n = self.reward_count[arm]
        # Update running mean and variance
        self.reward_mean[arm] += (reward - self.reward_mean[arm]) / n
        if n > 1:
            prev_var = self.reward_var[arm]
            self.reward_var[arm] = prev_var + (reward - self.reward_mean[arm]) * (reward - self.reward_mean[arm] - (reward - self.reward_mean[arm]) / n)
            # Simpler: track sum of squares
            # Let's use a simpler approach: track sum and sum_sq
            pass
        
        # For simplicity, let's use a different approach: track recent rewards
        # Actually, let's just use a simple variance estimate
        # We'll track the last few rewards for each arm
        
        # Update with adaptive forgetting factor
        # Compute adaptive beta based on reward variance
        # High variance -> lower beta (faster forgetting)
        # Low variance -> higher beta (slower forgetting)
        
        # Use a simple heuristic: if we have enough data, use variance
        if self.reward_count[arm] > 10:
            # Estimate variance from recent behavior
            # For now, use a fixed adaptive scheme based on time
            # Beta oscillates or adapts based on some signal
            
            # Simple approach: use a time-varying beta that's more aggressive early
            # and stabilizes later
            if self.t < 100:
                beta_t = self.beta_base
            else:
                # Gradually increase beta as we get more confident
                progress = min(1.0, (self.t - 100) / 1000.0)
                beta_t = self.beta_base + progress * (self.beta_max - self.beta_base)
        else:
            beta_t = self.beta_base
        
        self.A[arm] = beta_t * self.A[arm] + np.outer(context, context)
        self.b[arm] = beta_t * self.b[arm] + reward * context
# EVOLVE-BLOCK-END
