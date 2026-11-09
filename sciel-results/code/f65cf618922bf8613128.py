"""Non-stationary linear contextual bandit policy using Thompson Sampling with exponential forgetting."""
import numpy as np
import time


class Policy:
    """
    Linear Thompson Sampling with exponential forgetting factor.
    
    Uses a posterior distribution over the weight vector for each arm.
    The forgetting factor allows the policy to adapt to non-stationary environments
    by gradually reducing the influence of older observations.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.dim = dim
        
        # Initialize inverse covariance (precision) and mean for each arm
        # Using a small regularization for numerical stability
        self.reg = 1e-3
        self.A_inv = [np.eye(dim) / self.reg for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Forgetting factor: controls how quickly old data is forgotten
        # 0.98 provides a good balance for non-stationary settings
        self.forgetting = 0.98
        
        # Noise variance estimate for Thompson Sampling
        self.sigma_sq = 1.0
        
        # Track time for deadline guard
        self._start_time = time.time()
        self._deadline = 140.0  # 140s to be safe within 150s limit
        
        # Track rewards for potential fallback
        self._reward_sum = [0.0 for _ in range(n_arms)]
        self._reward_count = [0.0 for _ in range(n_arms)]

    def select(self, context):
        # Deadline guard: if we're running out of time, use a simple heuristic
        if time.time() > self._deadline:
            # Fallback: pick arm with highest average reward so far
            avg_rewards = [
                self._reward_sum[a] / max(self._reward_count[a], 1.0)
                for a in range(self.n)
            ]
            return int(np.argmax(avg_rewards))
        
        context = np.asarray(context, dtype=np.float64)
        
        # Thompson Sampling: sample from posterior for each arm
        samples = []
        for a in range(self.n):
            # Posterior mean: A_inv @ b
            theta_mean = self.A_inv[a] @ self.b[a]
            
            # Posterior covariance: A_inv * sigma_sq
            cov = self.A_inv[a] * self.sigma_sq
            
            # Sample theta from N(theta_mean, cov)
            # Use Cholesky decomposition for efficient sampling
            try:
                L = np.linalg.cholesky(cov + 1e-10 * np.eye(self.d))
                z = np.random.randn(self.d)
                theta_sample = theta_mean + L @ z
            except np.linalg.LinAlgError:
                # Fallback if Cholesky fails
                theta_sample = theta_mean
            
            # Score: sampled reward for this arm
            score = theta_sample @ context
            samples.append(score)
        
        return int(np.argmax(samples))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        
        # Track rewards for fallback
        self._reward_sum[arm] += reward
        self._reward_count[arm] += 1.0
        
        # Exponential forgetting: decay old information
        self.A_inv[arm] = self.A_inv[arm] / self.forgetting
        self.b[arm] = self.b[arm] * self.forgetting
        
        # Update with new observation using Sherman-Morrison-Woodbury
        # A_inv_new = A_inv - (A_inv @ x @ x^T @ A_inv) / (sigma_sq + x^T @ A_inv @ x)
        Ax = self.A_inv[arm] @ context
        denom = self.sigma_sq + context @ Ax
        
        if abs(denom) > 1e-12:
            update = np.outer(Ax, Ax) / denom
            self.A_inv[arm] = self.A_inv[arm] - update
        
        # Update mean: b_new = b + (reward - x^T @ theta_old) * A_inv @ x / sigma_sq
        # Simpler: b_new = b + (reward - context @ (A_inv @ b)) * A_inv @ x
        # But since we're doing TS with exponential forgetting, we can just do:
        # b_new = b + (reward - context @ theta_old) * A_inv @ x
        # where theta_old = A_inv @ b before update
        
        # Actually, for Thompson Sampling with known noise variance:
        # The posterior update for the mean is:
        # theta_new = theta_old + A_inv @ x * (reward - x^T @ theta_old) / sigma_sq
        # In terms of A_inv and b:
        # b_new = b + (reward - x^T @ (A_inv @ b)) * A_inv @ x
        
        theta_old = self.A_inv[arm] @ self.b[arm]
        residual = reward - context @ theta_old
        self.b[arm] = self.b[arm] + residual * Ax / self.sigma_sq
        
        # Symmetrize A_inv to reduce numerical drift
        self.A_inv[arm] = (self.A_inv[arm] + self.A_inv[arm].T) / 2.0
