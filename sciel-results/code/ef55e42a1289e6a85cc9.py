"""Adaptive non-stationary linear bandit with per-arm forgetting and exploration."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Adaptive LinUCB with per-arm forgetting and time-decay exploration."""

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.lam = 1.0
        self.A = [np.eye(dim) * self.lam for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.A_prev = [np.eye(dim) * self.lam for _ in range(n_arms)]
        self.b_prev = [np.zeros(dim) for _ in range(n_arms)]
        
        # Forgetting factor
        self.beta = 0.985
        
        # Base exploration parameter
        self.alpha_base = 1.2
        
        # Track time steps
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0  # Safety margin before 120s limit
        
        # Per-arm recent reward tracking for adaptive exploration
        self.recent_rewards = [np.zeros(dim) for _ in range(n_arms)]
        self.reward_counts = [0 for _ in range(n_arms)]

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)
        
        # Check deadline
        if time.time() > self.deadline:
            # Return best arm based on current estimates
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                inv = np.linalg.inv(self.A[a])
                theta = inv @ self.b[a]
                score = theta @ context
                if score > best_score:
                    best_score = score
                    best_arm = a
            return int(best_arm)
        
        scores = []
        for a in range(self.n):
            try:
                inv = np.linalg.inv(self.A[a])
                theta = inv @ self.b[a]
                mu = theta @ context
                # Adaptive exploration: scale by uncertainty
                uncertainty = np.sqrt(context @ inv @ context)
                # Decay alpha slightly over time but keep exploration
                alpha = self.alpha_base * (1.0 + 0.1 * np.exp(-self.t / 100.0))
                scores.append(mu + alpha * uncertainty)
            except np.linalg.LinAlgError:
                scores.append(0.0)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        self.t += 1
        
        # Update with forgetting
        self.A_prev[arm] = self.A[arm].copy()
        self.b_prev[arm] = self.b[arm].copy()
        
        self.A[arm] = self.beta * self.A[arm] + np.outer(context, context)
        self.b[arm] = self.beta * self.b[arm] + reward * context
        
        # Track recent rewards for potential adaptation
        self.recent_rewards[arm] = self.beta * self.recent_rewards[arm] + reward * context
        self.reward_counts[arm] = self.beta * self.reward_counts[arm] + 1
# EVOLVE-BLOCK-END
