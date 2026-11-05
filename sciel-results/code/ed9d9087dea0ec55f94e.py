"""Combined non-stationary linear contextual bandit: robust adaptive LinUCB."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit using:
      - Per-arm ridge regression with exponential forgetting (decay).
      - Cholesky-based inversion for numerical stability and speed.
      - Adaptive beta (exploration) based on local reward variance and horizon.
      - A lightweight time-deadline guard to ensure we finish in time.
      - Direct, closed-form updates (no iterative solvers in the hot path).
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.dim = dim
        self.n_arms = n_arms

        # Base decay factor; we will adapt it slightly.
        self.base_decay = 0.99
        self.decay = self.base_decay

        # Regularization
        self.ridge = 1.0

        # Per-arm state
        self.A = [np.eye(dim) * self.ridge for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]

        # Pre-compute inverse lazily; store flag to know if dirty
        self.A_inv = [np.linalg.inv(self.A[i]) for i in range(n_arms)]
        self.dirty = [False] * n_arms

        # Tracking for adaptive exploration
        self.recent_rewards = []
        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.t = 0

        # Deadline guard
        self.start_time = time.time()
        self.deadline = self.start_time + 115  # 5s safety margin out of 120s

        # Exploration scale
        self.base_beta = 1.0
        self.beta = self.base_beta

    def _check_deadline(self):
        """Return True if we are close to the deadline."""
        if time.time() > self.deadline:
            return True
        return False

    def _update_adaptive_params(self):
        """Update decay and beta based on observed reward statistics."""
        # Keep a rolling window of rewards
        if len(self.recent_rewards) > 200:
            self.recent_rewards = self.recent_rewards[-200:]

        if len(self.recent_rewards) >= 10:
            rewards = np.array(self.recent_rewards)
            self.reward_mean = np.mean(rewards)
            self.reward_var = max(np.var(rewards), 1e-6)
        else:
            self.reward_mean = 0.0
            self.reward_var = 1.0

        # Adapt decay: if variance is high, we might want to adapt faster (lower decay)
        # If variance is low, we can be more stable (higher decay)
        # Map variance to decay in [0.95, 0.999]
        # Higher variance -> lower decay (more forgetting)
        var_scale = min(self.reward_var / 5.0, 1.0)  # cap influence
        self.decay = 0.999 - 0.04 * var_scale  # range [0.959, 0.999]

        # Adapt beta: increase exploration if variance is high or early on
        time_factor = 1.0 / (1.0 + self.t * 0.01)  # decays over time
        var_factor = 1.0 + np.sqrt(self.reward_var) * 0.5
        self.beta = self.base_beta * var_factor * (1.0 + time_factor)

    def _get_theta_and_cov(self, arm):
        """Return theta and inverse covariance for a given arm, updating if dirty."""
        if self.dirty[arm]:
            # Use Cholesky for better numerical stability
            try:
                L = np.linalg.cholesky(self.A[arm])
                # A_inv = inv(L^T) @ inv(L)
                self.A_inv[arm] = np.linalg.inv(L.T) @ np.linalg.inv(L)
            except np.linalg.LinAlgError:
                # Fallback to standard inverse if Cholesky fails
                self.A_inv[arm] = np.linalg.inv(self.A[arm])
            self.dirty[arm] = False
        theta = self.A_inv[arm] @ self.b[arm]
        return theta, self.A_inv[arm]

    def select(self, context):
        # Convert context to numpy array for consistent operations
        x = np.asarray(context, dtype=np.float64).flatten()

        # Check deadline; if close, use a deterministic greedy choice based on current estimates
        if self._check_deadline():
            # Greedy fallback: pick arm with highest estimated mean
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                theta, _ = self._get_theta_and_cov(a)
                score = theta @ x
                if score > best_score:
                    best_score = score
                    best_arm = a
            return int(best_arm)

        # Normal selection with UCB
        scores = np.zeros(self.n)
        for a in range(self.n):
            theta, A_inv = self._get_theta_and_cov(a)
            mean = theta @ x
            variance = x @ A_inv @ x
            # Use adaptive beta
            scores[a] = mean + self.beta * np.sqrt(max(variance, 0.0))

        # Add small random tie-breaker to avoid systematic bias
        scores += np.random.normal(0, 1e-6, size=self.n)

        best_arm = int(np.argmax(scores))
        self.t += 1
        return best_arm

    def update(self, context, arm, reward):
        x = np.asarray(context, dtype=np.float64).flatten()
        r = float(reward)

        # Store reward for adaptive parameter tuning
        self.recent_rewards.append(r)

        # Update per-arm statistics with forgetting
        decay = self.decay
        self.A[arm] = decay * self.A[arm] + np.outer(x, x)
        self.b[arm] = decay * self.b[arm] + r * x
        self.dirty[arm] = True

        # Adaptively update decay and beta
        self._update_adaptive_params()
# EVOLVE-BLOCK-END
