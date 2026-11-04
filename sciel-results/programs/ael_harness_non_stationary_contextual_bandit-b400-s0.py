"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Adaptive non-stationary linear bandit with inverse-covariance tracking."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = 145.0  # safety margin before 150s kill

        # Inverse covariance matrices V = A^{-1} for each arm
        self.V = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.counts = [0] * n_arms
        self.recent_rewards = [[] for _ in range(n_arms)]

        # Adaptive parameters
        self.base_gamma = 0.98
        self.gamma_min = 0.90
        self.gamma_max = 0.999
        self.C = 1.0  # exploration coefficient
        self.C_min = 0.3
        self.C_max = 3.0

        # For tracking reward variance per arm
        self.reward_sums = [0.0] * n_arms
        self.reward_sq_sums = [0.0] * n_arms

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def _adaptive_gamma(self, arm):
        """Adapt forgetting factor based on reward variance and count."""
        c = self.counts[arm]
        if c < 5:
            return self.base_gamma

        # Compute local reward variance
        mean_r = self.reward_sums[arm] / c
        var_r = max(0.0, self.reward_sq_sums[arm] / c - mean_r ** 2)

        # Higher variance -> faster forgetting (lower gamma)
        # Scale variance to adjust gamma
        var_scale = min(var_r, 10.0)
        gamma = self.base_gamma - 0.08 * var_scale
        gamma = max(self.gamma_min, min(self.gamma_max, gamma))
        return gamma

    def _adaptive_C(self, arm):
        """Adapt exploration based on count and uncertainty."""
        c = self.counts[arm]
        if c < 3:
            return self.C_max
        if c < 10:
            return self.C * 1.5
        return self.C

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)

        # Deadline guard: if close to deadline, use greedy (least variance)
        if not self._check_deadline():
            # Greedy fallback: pick arm with highest mean estimate
            scores = []
            for a in range(self.n):
                theta = self.V[a] @ self.b[a]
                scores.append(theta @ context)
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            theta = self.V[a] @ self.b[a]
            mu = theta @ context
            # UCB bonus
            c_val = self._adaptive_C(a)
            bonus = c_val * np.sqrt(context @ self.V[a] @ context)
            scores.append(mu + bonus)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)

        # Track statistics
        self.counts[arm] += 1
        self.reward_sums[arm] += reward
        self.reward_sq_sums[arm] += reward * reward
        self.recent_rewards[arm].append(reward)
        if len(self.recent_rewards[arm]) > 50:
            self.recent_rewards[arm].pop(0)

        # Adaptive forgetting
        gamma = self._adaptive_gamma(arm)

        # Update inverse covariance using Sherman-Morrison with forgetting
        # V_new = (gamma * V_old^{-1} + x x^T)^{-1}
        # Using matrix inversion lemma:
        # (gamma * V^{-1} + x x^T)^{-1} = (1/gamma) * V - (1/gamma)^2 * V x x^T V / (gamma + x^T V x)
        # Actually: (A + u v^T)^{-1} = A^{-1} - A^{-1} u v^T A^{-1} / (1 + v^T A^{-1} u)
        # Here A = gamma * V^{-1}, so A^{-1} = (1/gamma) * V
        # We want (gamma * V^{-1} + x x^T)^{-1}
        # Let M = gamma * V^{-1}, then M^{-1} = (1/gamma) V
        # (M + x x^T)^{-1} = M^{-1} - M^{-1} x x^T M^{-1} / (1 + x^T M^{-1} x)
        # = (1/gamma) V - (1/gamma)^2 V x x^T V / (1 + (1/gamma) x^T V x)
        # = (1/gamma) V - (1/gamma)^2 V x x^T V / ((gamma + x^T V x) / gamma)
        # = (1/gamma) V - (1/gamma) V x x^T V / (gamma + x^T V x)

        V_old = self.V[arm]
        Vx = V_old @ context
        xVx = context @ Vx  # scalar
        denom = gamma + xVx

        if denom > 1e-12:
            self.V[arm] = (1.0 / gamma) * V_old - (1.0 / gamma) * np.outer(Vx, Vx) / denom
        else:
            # Fallback: reset to identity scaled
            self.V[arm] = np.eye(self.d)

        # Update b with forgetting
        self.b[arm] = gamma * self.b[arm] + reward * context
# EVOLVE-BLOCK-END
