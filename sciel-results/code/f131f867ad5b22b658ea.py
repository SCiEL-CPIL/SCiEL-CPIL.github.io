"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Adaptive LinUCB with Sherman-Morrison inverse updates, adaptive forgetting, and deadline guard."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0

        # Inverse covariance matrices V = A^{-1}
        self.V = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.counts = [0] * n_arms

        # Adaptive parameters
        self.gamma = 0.97
        self.gamma_min = 0.85
        self.gamma_max = 0.999
        self.C = 1.5
        self.C_min = 0.5
        self.C_max = 3.0

        # Deadline guard
        self.start_time = time.time()
        self.deadline = 140.0  # Safety margin before 150s limit
        self._past_deadline = False

        # Reward tracking for adaptive gamma
        self.reward_sums = [0.0] * n_arms
        self.reward_sq_sums = [0.0] * n_arms

    def _check_deadline(self):
        if not self._past_deadline:
            if time.time() - self.start_time > self.deadline:
                self._past_deadline = True
        return self._past_deadline

    def _adaptive_gamma(self):
        """Adapt forgetting factor based on reward variance."""
        if self.t < 10:
            return self.gamma
        total_var = 0.0
        for a in range(self.n):
            if self.counts[a] > 5:
                mean = self.reward_sums[a] / self.counts[a]
                var = max(0.0, self.reward_sq_sums[a] / self.counts[a] - mean * mean)
                total_var += var
        avg_var = total_var / self.n
        # Higher variance -> lower gamma (faster forgetting)
        self.gamma = self.gamma_min + (self.gamma_max - self.gamma_min) * max(0.0, 1.0 - avg_var * 2.0)
        self.gamma = max(self.gamma_min, min(self.gamma_max, self.gamma))
        return self.gamma

    def _adaptive_C(self):
        """Adapt exploration parameter over time."""
        if self.t < 20:
            return self.C
        # Decay exploration over time but keep a floor
        progress = min(1.0, self.t / 2000.0)
        self.C = self.C_max - (self.C_max - self.C_min) * progress
        return self.C

    def select(self, context):
        self.t += 1
        if self._check_deadline():
            # Return arm with highest mean estimate
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                theta = self.V[a] @ self.b[a]
                score = theta @ context
                if score > best_score:
                    best_score = score
                    best_arm = a
            return int(best_arm)

        gamma = self._adaptive_gamma()
        C = self._adaptive_C()

        scores = np.zeros(self.n)
        for a in range(self.n):
            theta = self.V[a] @ self.b[a]
            mean = theta @ context
            # Uncertainty: C * sqrt(context^T V context)
            vctx = self.V[a] @ context
            uncertainty = C * np.sqrt(max(0.0, context @ vctx))
            scores[a] = mean + uncertainty

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        if self._check_deadline():
            return

        gamma = self._adaptive_gamma()

        # Track rewards for variance estimation
        self.reward_sums[arm] += reward
        self.reward_sq_sums[arm] += reward * reward
        self.counts[arm] += 1

        # Sherman-Morrison update for inverse covariance
        # V_new = (1/gamma) * V - (1/gamma) * V x x^T V / (gamma + x^T V x)
        V = self.V[arm]
        b = self.b[arm]

        # Scale V and b by forgetting factor
        V = V / gamma
        b = b / gamma

        # Sherman-Morrison: V_new = V - V x x^T V / (1 + x^T V x)
        # But we already scaled, so: V_new = V - V x x^T V / (gamma + x^T (gamma * V_old) x)
        # Actually, let's do it more carefully.
        # A_new = gamma * A_old + x x^T
        # V_new = A_new^{-1} = (gamma * A_old + x x^T)^{-1}
        # Using Sherman-Morrison on gamma * A_old:
        # (gamma * A_old + x x^T)^{-1} = (1/gamma) * A_old^{-1} - (1/gamma) * A_old^{-1} x x^T A_old^{-1} / (gamma + x^T A_old^{-1} x)
        # = (1/gamma) * V_old - (1/gamma) * V_old x x^T V_old / (gamma + x^T V_old x)

        V_old = self.V[arm]  # This is V_old = A_old^{-1}
        Vx = V_old @ context
        xVx = context @ Vx

        V_new = (1.0 / gamma) * V_old - (1.0 / (gamma * (gamma + xVx))) * np.outer(Vx, Vx)
        self.V[arm] = V_new

        self.b[arm] = b + reward * context
# EVOLVE-BLOCK-END
