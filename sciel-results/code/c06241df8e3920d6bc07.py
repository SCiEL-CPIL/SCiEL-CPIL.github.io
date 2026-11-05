# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """Non-stationary linear contextual bandit with discounted LinUCB and adaptive forgetting."""

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.round_count = 0

        # LinUCB state per arm: A_inv (dim x dim), b (dim,)
        self.A_inv = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]

        # Discount factor for non-stationarity (gamma < 1 means forgetting)
        self.gamma = 0.95
        # Adaptive: track per-arm innovation to adjust gamma
        self.innov_sq = [0.0 for _ in range(n_arms)]
        self.innov_sq_ewma = [1.0 for _ in range(n_arms)]
        self.ewma_alpha = 0.1  # EWMA weight for new observation

        # Exploration parameter
        self.C = 1.0

        # Deadline guard
        self._start_time = time.time()
        self._deadline = 115.0  # 5s safety margin

    def _theta_and_uncertainty(self, A_inv, b, context):
        theta = A_inv @ b
        pred = theta @ context
        var = context @ A_inv @ context
        return pred, np.sqrt(max(var, 0.0))

    def select(self, context):
        # Deadline guard: if close to timeout, just pick the arm with highest predicted mean
        if time.time() - self._start_time > self._deadline:
            preds = []
            for a in range(self.n):
                pred, _ = self._theta_and_uncertainty(self.A_inv[a], self.b[a], context)
                preds.append(pred)
            return int(np.argmax(preds))

        scores = []
        for a in range(self.n):
            pred, unc = self._theta_and_uncertainty(self.A_inv[a], self.b[a], context)
            scores.append(pred + self.C * unc)
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.round_count += 1

        # Deadline guard
        if time.time() - self._start_time > self._deadline:
            return

        x = context

        # Compute prediction before update for innovation
        pred, unc = self._theta_and_uncertainty(self.A_inv[arm], self.b[arm], x)
        innovation = reward - pred
        innov_sq = innovation ** 2

        # Update EWMA of squared innovation
        self.innov_sq_ewma[arm] = (1 - self.ewma_alpha) * self.innov_sq_ewma[arm] + self.ewma_alpha * innov_sq

        # Adaptive discount: if innovations are large, increase forgetting (lower gamma)
        # Base gamma is 0.95; if std is high, reduce gamma to forget faster
        std_est = np.sqrt(max(self.innov_sq_ewma[arm], 1e-6))
        # Map std to gamma: higher std -> lower gamma (more forgetting)
        # gamma ranges from 0.90 (very adaptive) to 0.99 (very stable)
        gamma_adaptive = np.clip(0.99 - 0.05 * std_est, 0.90, 0.99)

        # Discount the existing state (forget older data)
        self.A_inv[arm] = self.A_inv[arm] / gamma_adaptive
        self.b[arm] = self.b[arm] / gamma_adaptive

        # Sherman-Morrison update for the new observation
        V = self.A_inv[arm]
        Vx = V @ x
        denom = 1.0 + x @ Vx
        V_new = V - np.outer(Vx, Vx) / denom
        V_new = 0.5 * (V_new + V_new.T)  # symmetrize
        b_new = self.b[arm] + reward * x

        self.A_inv[arm] = V_new
        self.b[arm] = b_new
# EVOLVE-BLOCK-END
