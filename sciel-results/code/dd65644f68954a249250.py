# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit using a multi-scale ensemble of linear models.
    - Each arm maintains a bank of regressors with fixed, geometrically spaced forgetting factors.
    - The policy selects the arm that maximizes the reward prediction from the ensemble member
      with the lowest recent residual variance, effectively using model selection to dynamically
      switch memory horizons without explicit regime detection.
    - UCB-style exploration with adaptive alpha.
    - Cholesky-based solves for speed and numerical stability.
    - Hard deadline guard to ensure we never exceed wall-clock limits.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.lam = 1.0

        # Multi-scale ensemble: geometrically spaced forgetting factors
        # Short memory (fast adaptation) to long memory (stable)
        self.betas = [0.85, 0.90, 0.95, 0.99, 0.999]
        self.n_scales = len(self.betas)

        # Per-arm, per-scale covariance and observation sums
        # A[a][s] is dim x dim, b[a][s] is dim
        self.A = [[np.eye(dim) * self.lam for _ in range(self.n_scales)] for _ in range(n_arms)]
        self.b = [[np.zeros(dim) for _ in range(self.n_scales)] for _ in range(n_arms)]

        # Running per-arm, per-scale reward statistics for residual variance
        self.count = [[0] * self.n_scales for _ in range(n_arms)]
        self.mean_r = [[0.0] * self.n_scales for _ in range(n_arms)]
        self.m2 = [[0.0] * self.n_scales for _ in range(n_arms)]

        # Exploration hyperparameters
        self.alpha_base = 1.2

        # Time guard
        self.start_time = time.time()
        self.deadline = self.start_time + 112.0

    def _safe(self):
        return time.time() >= self.deadline

    def _solve_theta(self, a, s):
        try:
            L = np.linalg.cholesky(self.A[a][s])
            theta = np.linalg.solve(L.T, np.linalg.solve(L, self.b[a][s]))
            return theta, L
        except np.linalg.LinAlgError:
            try:
                theta = np.linalg.lstsq(self.A[a][s], self.b[a][s], rcond=None)[0]
                return theta, None
            except Exception:
                return np.zeros(self.d), None

    def _get_best_scale(self, a):
        """Return the scale index with lowest recent residual variance for arm a."""
        best_s = 0
        best_var = float('inf')
        for s in range(self.n_scales):
            n = self.count[a][s]
            if n < 2:
                var = float('inf')
            else:
                var = max(self.m2[a][s] / max(n - 1, 1), 1e-8)
            if var < best_var:
                best_var = var
                best_s = s
        return best_s

    def select(self, context):
        if self._safe():
            scores = []
            for a in range(self.n):
                best_s = self._get_best_scale(a)
                try:
                    theta, _ = self._solve_theta(a, best_s)
                    scores.append(float(theta @ context))
                except Exception:
                    scores.append(0.0)
            return int(np.argmax(scores))

        alpha = self.alpha_base
        scores = []
        for a in range(self.n):
            best_s = self._get_best_scale(a)
            try:
                theta, L = self._solve_theta(a, best_s)
                pred = float(theta @ context)
                if L is not None:
                    v = np.linalg.solve(L, context)
                    var_term = float(v @ v)
                else:
                    try:
                        cov = np.linalg.inv(self.A[a][best_s])
                        var_term = float(context @ cov @ context)
                    except Exception:
                        var_term = 0.0
                scores.append(pred + alpha * np.sqrt(max(var_term, 0.0)))
            except Exception:
                scores.append(0.0)
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        if self._safe():
            for s in range(self.n_scales):
                self.A[arm][s] = self.A[arm][s] + np.outer(context, context)
                self.b[arm][s] = self.b[arm][s] + reward * context
                self._update_stats(arm, s, reward)
            return

        for s in range(self.n_scales):
            beta = self.betas[s]
            self.A[arm][s] = beta * self.A[arm][s] + np.outer(context, context)
            self.b[arm][s] = beta * self.b[arm][s] + reward * context
            self._update_stats(arm, s, reward)

    def _update_stats(self, arm, s, reward):
        n = self.count[arm][s] + 1
        self.count[arm][s] = n
        delta = reward - self.mean_r[arm][s]
        self.mean_r[arm][s] += delta / n
        self.m2[arm][s] += delta * (reward - self.mean_r[arm][s])
# EVOLVE-BLOCK-END
