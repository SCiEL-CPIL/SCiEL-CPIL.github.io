# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit using a multi-scale ensemble of linear models.
    - Each arm maintains a bank of regressors with fixed, geometrically spaced forgetting factors.
    - Soft-max (log-sum-exp) weighting across scales based on negative residual variance,
      providing a smooth surrogate for hard model selection.
    - UCB-style exploration with adaptive alpha that decays with confidence.
    - Cholesky-based solves for speed and numerical stability.
    - Hard deadline guard to ensure we never exceed wall-clock limits.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.lam = 1.0

        # Multi-scale ensemble: geometrically spaced forgetting factors
        # Slightly tighter to respond faster to non-stationarity
        self.betas = [0.82, 0.88, 0.93, 0.97, 0.995]
        self.n_scales = len(self.betas)

        # Per-arm, per-scale covariance and observation sums
        self.A = [[np.eye(dim) * self.lam for _ in range(self.n_scales)] for _ in range(n_arms)]
        self.b = [[np.zeros(dim) for _ in range(self.n_scales)] for _ in range(n_arms)]

        # Running per-arm, per-scale reward statistics for residual variance
        self.count = [[0] * self.n_scales for _ in range(n_arms)]
        self.mean_r = [[0.0] * self.n_scales for _ in range(n_arms)]
        self.m2 = [[0.0] * self.n_scales for _ in range(n_arms)]

        # Exploration hyperparameters
        self.alpha_base = 1.4
        self.alpha_min = 0.3
        self.alpha_decay = 0.9995

        # Soft-max temperature for scale weighting
        self.softmax_temp = 5.0

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

    def _scale_weights(self, a):
        """
        Compute soft-max weights over scales for arm a based on negative log-variance.
        Lower residual variance -> higher weight. Uses log-sum-exp for numerical stability.
        """
        scores = np.zeros(self.n_scales)
        for s in range(self.n_scales):
            n = self.count[a][s]
            if n < 2:
                scores[s] = -1e6  # heavily penalize scales with no data
            else:
                var = max(self.m2[a][s] / max(n - 1, 1), 1e-10)
                # Negative log-variance: lower variance -> higher score
                scores[s] = -np.log(var)
        
        # Soft-max via log-sum-exp
        max_score = np.max(scores)
        exp_scores = np.exp((scores - max_score) / self.softmax_temp)
        total = np.sum(exp_scores)
        if total < 1e-30:
            # Uniform fallback
            return np.ones(self.n_scales) / self.n_scales
        weights = exp_scores / total
        return weights

    def select(self, context):
        if self._safe():
            # Fast path: use argmax over weighted predictions without exploration
            scores = []
            for a in range(self.n):
                w = self._scale_weights(a)
                pred = 0.0
                for s in range(self.n_scales):
                    if w[s] < 1e-6:
                        continue
                    try:
                        theta, _ = self._solve_theta(a, s)
                        pred += w[s] * float(theta @ context)
                    except Exception:
                        pass
                scores.append(pred)
            return int(np.argmax(scores))

        # Adaptive alpha: decays as we accumulate data
        total_count = sum(self.count[a][s] for a in range(self.n) for s in range(self.n_scales))
        alpha = max(self.alpha_base * (self.alpha_decay ** total_count), self.alpha_min)

        scores = []
        for a in range(self.n):
            w = self._scale_weights(a)
            pred = 0.0
            var_term = 0.0
            for s in range(self.n_scales):
                if w[s] < 1e-6:
                    continue
                try:
                    theta, L = self._solve_theta(a, s)
                    pred += w[s] * float(theta @ context)
                    if L is not None:
                        v = np.linalg.solve(L, context)
                        var_term += w[s] * float(v @ v)
                    else:
                        try:
                            cov = np.linalg.inv(self.A[a][s])
                            var_term += w[s] * float(context @ cov @ context)
                        except Exception:
                            pass
                except Exception:
                    pass
            scores.append(pred + alpha * np.sqrt(max(var_term, 0.0)))
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
