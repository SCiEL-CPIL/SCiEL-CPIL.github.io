"""
Non-stationary linear contextual bandit with event-driven adaptive forgetting.
Uses a single Bayesian linear model per arm with dynamic forgetting factor
adjusted based on prediction error. Thompson Sampling for exploration.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t_start = time.time()
        self.deadline = self.t_start + 115.0

        # One Bayesian linear model per arm
        # V = inverse covariance (precision), b = sum of x*r
        self.V = [np.eye(dim) * 0.1 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]

        # Forgetting factor (adaptive)
        self.forgetting = 0.99
        self.min_forgetting = 0.7
        self.max_forgetting = 0.999
        self.forgetting_rate = 0.1

        # Prediction error tracking
        self.error_history = [[] for _ in range(n_arms)]
        self.error_window = 20

        # Precompute Cholesky for initial V
        self._L = [np.linalg.cholesky(self.V[a]) for a in range(n_arms)]

        # Track number of observations per arm
        self.counts = [0] * n_arms

    def _update_chol(self, arm, V):
        """Update Cholesky factor of V."""
        try:
            return np.linalg.cholesky(V + 1e-8 * np.eye(self.d))
        except np.linalg.LinAlgError:
            V = V + 1e-4 * np.eye(self.d)
            try:
                return np.linalg.cholesky(V)
            except np.linalg.LinAlgError:
                return np.eye(self.d)

    def _get_theta(self, arm):
        """Get posterior mean theta."""
        V = self.V[arm]
        b = self.b[arm]
        try:
            return np.linalg.solve(V, b)
        except np.linalg.LinAlgError:
            return np.zeros(self.d)

    def select(self, context):
        # Deadline guard
        if time.time() > self.deadline:
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                try:
                    theta = self._get_theta(a)
                    score = theta @ context
                    if score > best_score:
                        best_score = score
                        best_arm = a
                except:
                    pass
            return int(best_arm)

        context = np.asarray(context, dtype=np.float64)
        scores = []

        for a in range(self.n):
            try:
                theta = self._get_theta(a)
                # Thompson Sampling: sample from posterior
                L = self._L[a]
                z = np.random.randn(self.d)
                # sqrt(V^{-1}) z = L^{-T} z gives noise in theta space
                # Actually we need sqrt(cov) z where cov = V^{-1}
                # cov = L^{-T} L^{-1}, so sqrt(cov) z = L^{-T} z
                noise = np.linalg.solve(L.T, z)
                theta_sample = theta + noise
                scores.append(theta_sample @ context)
            except Exception:
                try:
                    theta = self._get_theta(a)
                    scores.append(theta @ context)
                except:
                    scores.append(0.0)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Deadline guard
        if time.time() > self.deadline:
            return

        context = np.asarray(context, dtype=np.float64)

        # Compute prediction error
        try:
            theta = self._get_theta(arm)
            pred = theta @ context
            residual = reward - pred
            # Normalize by prediction uncertainty
            V = self.V[arm]
            ctx_var = context @ np.linalg.solve(V, context)
            if ctx_var > 1e-10:
                mahalanobis = abs(residual) / np.sqrt(ctx_var)
            else:
                mahalanobis = abs(residual)
        except Exception:
            mahalanobis = 1.0

        # Update error history
        self.error_history[arm].append(mahalanobis)
        if len(self.error_history[arm]) > self.error_window:
            self.error_history[arm].pop(0)

        # Adaptive forgetting: large errors -> faster forgetting
        if mahalanobis > 3.0:
            target_forgetting = self.min_forgetting
        elif mahalanobis > 2.0:
            target_forgetting = 0.92
        elif mahalanobis > 1.0:
            mean_err = np.mean(self.error_history[arm]) if self.error_history[arm] else 0
            if mean_err > 1.5:
                target_forgetting = 0.95
            else:
                target_forgetting = 0.99
        else:
            mean_err = np.mean(self.error_history[arm]) if self.error_history[arm] else 0
            if mean_err < 0.5:
                target_forgetting = self.max_forgetting
            else:
                target_forgetting = 0.995

        self.forgetting += self.forgetting_rate * (target_forgetting - self.forgetting)
        self.forgetting = np.clip(self.forgetting, self.min_forgetting, self.max_forgetting)

        f = self.forgetting

        # Update Bayesian model: V_new = f * V_old + x x^T, b_new = f * b_old + r * x
        self.V[arm] = f * self.V[arm] + np.outer(context, context)
        self.b[arm] = f * self.b[arm] + reward * context
        self._L[arm] = self._update_chol(arm, self.V[arm])
        self.counts[arm] += 1
# EVOLVE-BLOCK-END
