"""Non-stationary linear contextual bandit: per-arm LinUCB with Cholesky variance,
adaptive forgetting via residual EWMA, and a global ridge value model for exploitation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """LinUCB with adaptive forgetting, Cholesky-based variance, and a global
    ridge value model blended into the score for stronger exploitation."""

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.t = 0
        self.A = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]

        # Global ridge value model (fit directly to reward)
        self.ridge = 1.0
        self.G = np.eye(dim) * self.ridge
        self.f = np.zeros(dim)
        self.theta = np.zeros(dim)
        self.theta_var = 1.0

        # Adaptive forgetting parameters
        self.base_forgetting = 0.995
        self.current_forgetting = self.base_forgetting
        self.f_min = 0.90
        self.f_max = 0.998

        # EWMA of squared residuals for non-stationarity detection
        self.residual_sq_ewma = [0.0 for _ in range(n_arms)]
        self.residual_ewma_alpha = 0.05

        # UCB exploration coefficient
        self.alpha = 0.5

        # Deadline guard
        self.start_time = time.time()
        self.deadline = 115.0
        self.deadline_hit = False

    def _check_deadline(self):
        if self.deadline_hit:
            return
        if time.time() - self.start_time > self.deadline:
            self.deadline_hit = True

    def _update_global(self):
        """Update the global ridge value model."""
        inv = np.linalg.inv(self.G + self.ridge * np.eye(self.d))
        self.theta = inv @ self.f
        self.theta_var = 0.5 * np.mean(np.abs(np.diag(inv)))

    def _get_forgetting(self, arm, context, reward):
        """Adapt forgetting rate based on observed non-stationarity."""
        if self.t > 0:
            inv_A = np.linalg.inv(self.A[arm] + 1e-6 * np.eye(self.d))
            theta = inv_A @ self.b[arm]
            pred = theta @ context
            residual = reward - pred
            residual_sq = residual ** 2

            # Update EWMA of squared residuals
            self.residual_sq_ewma[arm] = (
                (1 - self.residual_ewma_alpha) * self.residual_sq_ewma[arm]
                + self.residual_ewma_alpha * residual_sq
            )

            # If residual is large, increase forgetting (lower alpha)
            if self.residual_sq_ewma[arm] > 0.1:
                self.current_forgetting = max(
                    self.f_min,
                    self.base_forgetting - 0.1 * np.sqrt(self.residual_sq_ewma[arm])
                )
            else:
                # Gradually return to base forgetting
                self.current_forgetting += 0.001 * (self.base_forgetting - self.current_forgetting)

        # Clamp
        self.current_forgetting = min(max(self.current_forgetting, self.f_min), self.f_max)
        return self.current_forgetting

    def select(self, context):
        self._check_deadline()
        ctx = np.asarray(context, dtype=float)

        if self.deadline_hit:
            # Pure exploitation via global value model
            if self.t > 0:
                return int(np.argmax(self.theta @ ctx))
            return 0

        scores = np.empty(self.n)
        for a in range(self.n):
            try:
                L = np.linalg.cholesky(self.A[a] + 1e-6 * np.eye(self.d))
                inv_A = np.linalg.inv(self.A[a] + 1e-6 * np.eye(self.d))
                theta = inv_A @ self.b[a]
                mean = theta @ ctx
                v = np.linalg.solve(L, ctx)
                variance = np.dot(v, v)
                scores[a] = mean + self.alpha * np.sqrt(variance)
            except np.linalg.LinAlgError:
                inv_A = np.linalg.inv(self.A[a] + 1e-6 * np.eye(self.d))
                theta = inv_A @ self.b[a]
                mean = theta @ ctx
                variance = ctx @ inv_A @ ctx
                scores[a] = mean + self.alpha * np.sqrt(variance)

        # Blend with global value model for stronger exploitation
        if self.t > 0:
            global_score = self.theta @ ctx + self.theta_var * np.linalg.norm(ctx) ** 2
            blended = 0.5 * scores + 0.5 * global_score
        else:
            blended = scores

        return int(np.argmax(blended))

    def update(self, context, arm, reward):
        self._check_deadline()
        ctx = np.asarray(context, dtype=float)
        reward = float(reward)

        if self.deadline_hit:
            return

        # Get adaptive forgetting rate
        alpha_f = self._get_forgetting(arm, ctx, reward)

        # Per-arm ridge update with forgetting
        self.A[arm] = alpha_f * self.A[arm] + np.outer(ctx, ctx)
        self.b[arm] = alpha_f * self.b[arm] + reward * ctx

        # Global value model update
        self.G = alpha_f * self.G + np.outer(ctx, ctx)
        self.f = alpha_f * self.f + reward * ctx
        self._update_global()

        self.t += 1
# EVOLVE-BLOCK-END
