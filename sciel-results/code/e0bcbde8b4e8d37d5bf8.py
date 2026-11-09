"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Discounted Linear Thompson Sampling with multi-timescale for non-stationary bandits."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.gamma_short = 0.95   # Fast forgetting for short-window
        self.gamma_long = 0.995   # Slow forgetting for long-window
        self.lam = 1.0            # Regularization

        # Short-window estimators (for rapid adaptation)
        self.A_short = [self.lam * np.eye(dim) for _ in range(n_arms)]
        self.b_short = [np.zeros(dim) for _ in range(n_arms)]

        # Long-window estimators (for stable baseline)
        self.A_long = [self.lam * np.eye(dim) for _ in range(n_arms)]
        self.b_long = [np.zeros(dim) for _ in range(n_arms)]

        # Track time for deadline guard
        self._start_time = time.time()
        self._deadline = 110.0  # Safety margin before 120s limit

        # Cholesky cache for short-window (recomputed each update)
        self._chol_short = [np.linalg.cholesky(self.A_short[a]) for a in range(n_arms)]

    def _update_cholesky_short(self, arm, context):
        """Incrementally update Cholesky factor for short-window A matrix."""
        # A_new = gamma * A_old + x x^T
        # We'll just recompute since dim is small (5 arms, dim likely small)
        self.A_short[arm] = self.gamma_short * self.A_short[arm] + np.outer(context, context)
        # Ensure positive definite
        self.A_short[arm] += 1e-8 * np.eye(self.d)
        try:
            self._chol_short[arm] = np.linalg.cholesky(self.A_short[arm])
        except np.linalg.LinAlgError:
            self.A_short[arm] = self.lam * np.eye(self.d)
            self._chol_short[arm] = np.linalg.cholesky(self.A_short[arm])

    def select(self, context):
        # Deadline guard
        if time.time() - self._start_time > self._deadline:
            # Return arm with highest mean estimate from long-window
            best_arm = 0
            best_mean = -np.inf
            for a in range(self.n):
                try:
                    theta = np.linalg.solve(self.A_long[a], self.b_long[a])
                    mean = theta @ context
                    if mean > best_mean:
                        best_mean = mean
                        best_arm = a
                except np.linalg.LinAlgError:
                    pass
            return best_arm

        context = np.asarray(context, dtype=np.float64)

        # Use short-window for Thompson Sampling (faster adaptation)
        scores = []
        for a in range(self.n):
            try:
                # Solve for theta using Cholesky
                L = self._chol_short[a]
                # Solve L * y = b
                y = np.linalg.solve(L, self.b_short[a])
                # theta = L^{-T} * y
                theta = np.linalg.solve(L.T, y)
                mean = theta @ context

                # Sample from posterior: theta ~ N(theta_hat, sigma^2 * (A^{-1}))
                # A^{-1} = L^{-1} L^{-T}
                # Sample: theta_sample = theta + sigma * L^{-1} * z
                z = np.random.randn(self.d)
                # Solve L * v = z
                v = np.linalg.solve(L, z)
                theta_sample = theta + v  # sigma=1 assumed for Gaussian reward
                scores.append(theta_sample @ context)
            except np.linalg.LinAlgError:
                scores.append(0.0)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)

        # Update short-window
        self._update_cholesky_short(arm, context)
        self.b_short[arm] = self.gamma_short * self.b_short[arm] + reward * context

        # Update long-window
        self.A_long[arm] = self.gamma_long * self.A_long[arm] + np.outer(context, context)
        self.b_long[arm] = self.gamma_long * self.b_long[arm] + reward * context
# EVOLVE-BLOCK-END
