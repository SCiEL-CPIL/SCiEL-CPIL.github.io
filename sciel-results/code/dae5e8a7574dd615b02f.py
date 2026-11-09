"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Linear Thompson Sampling with a forgetting factor for non-stationary linear bandits."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.alpha = 0.99
        self.sigma_sq = 1.0
        self._start_time = time.time()
        self._deadline = 115.0  # safety margin before 120s limit
        self._round_count = 0

    def _compute_posterior(self, arm, context):
        """Compute posterior mean and sample from it using Cholesky."""
        A = self.A[arm]
        b = self.b[arm]
        try:
            L = np.linalg.cholesky(A)
            # Solve L * y = b for y
            y = np.linalg.solve(L, b)
            # theta_hat = L^{-T} * y
            theta_hat = np.linalg.solve(L.T, y)
            # Sample: theta = theta_hat + sigma * L^{-T} * z where z ~ N(0, I)
            z = np.random.randn(self.d)
            L_inv_T_z = np.linalg.solve(L.T, z)
            theta = theta_hat + self.sigma_sq * L_inv_T_z
            return theta
        except np.linalg.LinAlgError:
            # Fallback to direct inverse if Cholesky fails
            try:
                inv_A = np.linalg.inv(A)
                theta_hat = inv_A @ b
                z = np.random.randn(self.d)
                # Covariance is sigma_sq * inv_A
                cov = self.sigma_sq * inv_A
                # Sample from multivariate normal
                L_cov = np.linalg.cholesky(cov)
                theta = theta_hat + L_cov @ z
                return theta
            except np.linalg.LinAlgError:
                # Last resort: just use the mean
                try:
                    inv_A = np.linalg.pinv(A)
                    return inv_A @ b
                except:
                    return np.zeros(self.d)

    def select(self, context):
        self._round_count += 1
        # Deadline guard: if we're close to the limit, exploit greedily
        elapsed = time.time() - self._start_time
        if elapsed > self._deadline:
            # Greedy fallback: use posterior means without sampling
            scores = []
            for a in range(self.n):
                try:
                    inv_A = np.linalg.inv(self.A[a])
                    theta = inv_A @ self.b[a]
                    scores.append(theta @ context)
                except:
                    scores.append(0.0)
            return int(np.argmax(scores))

        # Thompson Sampling: sample from posterior for each arm
        scores = []
        for a in range(self.n):
            theta = self._compute_posterior(a, context)
            scores.append(theta @ context)
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Update with forgetting factor
        self.A[arm] = self.alpha * self.A[arm] + np.outer(context, context)
        self.b[arm] = self.alpha * self.b[arm] + reward * context
# EVOLVE-BLOCK-END
