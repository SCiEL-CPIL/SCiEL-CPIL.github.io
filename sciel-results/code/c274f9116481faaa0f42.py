"""Non-stationary linear contextual bandit with dual-timescale tracking."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Dual-timescale Thompson Sampling for non-stationary linear bandits.
    
    Maintains two parallel estimators per arm:
    1. A fast (short-horizon) estimator that adapts quickly to recent changes
    2. A slow (long-horizon) estimator that provides stability
    
    The selection uses a mixture of both, with the mixing weight adapted
    based on recent prediction errors. This allows rapid response to
    non-stationarity while maintaining robustness.
    """

    def __init__(self, n_arms, dim):
        self.n_arms = n_arms
        self.dim = dim
        self.t_start = time.time()
        self.deadline = 145.0  # wall-clock limit with safety margin

        # Fast estimator (high discount, quick adaptation)
        self.A_fast = [np.eye(dim) * 0.1 for _ in range(n_arms)]
        self.b_fast = [np.zeros(dim) for _ in range(n_arms)]
        self.discount_fast = 0.95

        # Slow estimator (low discount, stable)
        self.A_slow = [np.eye(dim) * 0.1 for _ in range(n_arms)]
        self.b_slow = [np.zeros(dim) for _ in range(n_arms)]
        self.discount_slow = 0.995

        # Track recent prediction errors per arm for adaptive mixing
        self.error_buffer = [np.zeros(20) for _ in range(n_arms)]
        self.error_idx = [0] * n_arms

        # Inverse matrix cache
        self.A_inv_fast = [np.eye(dim) for _ in range(n_arms)]
        self.A_inv_slow = [np.eye(dim) for _ in range(n_arms)]
        self._valid_fast = [True] * n_arms
        self._valid_slow = [True] * n_arms

        # Noise variance estimate
        self.sigma2 = 1.0

    def _update_inverse(self, A, A_inv, valid):
        if not valid:
            try:
                A_inv[:] = np.linalg.inv(A)
            except np.linalg.LinAlgError:
                A_inv[:] = np.eye(self.dim)
            valid = True
        return valid

    def _get_theta_cov_fast(self, arm):
        self._valid_fast[arm] = self._update_inverse(
            self.A_fast[arm], self.A_inv_fast[arm], self._valid_fast[arm]
        )
        theta = self.A_inv_fast[arm] @ self.b_fast[arm]
        cov = self.A_inv_fast[arm]
        return theta, cov

    def _get_theta_cov_slow(self, arm):
        self._valid_slow[arm] = self._update_inverse(
            self.A_slow[arm], self.A_inv_slow[arm], self._valid_slow[arm]
        )
        theta = self.A_inv_slow[arm] @ self.b_slow[arm]
        cov = self.A_inv_slow[arm]
        return theta, cov

    def _get_mixed_estimate(self, arm):
        """Get mixed theta and covariance from fast and slow estimators."""
        theta_f, cov_f = self._get_theta_cov_fast(arm)
        theta_s, cov_s = self._get_theta_cov_slow(arm)

        # Compute adaptive mixing weight based on recent errors
        errors = self.error_buffer[arm]
        if np.any(errors != 0):
            mean_error = np.mean(errors[errors != 0])
            # Higher error -> more weight on fast estimator
            mix = np.clip(mean_error / 2.0, 0.3, 0.9)
        else:
            mix = 0.5

        # Mixture of Gaussians: use weighted average for theta
        # For covariance, use a simple weighted combination
        theta = mix * theta_f + (1 - mix) * theta_s
        # Approximate covariance as weighted sum (conservative)
        cov = mix * cov_f + (1 - mix) * cov_s

        return theta, cov

    def select(self, context):
        """Select an arm using Thompson Sampling on the mixed posterior."""
        context = np.asarray(context, dtype=np.float64)

        # Deadline guard
        if time.time() - self.t_start > self.deadline:
            # Fallback: use slow estimator mean
            means = []
            for a in range(self.n_arms):
                theta, _ = self._get_theta_cov_slow(a)
                means.append(theta @ context)
            return int(np.argmax(means))

        scores = []
        for a in range(self.n_arms):
            theta, cov = self._get_mixed_estimate(a)
            # Thompson Sampling: draw from N(theta, cov)
            # Use Cholesky for efficiency
            try:
                L = np.linalg.cholesky(cov + 1e-8 * np.eye(self.dim))
                z = np.random.randn(self.dim)
                sample = theta + L @ z
            except np.linalg.LinAlgError:
                sample = theta
            scores.append(sample @ context)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        """Update both fast and slow estimators with a new observation."""
        context = np.asarray(context, dtype=np.float64)

        # Compute prediction error using mixed estimate
        theta, cov = self._get_mixed_estimate(arm)
        pred = theta @ context
        residual = reward - pred

        # Store error for adaptive mixing
        self.error_buffer[arm][self.error_idx[arm]] = abs(residual)
        self.error_idx[arm] = (self.error_idx[arm] + 1) % 20

        # Update fast estimator
        self.A_fast[arm] = self.discount_fast * self.A_fast[arm] + np.outer(context, context)
        self.b_fast[arm] = self.discount_fast * self.b_fast[arm] + reward * context
        self._valid_fast[arm] = False

        # Update slow estimator
        self.A_slow[arm] = self.discount_slow * self.A_slow[arm] + np.outer(context, context)
        self.b_slow[arm] = self.discount_slow * self.b_slow[arm] + reward * context
        self._valid_slow[arm] = False
# EVOLVE-BLOCK-END
