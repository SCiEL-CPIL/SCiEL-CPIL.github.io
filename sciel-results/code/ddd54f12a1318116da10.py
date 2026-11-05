# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit policy.
    
    Uses per-arm linear models with:
    1. Forgetting factor for non-stationarity
    2. Time-adaptive exploration (UCB)
    3. Cholesky-based efficient inverse computation
    4. Deadline guard for safety
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.counts = [0 for _ in range(n_arms)]
        self.start_time = time.time()
        self.deadline = 110.0  # Safety margin before 120s limit
        self.forgetting = 0.98
        self.alpha = 1.0  # Exploration coefficient
        self.ri = 0.1  # Ridge regularization
        self._chol_cache = [None] * n_arms

    def _get_theta(self, arm):
        """Compute theta = A^{-1} b using Cholesky decomposition."""
        A = self.A[arm]
        # Add ridge for numerical stability
        A_reg = A + self.ri * np.eye(self.d)
        try:
            L = np.linalg.cholesky(A_reg)
            # Solve L y = b
            y = np.linalg.solve(L, self.b[arm])
            # Solve L^T theta = y
            theta = np.linalg.solve(L.T, y)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if Cholesky fails
            theta = np.linalg.pinv(A_reg) @ self.b[arm]
        return theta

    def _get_variance(self, arm, context):
        """Compute variance estimate for the prediction."""
        A = self.A[arm]
        A_reg = A + self.ri * np.eye(self.d)
        try:
            L = np.linalg.cholesky(A_reg)
            # variance = context^T (A^{-1}) context
            # = ||L^{-1} context||^2
            z = np.linalg.solve(L, context)
            var = np.dot(z, z)
        except np.linalg.LinAlgError:
            # Fallback
            inv_A = np.linalg.pinv(A_reg)
            var = context @ inv_A @ context
        return var

    def select(self, context):
        # Deadline guard
        if time.time() - self.start_time > self.deadline:
            return 0

        context = np.asarray(context, dtype=np.float64)
        scores = np.zeros(self.n)

        for a in range(self.n):
            theta = self._get_theta(a)
            pred = np.dot(theta, context)
            var = self._get_variance(a, context)
            # Time-adaptive exploration: increase exploration when count is low
            # or when model is uncertain
            count = self.counts[a]
            # Adaptive alpha: higher for less explored arms
            adaptive_alpha = self.alpha * (1.0 + 0.5 / (1.0 + count * 0.1))
            scores[a] = pred + adaptive_alpha * np.sqrt(var)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)

        # Apply forgetting factor
        self.A[arm] = self.forgetting * self.A[arm] + np.outer(context, context)
        self.b[arm] = self.forgetting * self.b[arm] + reward * context
        self.counts[arm] += 1

        # Invalidate Cholesky cache
        self._chol_cache[arm] = None
# EVOLVE-BLOCK-END
