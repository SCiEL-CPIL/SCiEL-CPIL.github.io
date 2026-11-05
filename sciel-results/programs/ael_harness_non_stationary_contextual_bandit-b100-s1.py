"""Non-stationary linear contextual bandit with adaptive forgetting."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Adaptive-forgetting LinUCB for non-stationary linear bandits."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.I = np.eye(dim)
        # Per-arm precision and cross-moment
        self.A = [self.I.copy() for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        # Adaptive forgetting parameters
        self.base_decay = 0.99       # mild forgetting each step
        self.shift_decay = 0.5       # strong forgetting on regime shift
        self.shift_threshold = 2.0   # residual/uncertainty threshold
        self.min_decay = 0.3         # floor for decay factor
        self.t = 0

    def _cholesky_inv(self, M):
        """Stable inverse via Cholesky, with fallback to pinv."""
        try:
            L = np.linalg.cholesky(M)
            # Solve L L^T X = I  =>  X = inv(L^T) inv(L)
            inv_L = np.linalg.inv(L)
            inv_M = inv_L.T @ inv_L
            return inv_M
        except np.linalg.LinAlgError:
            return np.linalg.pinv(M)

    def select(self, context):
        scores = []
        for a in range(self.n):
            A_inv = self._cholesky_inv(self.A[a])
            theta = A_inv @ self.b[a]
            mu = theta @ context
            # Uncertainty: sqrt(context^T A^{-1} context)
            var = context @ A_inv @ context
            std = np.sqrt(max(var, 1e-12))
            # UCB score
            score = mu + 1.0 * std
            scores.append(score)
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.t += 1
        # Compute current model prediction and uncertainty for this arm
        A_inv = self._cholesky_inv(self.A[arm])
        theta = A_inv @ self.b[arm]
        mu = theta @ context
        var = context @ A_inv @ context
        std = np.sqrt(max(var, 1e-12))

        # Normalized residual: how surprised are we?
        if std > 1e-8:
            normalized_residual = abs(reward - mu) / std
        else:
            normalized_residual = abs(reward - mu)

        # Decide decay factor: strong forgetting if regime shift detected
        if normalized_residual > self.shift_threshold:
            decay = self.shift_decay
        else:
            decay = self.base_decay

        # Apply forgetting
        self.A[arm] = decay * self.A[arm]
        self.b[arm] = decay * self.b[arm]

        # Add new observation
        self.A[arm] = self.A[arm] + np.outer(context, context)
        self.b[arm] = self.b[arm] + reward * context
# EVOLVE-BLOCK-END
