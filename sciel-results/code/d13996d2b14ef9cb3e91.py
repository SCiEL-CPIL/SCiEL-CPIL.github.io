"""Adaptive Exponential Forgetting Thompson Sampling with Sherman-Morrison-Woodbury updates."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Non-stationary linear contextual bandit policy using:
    - Discounted Least Squares with adaptive forgetting factor (gamma)
    - Thompson Sampling for exploration (correct posterior sampling)
    - Sherman-Morrison-Woodbury rank-1 updates for O(d^2) per step
    - Adaptive gamma based on residual magnitude
    - Periodic full re-inversion to prevent numerical drift
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0

        # Regularization
        self.reg = 1.0

        # Thompson Sampling noise variance
        self.sigma_sq = 1.0

        # Per-arm statistics: A (covariance inverse), b (moment)
        self.A = [self.reg * np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.A_inv = [np.eye(dim) / self.reg for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]

        # Pre-compute Cholesky of A_inv for fast TS sampling (updated periodically)
        self.A_inv_cholesky = []
        for a in range(n_arms):
            try:
                L = np.linalg.cholesky(self.A_inv[a])
                self.A_inv_cholesky.append(L)
            except np.linalg.LinAlgError:
                self.A_inv_cholesky.append(np.eye(dim) / np.sqrt(self.reg))

        # Adaptive forgetting parameters
        self.base_gamma = 0.95
        self.min_gamma = 0.75
        self.max_gamma = 0.99

        # EWMA of squared residuals to drive adaptation
        self.residual_sq_ewma = [0.0 for _ in range(n_arms)]
        self.ewma_alpha = 0.2

        # Current gamma for each arm
        self.gamma = [self.base_gamma for _ in range(n_arms)]

        # Periodic re-inversion interval
        self.reinv_interval = 50

        # Deadline guard
        self.start_time = time.time()
        self.deadline = 115.0

    def _update_gamma(self, arm, residual_sq):
        """
        Update the forgetting factor based on the magnitude of the residual.
        Large residual -> model is wrong -> forget old data faster (lower gamma).
        Small residual -> model is good -> keep memory (higher gamma).
        """
        # Use a more responsive mapping
        r = self.residual_sq_ewma[arm]
        # Map residual magnitude to gamma: high residual -> low gamma
        # Use sigmoid-like mapping
        scale = 2.0
        factor = 1.0 / (1.0 + r * scale)

        new_gamma = self.min_gamma + (self.max_gamma - self.min_gamma) * factor

        # Smooth the gamma update to avoid oscillation
        self.gamma[arm] = 0.8 * self.gamma[arm] + 0.2 * new_gamma

    def _recompute_arm_stats(self, arm):
        """Recompute inverse and theta from current A and b."""
        try:
            self.A_inv[arm] = np.linalg.inv(self.A[arm])
            self.theta[arm] = self.A_inv[arm] @ self.b[arm]
            # Update Cholesky for TS sampling
            try:
                self.A_inv_cholesky[arm] = np.linalg.cholesky(self.A_inv[arm])
            except np.linalg.LinAlgError:
                self.A_inv_cholesky[arm] = np.eye(self.d) / np.sqrt(self.reg)
        except np.linalg.LinAlgError:
            self.A_inv[arm] = np.eye(self.d) / self.reg
            self.theta[arm] = np.zeros(self.d)
            self.A_inv_cholesky[arm] = np.eye(self.d) / np.sqrt(self.reg)

    def select(self, context):
        # Check deadline
        if time.time() - self.start_time > self.deadline:
            context = np.asarray(context, dtype=float)
            means = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(means))
        
        context = np.asarray(context, dtype=float)
        scores = []
        
        for a in range(self.n):
            theta_mean = self.theta[a]
            
            # Sample theta from posterior N(theta_mean, sigma^2 * A_inv[a])
            # Use pre-computed Cholesky of A_inv for efficiency
            try:
                L = self.A_inv_cholesky[a]
                z = np.random.randn(self.d)
                # Covariance = sigma^2 * A_inv, so std = sigma * sqrt(A_inv)
                # Sample: theta = theta_mean + sigma * L @ z
                w = L @ z
                theta_sample = theta_mean + np.sqrt(self.sigma_sq) * w
            except Exception:
                theta_sample = theta_mean
            
            score = theta_sample @ context
            scores.append(score)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=float)
        reward = float(reward)
        
        # 1. Compute residual
        pred = self.theta[arm] @ context
        residual = reward - pred
        residual_sq = residual ** 2
        
        # 2. Update EWMA of squared residuals
        self.residual_sq_ewma[arm] = (1 - self.ewma_alpha) * self.residual_sq_ewma[arm] + self.ewma_alpha * residual_sq
        
        # 3. Update adaptive gamma
        self._update_gamma(arm, residual_sq)
        
        # 4. Update statistics with forgetting using SMMW
        gamma = self.gamma[arm]
        
        # A_new = gamma * A_old + x x^T
        # b_new = gamma * b_old + reward * x
        
        # For A_inv, use Sherman-Morrison-Woodbury:
        # Let M = gamma * A_old, so M^{-1} = (1/gamma) * A_old^{-1}
        # A_new = M + x x^T
        # A_new^{-1} = M^{-1} - M^{-1} x x^T M^{-1} / (1 + x^T M^{-1} x)
        
        M_inv = (1.0 / gamma) * self.A_inv[arm]
        M_inv_x = M_inv @ context
        denom = 1.0 + context @ M_inv_x
        
        if abs(denom) > 1e-10:
            # A_new^{-1} = M_inv - M_inv @ x @ x^T @ M_inv / denom
            self.A_inv[arm] = M_inv - np.outer(M_inv_x, M_inv_x) / denom
        else:
            # Fallback to full inversion
            self.A[arm] = gamma * self.A[arm] + np.outer(context, context)
            try:
                self.A_inv[arm] = np.linalg.inv(self.A[arm])
            except np.linalg.LinAlgError:
                self.A_inv[arm] = np.eye(self.d) / self.reg
        
        # Update A and b for future reference
        self.A[arm] = gamma * self.A[arm] + np.outer(context, context)
        self.b[arm] = gamma * self.b[arm] + reward * context
        
        # 5. Recompute theta
        self.theta[arm] = self.A_inv[arm] @ self.b[arm]
        
        # 6. Periodically re-invert to prevent numerical drift
        if self.t % self.reinv_interval == 0 and self.t > 0:
            self._recompute_arm_stats(arm)
        
        self.t += 1
# EVOLVE-BLOCK-END
