"""Non-stationary linear contextual bandit with adaptive Thompson Sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Adaptive Thompson Sampling for non-stationary linear bandits.
    
    Combines:
    - Thompson Sampling for exploration (sampling from posterior)
    - Adaptive forgetting based on Mahalanobis distance of residuals
    - Efficient matrix operations with deadline guard
    - Smooth decay modulation via EMA of normalized residuals
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.I = np.eye(dim)
        
        # Per-arm precision and cross-moment
        self.A = [self.I.copy() for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Adaptive forgetting parameters
        self.base_decay = 0.99       # mild decay for normal operation
        self.shift_decay = 0.45      # strong forgetting on regime shift
        self.shift_threshold = 2.0   # residual/uncertainty threshold
        self.t = 0
        
        # Track recent regime-shift signals for smoothing
        self.shift_signal = [0.0 for _ in range(n_arms)]
        self.shift_alpha = 0.3       # smoothing factor for shift signal
        
        # Time management
        self.start_time = time.time()
        self.deadline = self.start_time + 148.0  # Leave 2s safety margin
        
        # Exploration scale for Thompson Sampling
        self.exploration_scale = 1.0

    def _safe(self):
        """Check if we're near the deadline."""
        return time.time() >= self.deadline

    def _cholesky_inv(self, M):
        """Stable inverse via Cholesky, with fallback to pinv."""
        try:
            L = np.linalg.cholesky(M)
            inv_L = np.linalg.inv(L)
            inv_M = inv_L.T @ inv_L
            return inv_M
        except np.linalg.LinAlgError:
            return np.linalg.pinv(M)

    def _compute_decay(self, arm, normalized_residual):
        """
        Compute adaptive decay: strong decay on regime shift, mild otherwise.
        Uses a smoothed shift signal to avoid oscillation.
        """
        # Binary shift indicator: 1 if residual exceeds threshold
        is_shift = 1.0 if normalized_residual > self.shift_threshold else 0.0
        
        # Smooth the shift signal to avoid rapid oscillation
        self.shift_signal[arm] = (1 - self.shift_alpha) * self.shift_signal[arm] + \
                                  self.shift_alpha * is_shift
        
        # Interpolate between base and shift decay based on smoothed signal
        decay = self.base_decay + (self.shift_decay - self.base_decay) * self.shift_signal[arm]
        
        return decay

    def select(self, context):
        # Deadline guard: return greedy if near deadline
        if self._safe():
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                A_inv = self._cholesky_inv(self.A[a])
                theta = A_inv @ self.b[a]
                mu = theta @ context
                if mu > best_score:
                    best_score = mu
                    best_arm = a
            return best_arm
        
        # Thompson Sampling: sample from posterior for each arm
        scores = np.empty(self.n)
        ctx = np.asarray(context)
        for a in range(self.n):
            A_inv = self._cholesky_inv(self.A[a])
            theta_mean = A_inv @ self.b[a]
            mu = theta_mean @ ctx
            
            # Sample theta from N(theta_mean, A_inv)
            # Use Cholesky of A_inv for sampling with ridge for stability
            try:
                ridge = 1e-6 * np.eye(self.d)
                L = np.linalg.cholesky(A_inv + ridge)
                z = np.random.randn(self.d)
                theta_sample = theta_mean + L @ z
                scores[a] = theta_sample @ ctx
            except np.linalg.LinAlgError:
                # Fallback to UCB if sampling fails
                var = ctx @ A_inv @ ctx
                std = np.sqrt(max(var, 1e-12))
                scores[a] = mu + 1.2 * std
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.t += 1
        
        # Compute current model prediction and uncertainty
        A_inv = self._cholesky_inv(self.A[arm])
        theta = A_inv @ self.b[arm]
        mu = theta @ context
        var = context @ A_inv @ context
        std = np.sqrt(max(var, 1e-12))
        
        # Normalized residual (Mahalanobis-like distance)
        if std > 1e-8:
            normalized_residual = abs(reward - mu) / std
        else:
            normalized_residual = abs(reward - mu)
        
        # Compute adaptive decay
        decay = self._compute_decay(arm, normalized_residual)
        
        # Apply forgetting
        self.A[arm] = decay * self.A[arm]
        self.b[arm] = decay * self.b[arm]
        
        # Add new observation
        ctx = np.asarray(context)
        self.A[arm] = self.A[arm] + np.outer(ctx, ctx)
        self.b[arm] = self.b[arm] + reward * ctx
        
        # Ensure A remains well-conditioned: add small ridge if needed
        # (cheap check every few steps to avoid overhead)
        if self.t % 25 == 0:
            try:
                np.linalg.cholesky(self.A[arm])
            except np.linalg.LinAlgError:
                self.A[arm] += 1e-4 * np.eye(self.d)
# EVOLVE-BLOCK-END
