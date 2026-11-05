"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit with:
    - Exponentially weighted linear regression (EWLR) with adaptive forgetting
    - Sherman-Morrison-like O(d^2) updates for inverse covariance
    - UCB exploration with adaptive beta (faster early growth)
    - Deadline guard for safety
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self.start_time = time.time()
        
        # Per-arm state: A_inv (d x d), b (d,)
        # Initialize with small ridge for numerical stability
        ridge = 0.1
        self.A_inv = [np.eye(dim) * ridge for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # For adaptive forgetting: track per-arm prediction error EMA
        self.reward_ema = [0.0 for _ in range(n_arms)]
        self.pred_ema = [0.0 for _ in range(n_arms)]
        self.error_ema = [1.0 for _ in range(n_arms)]
        
        # Adaptive forgetting factor - tuned range
        self.min_gamma = 0.80
        self.max_gamma = 0.98
        
        # Exploration parameter - higher base for more early exploration
        self.beta0 = 1.5
        self.counts = [0 for _ in range(n_arms)]
        
        # Deadline safety
        self.deadline = 145.0  # seconds from start, leave 5s margin

    def _get_gamma(self, arm):
        """Adaptive forgetting factor based on prediction error."""
        err = self.error_ema[arm]
        # Use a smoother mapping: gamma = max_gamma - range * sigmoid-like function of err
        # Normalize error to [0,1] using a soft threshold
        norm_err = min(err / 1.5, 1.0)
        # Smooth S-curve mapping
        gamma = self.max_gamma - (self.max_gamma - self.min_gamma) * (norm_err / (1.0 + norm_err))
        return max(self.min_gamma, min(self.max_gamma, gamma))

    def select(self, context):
        self.t += 1
        context = np.asarray(context, dtype=np.float64)
        
        # Deadline check
        if time.time() - self.start_time > self.deadline:
            return 0
        
        scores = np.zeros(self.n)
        for a in range(self.n):
            if self.counts[a] == 0:
                # No data yet: pure exploration with high uncertainty
                scores[a] = self.beta0 * np.sqrt(context @ self.A_inv[a] @ context)
            else:
                theta = self.A_inv[a] @ self.b[a]
                mu = theta @ context
                var = context @ self.A_inv[a] @ context
                # Faster-growing beta: quadratic early, then linear
                if self.t < 100:
                    beta = self.beta0 * np.sqrt(1.0 + self.t * 0.01)
                else:
                    beta = self.beta0 * np.sqrt(1.0 + 100 * 0.01) * (1.0 + (self.t - 100) * 0.0005)
                scores[a] = mu + beta * np.sqrt(max(var, 1e-8))
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        self.counts[arm] += 1
        
        # Compute prediction BEFORE update for error tracking
        theta = self.A_inv[arm] @ self.b[arm]
        pred = theta @ context
        self.reward_ema[arm] = 0.9 * self.reward_ema[arm] + 0.1 * reward
        self.pred_ema[arm] = 0.9 * self.pred_ema[arm] + 0.1 * pred
        self.error_ema[arm] = 0.9 * self.error_ema[arm] + 0.1 * (reward - pred) ** 2
        
        # Adaptive forgetting factor
        gamma = self._get_gamma(arm)
        
        # Update b with forgetting
        self.b[arm] = gamma * self.b[arm] + reward * context
        
        # Update A_inv using Sherman-Morrison-like approach with forgetting
        A_inv_old = self.A_inv[arm]
        Ax = A_inv_old @ context
        denom = 1.0 + (1.0 / gamma) * (context @ Ax)
        
        A_inv_new = (1.0 / gamma) * A_inv_old - (1.0 / (gamma * gamma)) * np.outer(Ax, Ax) / denom
        
        # Symmetrize to reduce numerical drift
        A_inv_new = 0.5 * (A_inv_new + A_inv_new.T)
        
        self.A_inv[arm] = A_inv_new
# EVOLVE-BLOCK-END
