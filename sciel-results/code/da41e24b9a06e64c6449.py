"""
Recursive Least Squares with Exponential Forgetting + Thompson Sampling
for Non-Stationary Linear Bandits.

Uses per-arm RLS with exponential forgetting for O(1) memory and O(d^2) updates,
combined with Thompson Sampling for exploration. This provides smooth adaptation
to non-stationarity without the abrupt information loss of sliding windows.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    def __init__(self, n_arms, dim):
        self.n_arms = n_arms
        self.dim = dim
        self.t = 0
        
        # RLS parameters
        self.lam = 0.95          # Forgetting factor (closer to 1 = slower forgetting)
        self.reg = 1.0           # Initial regularization
        self.noise_floor = 0.05  # Minimum noise variance
        self.alpha = 0.1         # Step size scaling for RLS
        
        # Per-arm state: P (covariance), theta (weight estimate), sigma_sq (noise)
        self.P = [np.eye(dim) for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.sigma_sq = [1.0 for _ in range(n_arms)]
        self.counts = [0 for _ in range(n_arms)]
        
        # Deadline guard
        self.start_time = time.time()
        self.deadline = 115.0
        
        # RNG for Thompson Sampling
        self.rng = np.random.default_rng(42)
        
        # Cache for fast select
        self._cached_scores = None
        self._cached_context = None

    def _rls_update(self, arm, context, reward):
        """
        Update RLS state for the given arm using exponential forgetting.
        O(d^2) per update.
        """
        d = self.dim
        x = np.asarray(context, dtype=np.float64)
        r = float(reward)
        
        P = self.P[arm]
        theta = self.theta[arm]
        lam = self.lam
        reg = self.reg
        alpha = self.alpha
        
        # Exponential forgetting: decay P
        P = P / lam
        
        # RLS gain
        Px = P @ x
        denom = 1.0 + alpha * x @ Px
        K = Px / denom  # gain vector
        
        # Update theta
        pred = theta @ x
        err = r - pred
        self.theta[arm] = theta + K * err
        
        # Update P (Joseph form for numerical stability)
        I_Kx = np.eye(d) - np.outer(K, x)
        self.P[arm] = (I_Kx @ P) / lam
        # Add small regularization to prevent singularity
        self.P[arm] += 1e-6 * np.eye(d)
        
        # Update noise estimate
        self.sigma_sq[arm] = 0.9 * self.sigma_sq[arm] + 0.1 * (err ** 2)
        if self.sigma_sq[arm] < self.noise_floor:
            self.sigma_sq[arm] = self.noise_floor
        
        self.counts[arm] += 1

    def select(self, context):
        # Check deadline
        if time.time() - self.start_time > self.deadline:
            return 0
        
        context = np.asarray(context, dtype=np.float64)
        d = self.dim
        n_arms = self.n_arms
        
        scores = np.zeros(n_arms)
        
        for a in range(n_arms):
            theta = self.theta[a]
            P = self.P[a]
            sigma_sq = self.sigma_sq[a]
            
            # Thompson Sampling: sample from posterior N(theta, sigma_sq * P)
            # Use Cholesky for efficiency
            try:
                L = np.linalg.cholesky(P + 1e-8 * np.eye(d))
                z = self.rng.standard_normal(d)
                theta_sample = theta + np.sqrt(sigma_sq) * (L @ z)
            except np.linalg.LinAlgError:
                # Fallback: use eigenvalue decomposition
                try:
                    eigvals, eigvecs = np.linalg.eigh(P)
                    eigvals = np.maximum(eigvals, 0)
                    L = eigvecs @ np.diag(np.sqrt(eigvals))
                    z = self.rng.standard_normal(d)
                    theta_sample = theta + np.sqrt(sigma_sq) * (L @ z)
                except:
                    theta_sample = theta
            
            scores[a] = context @ theta_sample
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self._rls_update(arm, context, reward)
        self.t += 1
# EVOLVE-BLOCK-END
