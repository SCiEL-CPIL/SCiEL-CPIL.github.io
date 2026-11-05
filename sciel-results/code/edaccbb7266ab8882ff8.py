# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque


class Policy:
    """
    Non-stationary linear bandit with Exponentially Weighted Least Squares (EWLS)
    and Thompson Sampling.
    
    Key structural changes from sliding window:
    1. Exponential forgetting instead of hard window: retains some long-term
       signal while adapting to non-stationarity.
    2. Per-arm discount factor tuned for adaptation speed.
    3. Thompson Sampling for exploration-exploitation balance.
    4. Efficient incremental updates of A and b matrices.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.discount = 0.95  # Exponential forgetting factor (higher = more memory)
        self.lambda_reg = 0.5  # Regularization
        self.sigma = 1.0  # Noise standard deviation
        
        # Per-arm state for EWLS
        self.A = [self.lambda_reg * np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.counts = [0 for _ in range(n_arms)]
        
        # Cache for Cholesky factorization
        self._chol_cache = [None for _ in range(n_arms)]
        self._chol_valid = [False for _ in range(n_arms)]
        
        # Time management
        self._t0 = time.time()
        self._deadline = self._t0 + 140.0  # Safety margin before 150s kill
        
        # Track last update time per arm for adaptive discount
        self._last_update = [0.0] * n_arms

    def _update_ewls(self, arm, context, reward):
        """Incrementally update A and b with exponential forgetting."""
        # Decay existing state
        self.A[arm] *= self.discount
        self.b[arm] *= self.discount
        
        # Add new observation
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
        
        self.counts[arm] += 1
        self._chol_valid[arm] = False  # Invalidate cache

    def _get_theta_sample(self, arm, context):
        """Get a Thompson sample for the given arm and context."""
        A = self.A[arm]
        b = self.b[arm]
        
        # Compute theta_mean = A^{-1} b
        # Use Cholesky for stability
        if not self._chol_valid[arm]:
            try:
                L = np.linalg.cholesky(A)
                self._chol_cache[arm] = L
                self._chol_valid[arm] = True
            except np.linalg.LinAlgError:
                # Fallback: add more regularization
                A = A + 1e-6 * np.eye(self.d)
                try:
                    L = np.linalg.cholesky(A)
                    self._chol_cache[arm] = L
                    self._chol_valid[arm] = True
                except np.linalg.LinAlgError:
                    # Last resort: use pseudoinverse
                    theta_mean = np.linalg.pinv(A) @ b
                    return theta_mean @ context + self.sigma * np.random.randn()
        
        L = self._chol_cache[arm]
        
        # theta_mean = A^{-1} b = (L L^T)^{-1} b = L^{-T} L^{-1} b
        # Solve L x = b, then L^T theta_mean = x
        x = np.linalg.solve(L, b)
        theta_mean = np.linalg.solve(L.T, x)
        
        # Covariance = sigma^2 * A^{-1} = sigma^2 * (L L^T)^{-1}
        # Sample: theta = theta_mean + sigma * L^{-T} z where z ~ N(0, I)
        # Actually: A^{-1} = (L L^T)^{-1} = L^{-T} L^{-1}
        # So Cov = sigma^2 * L^{-T} L^{-1}
        # We can write: Cov = (sigma * L^{-T}) (sigma * L^{-1})^T
        # Sample: z ~ N(0, I), theta = theta_mean + sigma * L^{-T} z
        
        z = np.random.randn(self.d)
        # Solve L^T w = z
        w = np.linalg.solve(L.T, z)
        theta_sample = theta_mean + self.sigma * w
        
        return theta_sample @ context

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)
        
        # Check deadline
        if time.time() > self._deadline:
            # Fallback: choose arm with highest count (exploit)
            if self.counts:
                return int(np.argmax(self.counts))
            return int(np.random.randint(self.n))
        
        scores = np.zeros(self.n)
        for a in range(self.n):
            scores[a] = self._get_theta_sample(a, context)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        self._update_ewls(arm, context, reward)
# EVOLVE-BLOCK-END
