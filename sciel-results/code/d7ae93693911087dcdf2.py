# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque


class Policy:
    """Non-stationary linear bandit with Sliding Window Thompson Sampling."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.window_size = 100  # Sliding window size for non-stationarity
        self.lambda_reg = 1.0   # Regularization parameter
        self.sigma = 1.0        # Noise variance
        self.history = [deque(maxlen=self.window_size) for _ in range(n_arms)]
        self.cached_V = [None for _ in range(n_arms)]
        self.cached_theta = [None for _ in range(n_arms)]
        self.dirty = [True for _ in range(n_arms)]
        self._t0 = time.time()
        self._deadline = self._t0 + 145.0

    def _recompute_params(self, arm):
        """Recompute V and theta_mean for the given arm from its history."""
        if len(self.history[arm]) == 0:
            self.cached_V[arm] = np.eye(self.d) * self.sigma**2
            self.cached_theta[arm] = np.zeros(self.d)
            return
        
        contexts = np.array([h[0] for h in self.history[arm]])
        rewards = np.array([h[1] for h in self.history[arm]])
        
        # A = lambda * I + contexts.T @ contexts
        A = self.lambda_reg * np.eye(self.d) + contexts.T @ contexts
        # b = contexts.T @ rewards
        b = contexts.T @ rewards
        
        try:
            V = np.linalg.inv(A)
            theta_mean = V @ b
            self.cached_V[arm] = V
            self.cached_theta[arm] = theta_mean
        except np.linalg.LinAlgError:
            self.cached_V[arm] = np.eye(self.d) * self.sigma**2
            self.cached_theta[arm] = np.zeros(self.d)

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)
        scores = np.zeros(self.n)
        
        # Check deadline
        if time.time() > self._deadline:
            return int(np.argmax(scores)) # Fallback to random if timeout

        for a in range(self.n):
            if self.dirty[a]:
                self._recompute_params(a)
                self.dirty[a] = False
            
            V = self.cached_V[a]
            theta_mean = self.cached_theta[a]
            
            # Sample theta from N(theta_mean, sigma^2 * V)
            # Note: If history is empty, V is sigma^2 * I, theta_mean is 0.
            # This matches the prior N(0, sigma^2 I).
            
            try:
                L = np.linalg.cholesky(V)
                z = np.random.randn(self.d)
                theta_sample = theta_mean + self.sigma * (L @ z)
            except np.linalg.LinAlgError:
                # Fallback if V is not positive definite
                theta_sample = theta_mean + self.sigma * np.random.randn(self.d)
            
            scores[a] = theta_sample @ context
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        self.history[arm].append((context, reward))
        self.dirty[arm] = True
# EVOLVE-BLOCK-END
