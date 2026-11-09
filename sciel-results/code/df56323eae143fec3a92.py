"""Combined approach: Adaptive Sliding Window LinUCB with Thompson Sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

class Policy:
    """
    Adaptive Sliding Window LinUCB with Thompson Sampling for non-stationary linear bandits.
    
    Combines:
    - Sliding window approach (from inspiration 1) for fast adaptation to non-stationarity
    - Thompson Sampling (from current program) for better exploration-exploitation balance
    - Adaptive window sizing based on residual variance
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        
        # Sliding window parameters
        self.max_window = 100
        self.base_window = 50
        self.min_window = 10
        self.reg = 1.0          # Regularization for stability
        
        # Adaptive window parameters
        self.residual_scale = 10.0
        self.ewma_alpha = 0.1
        
        # Thompson Sampling noise variance
        self.sigma_sq = 1.0
        
        # Data structures: deques to store (context, reward) for each arm
        self.windows = [deque(maxlen=self.max_window) for _ in range(n_arms)]
        
        # Per-arm statistics
        self.A = [self.reg * np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.A_inv = [np.eye(dim) / self.reg for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        
        # Non-stationarity detection
        self.residual_sq_ewma = [0.0 for _ in range(n_arms)]
        
        # Deadline guard
        self.start_time = time.time()
        self.deadline = 115.0  # Safety margin before 150s limit

    def _effective_window_size(self, arm):
        """Compute effective window size based on residual EWMA."""
        W = len(self.windows[arm])
        if W == 0:
            return 0
        
        residual_factor = self.residual_sq_ewma[arm]
        # Higher residuals -> shorter window (faster forgetting)
        effective_window = int(self.base_window / (1.0 + self.residual_scale * residual_factor))
        effective_window = max(self.min_window, effective_window)
        effective_window = min(effective_window, W, self.max_window)
        return effective_window

    def _recompute_arm_stats(self, arm):
        """Recompute A, b, A_inv, theta for a specific arm based on its current window."""
        W = len(self.windows[arm])
        
        if W == 0:
            self.A[arm] = self.reg * np.eye(self.d)
            self.b[arm] = np.zeros(self.d)
            self.A_inv[arm] = np.eye(self.d) / self.reg
            self.theta[arm] = np.zeros(self.d)
            return

        effective_window = self._effective_window_size(arm)
        
        # Get the most recent 'effective_window' samples
        all_data = list(self.windows[arm])
        recent_data = all_data[-effective_window:]
        
        A = self.reg * np.eye(self.d)
        b = np.zeros(self.d)
        
        for x, r in recent_data:
            A += np.outer(x, x)
            b += r * x
            
        self.A[arm] = A
        self.b[arm] = b
        
        try:
            A_inv = np.linalg.inv(A)
            self.A_inv[arm] = A_inv
            self.theta[arm] = A_inv @ b
        except np.linalg.LinAlgError:
            self.A_inv[arm] = np.eye(self.d) / self.reg
            self.theta[arm] = np.zeros(self.d)

    def select(self, context):
        # Check deadline
        if time.time() - self.start_time > self.deadline:
            # Fallback: return arm with highest mean prediction
            context = np.asarray(context, dtype=float)
            means = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(means))
        
        context = np.asarray(context, dtype=float)
        scores = []
        
        for a in range(self.n):
            # Compute posterior mean
            theta_mean = self.theta[a]
            
            # Sample theta from posterior N(theta_mean, sigma^2 * A_inv[a])
            # Using Cholesky decomposition for efficient sampling
            try:
                L = np.linalg.cholesky(self.A[a])
                z = np.random.randn(self.d)
                # Solve L^T w = z, then sample = theta_mean + sigma_sq * w
                w = np.linalg.solve(L.T, z)
                theta_sample = theta_mean + self.sigma_sq * w
            except np.linalg.LinAlgError:
                # Fallback to mean if Cholesky fails
                theta_sample = theta_mean
            
            score = theta_sample @ context
            scores.append(score)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=float)
        reward = float(reward)
        
        # 1. Compute residual using current estimates BEFORE updating
        pred = self.theta[arm] @ context
        residual = reward - pred
        residual_sq = residual ** 2
        
        # Update EWMA of squared residuals
        self.residual_sq_ewma[arm] = (1 - self.ewma_alpha) * self.residual_sq_ewma[arm] + self.ewma_alpha * residual_sq
        
        # 2. Add to window
        self.windows[arm].append((context.copy(), reward))
        
        # 3. Recompute stats for this arm
        self._recompute_arm_stats(arm)
        
        self.t += 1
# EVOLVE-BLOCK-END
