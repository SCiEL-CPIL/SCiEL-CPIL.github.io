"""Non-stationary linear contextual bandit with sliding-window ridge regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear bandit using per-arm sliding-window ridge regression.
    
    For each arm, we maintain a sliding window of recent (context, reward) pairs.
    We fit a ridge regression model on the window to estimate the arm-specific
    weight vector. For exploration, we use an optimistic bonus based on the
    uncertainty from the window's covariance structure.
    
    The sliding window discards stale observations, allowing faster adaptation
    to non-stationary arm weights compared to exponential forgetting.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.window_size = 150  # Number of recent observations to keep per arm
        self.ridge_lambda = 1.0  # Ridge regularization strength
        
        # Per-arm storage of recent observations
        self.contexts = [[] for _ in range(n_arms)]
        self.rewards = [[] for _ in range(n_arms)]
        
        # Precompute identity for efficiency
        self.eye_d = np.eye(dim)
        
        # Deadline guard
        self.start_time = time.time()
        self.deadline = 140.0  # Safety margin before 150s limit

    def _fit_ridge(self, arm):
        """Fit ridge regression on the sliding window for the given arm."""
        X = self.contexts[arm]
        y = self.rewards[arm]
        
        if len(X) == 0:
            return np.zeros(self.d), self.eye_d
        
        X_arr = np.array(X)
        y_arr = np.array(y)
        
        # Ridge regression: theta = (X^T X + lambda * I)^{-1} X^T y
        XtX = X_arr.T @ X_arr
        Xty = X_arr.T @ y_arr
        A = XtX + self.ridge_lambda * self.eye_d
        
        try:
            theta = np.linalg.solve(A, Xty)
            # Uncertainty: inverse of A (used for optimistic bonus)
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            theta = np.zeros(self.d)
            A_inv = self.eye_d / self.ridge_lambda
        
        return theta, A_inv

    def select(self, context):
        # Deadline guard
        if time.time() - self.start_time > self.deadline:
            return 0  # Safe fallback
        
        context = np.asarray(context, dtype=np.float64)
        scores = []
        
        for a in range(self.n):
            theta, A_inv = self._fit_ridge(a)
            # Predicted reward
            pred = theta @ context
            # Optimistic bonus: sqrt(context^T A_inv context)
            bonus = 0.5 * np.sqrt(context @ A_inv @ context)
            scores.append(pred + bonus)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        
        # Add to sliding window
        self.contexts[arm].append(context)
        self.rewards[arm].append(reward)
        
        # Trim window if it exceeds window_size
        if len(self.contexts[arm]) > self.window_size:
            self.contexts[arm].pop(0)
            self.rewards[arm].pop(0)
# EVOLVE-BLOCK-END
