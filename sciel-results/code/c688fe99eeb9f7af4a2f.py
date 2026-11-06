# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

class Policy:
    """
    Dismissal LinUCB for non-stationary linear bandits.
    
    Uses a per-arm dismissal mechanism: if the recent prediction error (MSE) for an
    arm exceeds a threshold, all data for that arm is cleared. This is a simple
    and effective method for handling abrupt changes in the underlying model,
    which is common in non-stationary settings.
    
    Efficiency:
    - Maintains A (covariance) and b (moment) incrementally with a forgetting factor.
    - Uses Sherman-Morrison-Woodbury for efficient inverse updates.
    - Fast select() with pre-computed theta and inv_A.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Forgetting factor for linear regression
        # A value of 1.0 means no forgetting, < 1.0 means exponential forgetting
        self.gamma = 0.98
        
        # Regularization parameter
        self.beta = 1.0
        
        # Exploration parameter
        self.alpha = 2.0
        
        # Dismissal parameters
        self.dismissal_threshold = 2.0  # MSE threshold for triggering dismissal
        self.dismissal_window = 20      # Number of recent samples to consider for MSE
        
        # Per-arm storage
        self.A = [np.eye(dim) * self.beta for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.inv_A = [np.eye(dim) / self.beta for _ in range(n_arms)]
        
        # Deques to store recent (context, reward) pairs for MSE calculation
        self.recent_data = [deque(maxlen=self.dismissal_window) for _ in range(n_arms)]
        
        # Deadline guard
        self._start_time = time.time()
        self._deadline = self._start_time + 140.0

    def _elapsed(self):
        return time.time() - self._start_time

    def _smw_update(self, arm):
        """
        Update inv_A using Sherman-Morrison-Woodbury formula.
        This is more efficient than computing the full inverse.
        """
        # A_new = A_old * gamma + x x^T
        # inv(A_new) = (A_old^{-1} * gamma^{-1} - x x^T) / (1 + x^T A_old^{-1} x / gamma)
        # But this is for A_new = gamma * A_old + x x^T
        # Let's use the standard SMW formula for rank-1 update
        # If A_new = A_old + v v^T, then inv(A_new) = inv(A_old) - inv(A_old) v v^T inv(A_old) / (1 + v^T inv(A_old) v)
        
        # Actually, with forgetting, it's:
        # A_new = gamma * A_old + x x^T
        # We can write this as A_new = gamma * (A_old + (1/gamma) x x^T)
        # So inv(A_new) = (1/gamma) * inv(A_old + (1/sqrt(gamma)) x (1/sqrt(gamma)) x^T)
        # Let v = x / sqrt(gamma)
        # inv(A_old + v v^T) = inv(A_old) - inv(A_old) v v^T inv(A_old) / (1 + v^T inv(A_old) v)
        
        x = self._last_context[arm]
        if x is None:
            return
            
        v = x / np.sqrt(self.gamma)
        inv_A_old = self.inv_A[arm]
        
        # v^T inv_A_old v
        tmp = inv_A_old @ v
        denom = 1.0 + float(v @ tmp)
        
        # inv(A_old + v v^T)
        inv_A_new_base = inv_A_old - np.outer(tmp, tmp) / denom
        
        # inv(A_new) = (1/gamma) * inv(A_new_base)
        self.inv_A[arm] = (1.0 / self.gamma) * inv_A_new_base
        
        # Update A and b
        self.A[arm] = self.gamma * self.A[arm] + np.outer(x, x)
        reward = self._last_reward[arm]
        self.b[arm] = self.gamma * self.b[arm] + reward * x
        
        # Update theta
        self.theta[arm] = self.inv_A[arm] @ self.b[arm]

    def _check_dismissal(self, arm, context, reward):
        """
        Check if the recent prediction error for this arm exceeds the threshold.
        If so, clear all data for this arm.
        """
        if len(self.recent_data[arm]) < self.dismissal_window:
            return False
            
        # Calculate MSE over recent data
        mse = 0.0
        for x, r in self.recent_data[arm]:
            pred = float(self.theta[arm] @ x)
            mse += (r - pred) ** 2
        mse /= self.dismissal_window
        
        if mse > self.dismissal_threshold:
            # Dismiss all data
            self.A[arm] = np.eye(self.d) * self.beta
            self.b[arm] = np.zeros(self.d)
            self.theta[arm] = np.zeros(self.d)
            self.inv_A[arm] = np.eye(self.d) / self.beta
            self.recent_data[arm].clear()
            return True
            
        return False

    def select(self, context):
        # Deadline check
        if self._elapsed() > self._deadline:
            # Fallback: greedy
            context = np.asarray(context, dtype=np.float64)
            scores = np.array([float(self.theta[a] @ context) for a in range(self.n)])
            return int(np.argmax(scores))

        context = np.asarray(context, dtype=np.float64)
        scores = np.empty(self.n)
        
        for a in range(self.n):
            # Mean prediction
            mu = float(self.theta[a] @ context)
            
            # Variance term
            var = float(context @ self.inv_A[a] @ context)
            std = np.sqrt(max(var, 0.0))
            
            # UCB score
            scores[a] = mu + self.alpha * std
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        
        # Deadline check
        if self._elapsed() > self._deadline:
            return

        # Store last context and reward for SMW update
        if not hasattr(self, '_last_context'):
            self._last_context = [None] * self.n
            self._last_reward = [0.0] * self.n
        self._last_context[arm] = context
        self._last_reward[arm] = reward
        
        # Check for dismissal
        self._check_dismissal(arm, context, reward)
        
        # Add to recent data
        self.recent_data[arm].append((context, reward))
        
        # Update model
        self._smw_update(arm)

# EVOLVE-BLOCK-END
