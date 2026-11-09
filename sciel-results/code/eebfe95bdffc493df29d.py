import numpy as np
import time

class Policy:
    """
    Non-stationary Linear Contextual Bandit Policy using Discounted UCB.
    
    Strategy:
    1. Per-arm Linear Regression models with a fixed, aggressive discount factor.
       Unlike the adaptive error-based decay in the parent, this uses a consistent
       exponential decay (alpha) to ensure the model always represents the recent
       statistical regime. This is more robust to noise in individual error signals.
    2. Action Selection via Upper Confidence Bound (UCB):
       score = theta^T x + beta * sqrt(x^T A^{-1} x)
       Here, beta is a fixed exploration parameter. The uncertainty term
       sqrt(x^T A^{-1} x) naturally increases as data is discounted (A shrinks),
       forcing exploration when the model becomes stale. This is deterministic
       and avoids the variance of Thompson Sampling.
    3. Global Non-stationarity Awareness:
       We monitor the global prediction error. If the global error spikes, we
       temporarily increase the exploration bonus (beta) to accelerate recovery,
       acting as a "reset" mechanism without fully clearing the memory.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Initialize A with regularization
        # Lambda controls initial confidence. Smaller lambda = higher initial uncertainty.
        self.reg = 1.0
        self.A = [np.eye(dim) * self.reg for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Discount factor: 0.99 is a common choice for non-stationary linear bandits.
        # It gives a "memory" of roughly 100 rounds.
        self.alpha = 0.99
        
        # UCB Exploration Parameter
        # In standard UCB, beta is often sqrt(log(t)) or a constant.
        # For non-stationary, a constant is safer to avoid over-exploration early on,
        # but we need it high enough to explore when variance is high.
        self.beta = 1.5
        
        # Global Error Tracking for Adaptive Beta
        self.global_error_ema = 0.0
        self.error_ema_alpha = 0.1
        self.base_beta = 1.5
        self.max_beta = 3.0
        
        # Time guard
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0

    def _get_theta_and_cov(self, A, b):
        """
        Compute theta = A^-1 b and the covariance matrix A^-1.
        """
        try:
            theta = np.linalg.solve(A, b)
            A_inv = np.linalg.inv(A)
            return theta, A_inv
        except np.linalg.LinAlgError:
            return np.zeros(self.d), np.eye(self.d)

    def select(self, context):
        # Deadline check
        if time.time() > self.deadline:
            return 0 # Fallback to first arm to avoid timeout
            
        context = np.asarray(context)
        
        best_arm = 0
        best_score = -np.inf
        
        # Dynamically adjust beta based on global error
        # If global error is high, we increase exploration
        err_norm = min(max(self.global_error_ema / 2.0, 0.0), 1.0)
        current_beta = self.base_beta + (self.max_beta - self.base_beta) * err_norm

        for a in range(self.n):
            try:
                theta, A_inv = self._get_theta_and_cov(self.A[a], self.b[a])
                
                # Expected reward
                mean_reward = theta @ context
                
                # Uncertainty
                # x^T A^-1 x is the variance of the reward estimate
                variance = context @ A_inv @ context
                if variance < 0:
                    variance = 0.0
                std_dev = np.sqrt(variance)
                
                # UCB Score
                score = mean_reward + current_beta * std_dev
                
                if score > best_score:
                    best_score = score
                    best_arm = a
                    
            except Exception:
                # If computation fails, fall back to mean
                try:
                    theta, _ = self._get_theta_and_cov(self.A[a], self.b[a])
                    score = theta @ context
                    if score > best_score:
                        best_score = score
                        best_arm = a
                except Exception:
                    continue

        return int(best_arm)

    def update(self, context, arm, reward):
        context = np.asarray(context)
        
        # 1. Compute prediction error for the selected arm
        try:
            theta = np.linalg.solve(self.A[arm], self.b[arm])
            predicted_reward = theta @ context
            error = reward - predicted_reward
            error_abs = abs(error)
        except Exception:
            error_abs = 1.0

        # 2. Update Global Error EMA
        self.global_error_ema = (1 - self.error_ema_alpha) * self.global_error_ema + self.error_ema_alpha * error_abs

        # 3. Update Model for the selected arm with discounting
        # This is the key structural change: consistent discounting rather than
        # error-adapted discounting.
        self.A[arm] = self.alpha * self.A[arm] + np.outer(context, context)
        self.b[arm] = self.alpha * self.b[arm] + reward * context
        
        # 4. Discount Other Arms
        # We must also decay the other arms' models to keep them "recent".
        # If we don't, their A matrices will remain large (low uncertainty)
        # even though we haven't seen them recently, leading to under-exploration.
        for a in range(self.n):
            if a != arm:
                self.A[a] = self.alpha * self.A[a]
                self.b[a] = self.alpha * self.b[a]

        # 5. Time guard
        if time.time() > self.deadline:
            pass
