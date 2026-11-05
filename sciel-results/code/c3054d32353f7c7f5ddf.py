"""Non-stationary linear contextual bandit with adaptive Kalman filter per arm."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Adaptive Kalman filter for non-stationary linear bandits.
    
    Each arm maintains a state vector theta (weights) and covariance P.
    The state evolves as a random walk: theta_t = theta_{t-1} + w_t, w_t ~ N(0, Q).
    Observation: reward = theta_t^T x + noise, noise ~ N(0, R).
    
    We use an adaptive process noise Q that scales with the prediction error,
    allowing the filter to respond quickly to distribution shifts while remaining
    stable when the environment is stationary.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Initialize each arm's state
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        # Start with large uncertainty to encourage exploration
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        
        # Process noise and observation noise
        self.Q_base = 0.01       # Base process noise
        self.R = 1.0             # Observation noise variance
        self.q_min = 0.001       # Minimum process noise
        self.q_max = 1.0         # Maximum process noise
        self.adapt_rate = 0.1    # How quickly Q adapts
        
        # Exploration parameter
        self.beta = 1.0
        
        # Deadline guard
        self.start_time = time.time()
        self.deadline = 145.0  # Safety margin before 150s limit

    def _check_deadline(self):
        """Check if we've exceeded our time budget."""
        return (time.time() - self.start_time) > self.deadline

    def select(self, context):
        """
        Select arm using optimistic value estimates (UCB-style).
        score_a = theta_a^T x + beta * sqrt(x^T P_a x)
        """
        if self._check_deadline():
            return 0
            
        x = np.asarray(context, dtype=np.float64)
        
        scores = []
        for a in range(self.n):
            # Predict step: add process noise to covariance
            Q = self.Q_base
            self.P[a] = self.P[a] + Q * np.eye(self.d)
            
            # Expected reward
            mu = self.theta[a] @ x
            
            # Uncertainty
            sigma = np.sqrt(max(0.0, x @ self.P[a] @ x))
            
            # Optimistic score
            score = mu + self.beta * sigma
            scores.append(score)
        
        arm = int(np.argmax(scores))
        return arm

    def update(self, context, arm, reward):
        """
        Update the model for the selected arm with the observed reward.
        """
        if self._check_deadline():
            return
            
        x = np.asarray(context, dtype=np.float64)
        
        # Innovation (prediction error)
        y_pred = self.theta[arm] @ x
        innovation = reward - y_pred
        
        # Innovation covariance
        S = x @ self.P[arm] @ x + self.R
        
        # Kalman gain
        K = (self.P[arm] @ x) / S
        
        # Update theta
        self.theta[arm] = self.theta[arm] + K * innovation
        
        # Update P using Joseph form for numerical stability
        I_Kx = np.eye(self.d) - np.outer(K, x)
        self.P[arm] = I_Kx @ self.P[arm] @ I_Kx.T
        
        # Adapt Q based on prediction error magnitude
        expected_var = S
        error_ratio = (innovation ** 2) / (expected_var + 1e-10)
        # If error is large, increase Q; if small, decrease Q
        target_Q = self.Q_base * min(10.0, max(0.1, error_ratio))
        target_Q = min(self.q_max, max(self.q_min, target_Q))
        self.Q_base = (1 - self.adapt_rate) * self.Q_base + self.adapt_rate * target_Q
# EVOLVE-BLOCK-END
