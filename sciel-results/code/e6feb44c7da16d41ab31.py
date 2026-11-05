"""Non-stationary linear contextual bandit with adaptive Kalman Filter."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit using a per-arm Kalman Filter.
    
    Models the weight vector theta for each arm as a random walk:
    theta_t = theta_{t-1} + w_t, where w_t ~ N(0, Q_t)
    
    The process noise Q_t is adapted dynamically based on recent prediction errors
    (innovations) to better track non-stationarity.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Initialize state for each arm: theta (mean) and P (covariance)
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        # Start with high uncertainty
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        
        # Process noise base and adaptive scaling
        self.Q_base = 0.05 * np.eye(dim)
        self.R = 1.0  # Observation noise variance
        
        # Adaptive parameters
        self.innovation_history = [[] for _ in range(n_arms)]
        self.hist_len = 20
        self.q_min = 0.01
        self.q_max = 1.0
        self.q_scale_factor = 0.5
        
        # Deadline management
        self.start_time = time.time()
        self.deadline = 110.0  # Safety margin for 120s limit

    def _check_deadline(self):
        """Check if we are close to the time limit."""
        return (time.time() - self.start_time) > self.deadline

    def _adapt_process_noise(self, arm, innovation):
        """
        Adapt the process noise Q for the given arm based on the innovation (prediction error).
        Larger recent errors imply faster drift, so we increase Q.
        """
        hist = self.innovation_history[arm]
        hist.append(innovation)
        if len(hist) > self.hist_len:
            hist.pop(0)
        
        if len(hist) < 5:
            # Not enough data, use base noise
            return self.Q_base.copy()
        
        # Calculate a robust measure of recent error magnitude
        # Using median absolute deviation or simple mean of squared errors
        recent_errors = np.array(hist[-5:])
        mean_sq_error = np.mean(recent_errors ** 2)
        
        # Scale Q based on the error magnitude
        # If error is large, Q should be large to allow theta to change
        # We map the error to a scaling factor between q_min and q_max
        # A simple linear mapping with saturation
        scale = np.clip(mean_sq_error, self.q_min, self.q_max)
        
        return self.Q_base * (scale / 0.1)  # 0.1 is a reference error level

    def select(self, context):
        """
        Select an arm using UCB based on the Kalman Filter posterior.
        """
        context = np.asarray(context)
        
        # If near deadline, exploit the best estimate to save computation
        if self._check_deadline():
            scores = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            # Mean reward prediction
            mu = self.theta[a] @ context
            
            # Variance of prediction
            # var(mu) = x^T P x
            var = context @ self.P[a] @ context
            
            # UCB bonus: alpha * std
            # We use a standard alpha, but can adapt if needed. 
            # For non-stationary, a slightly lower alpha might be better to reduce exploration overhead,
            # but UCB is generally robust. Let's stick with 1.0.
            bonus = 1.0 * np.sqrt(max(0.0, var))
            
            scores.append(mu + bonus)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        """
        Update the Kalman Filter state for the selected arm.
        """
        context = np.asarray(context)
        
        # 1. Predict step: Expand covariance due to process noise
        # First, adapt Q based on the *previous* innovation? 
        # Usually, we predict, then update. 
        # The innovation is computed during the update step.
        # To adapt Q for the *current* step, we can use the history from previous steps.
        # Or, we can compute the innovation, then update P.
        
        # Standard Kalman Filter Update:
        
        # 1. Predict
        # theta_pred = theta (random walk, no change in mean)
        # P_pred = P + Q
        
        # We need Q. Let's use the adapted Q from the last innovation.
        Q = self._adapt_process_noise(arm, 0.0) # Pass 0.0 as placeholder, we use history
        
        self.P[arm] = self.P[arm] + Q
        
        # 2. Update
        # Innovation (prediction error)
        # y = reward - theta^T x
        y = reward - self.theta[arm] @ context
        
        # Innovation covariance
        S = context @ self.P[arm] @ context + self.R
        
        # Kalman Gain
        # K = P x^T / S
        K = (self.P[arm] @ context) / S
        
        # Update mean
        self.theta[arm] = self.theta[arm] + K * y
        
        # Update covariance
        # P = (I - K x^T) P
        # Using Joseph form for numerical stability:
        # P = (I - K x^T) P (I - K x^T)^T + K R K^T
        # Since K = P x^T / S, we can simplify, but Joseph form is safer.
        I_KxT = np.eye(self.d) - np.outer(K, context)
        self.P[arm] = I_KxT @ self.P[arm] @ I_KxT.T + np.outer(K, K) * self.R
        
        # Ensure symmetry
        self.P[arm] = (self.P[arm] + self.P[arm].T) / 2.0
        
        # Record the actual innovation for future Q adaptation
        self.innovation_history[arm].append(y)
        if len(self.innovation_history[arm]) > self.hist_len:
            self.innovation_history[arm].pop(0)

# EVOLVE-BLOCK-END
