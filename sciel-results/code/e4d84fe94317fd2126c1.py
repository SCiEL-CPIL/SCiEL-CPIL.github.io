"""Non-stationary linear contextual bandit with Kalman Filter and UCB."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Non-Stationary Linear Contextual Bandit Policy using Kalman Filters.
    
    Strategy:
    - Per-arm Kalman Filter for weight estimation with adaptive process noise.
    - UCB (Upper Confidence Bound) exploration based on prediction uncertainty.
    - Deadline guard to prevent timeouts.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Initialize weight vectors (means) to zero
        self.means = [np.zeros(dim) for _ in range(n_arms)]
        
        # Initialize covariance matrices to large identity (high initial uncertainty)
        self.covs = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        
        # Hyperparameters
        self.process_noise_base = 1e-3
        self.process_noise_scale = 0.1
        self.measurement_noise = 0.1
        self.ucb_beta = 1.5
        
        # Deadline guard
        self.start_time = time.time()
        self.deadline = self.start_time + 140.0  # 10s safety margin

    def _check_deadline(self):
        """Check if we are approaching the time limit."""
        return time.time() > self.deadline

    def select(self, context):
        if self._check_deadline():
            return int(np.random.randint(self.n))
        
        context = np.asarray(context, dtype=np.float64)
        
        # Compute UCB scores for each arm
        scores = np.zeros(self.n)
        for a in range(self.n):
            # Mean prediction
            mean_pred = self.means[a] @ context
            
            # Uncertainty: sqrt(x^T P x)
            # We use the covariance matrix P to estimate variance of the prediction
            # var = x^T P x
            var = context @ self.covs[a] @ context
            std = np.sqrt(var)
            
            # UCB score: mean + beta * std
            scores[a] = mean_pred + self.ucb_beta * std
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        if self._check_deadline():
            return
            
        context = np.asarray(context, dtype=np.float64)
        
        # Kalman Filter Update Step
        # 1. Prediction Error (Innovation)
        pred_mean = self.means[arm] @ context
        innovation = reward - pred_mean
        
        # 2. Innovation Variance
        # S = H P H^T + R, where H = x (linear model), R = measurement noise
        # S = x^T P x + R
        s = context @ self.covs[arm] @ context + self.measurement_noise
        
        # 3. Kalman Gain
        # K = P H^T S^-1
        # K = P x / S  (vector)
        k_gain = (self.covs[arm] @ context) / s
        
        # 4. Update Mean
        # x_new = x_old + K * innovation
        self.means[arm] += k_gain * innovation
        
        # 5. Update Covariance
        # P_new = (I - K H) P
        # P_new = P - K x^T P
        # To maintain symmetry and positive semi-definiteness, we can use the Joseph form
        # or the standard form. Standard form is faster.
        # P_new = (I - K x^T) P
        # Let's compute (I - K x^T) P = P - K (x^T P)
        # Note: x^T P is a row vector. K is a column vector.
        # P_new = P - np.outer(K, x @ P) ? No.
        # K = P x / S
        # P_new = P - (P x x^T P) / S
        # This is more stable and symmetric.
        
        px = self.covs[arm] @ context
        # P_new = P - (px x^T px) / S
        self.covs[arm] -= np.outer(px, px) / s
        
        # 6. Adaptive Process Noise (Time Update / Prediction Step for non-stationarity)
        # Add noise to covariance to model changing weights.
        # Scale process noise by innovation magnitude to react to big changes.
        q = self.process_noise_base + self.process_noise_scale * (innovation ** 2)
        self.covs[arm] += q * np.eye(self.d)
        
        # Ensure symmetry (numerical stability)
        self.covs[arm] = (self.covs[arm] + self.covs[arm].T) / 2.0
# EVOLVE-BLOCK-END
