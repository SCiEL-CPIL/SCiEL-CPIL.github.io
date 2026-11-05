import numpy as np
import time

class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        
        # State variables
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        # Start with large covariance to encourage exploration
        self.P = [np.eye(dim) * 5.0 for _ in range(n_arms)]
        
        # Adaptive Q parameters
        self.error_ema = [0.0 for _ in range(n_arms)]
        self.ema_decay = 0.9
        self.q_base = 0.05
        self.q_max = 0.2
        self.q_scale = 5.0
        
        # Exploration parameter
        self.beta_init = 2.0
        self.beta_min = 0.5
        self.beta_decay = 0.9995
        
        # Deadline guard
        self.start_time = time.time()
        self.deadline = 145.0 # Seconds

    def _get_adaptive_q(self, arm_idx):
        # Calculate Q based on current error EMA
        # tanh maps (-inf, inf) to (-1, 1), we want positive Q
        # q_val is between q_base and q_max
        q_val = self.q_base + (self.q_max - self.q_base) * np.tanh(self.q_scale * self.error_ema[arm_idx])
        return np.eye(self.d) * q_val

    def select(self, context):
        # Deadline check
        if time.time() - self.start_time > self.deadline:
            # Fallback: just return the arm with highest current theta prediction
            context = np.asarray(context)
            preds = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(preds))

        context = np.asarray(context)
        scores = []
        # Dynamic beta: starts high, decays over time
        # Using a simple exponential decay or time-based
        # Let's use time-based for smoother transition
        # beta = beta_min + (beta_init - beta_min) * exp(-t/100)
        # Or just keep it constant if adaptive Q is doing the work.
        # The prompt suggests "decaying exploration coefficient".
        self.beta = self.beta_min + (self.beta_init - self.beta_min) * (self.beta_decay ** self.t)
        
        for a in range(self.n):
            # Prediction
            pred = self.theta[a] @ context
            
            # Uncertainty
            # P is symmetric positive definite
            var = context @ self.P[a] @ context
            std = np.sqrt(max(var, 0))
            
            # UCB Score
            score = pred + self.beta * std
            scores.append(score)
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.t += 1
        context = np.asarray(context)
        
        # 1. Predict (Process Noise)
        # Update Q based on previous error? 
        # Usually Q is applied to P before update.
        # We need the error to update Q. 
        # Strategy: 
        # 1. Calculate prediction error using CURRENT theta (before update).
        # 2. Update Error EMA.
        # 3. Calculate Q.
        # 4. Update P with Q (Prediction step).
        # 5. Update theta and P with observation (Update step).
        
        pred = self.theta[arm] @ context
        error = reward - pred
        
        # Update Error EMA
        # Use a faster decay for more responsiveness?
        # self.error_ema[arm] = self.ema_decay * self.error_ema[arm] + (1 - self.ema_decay) * (error ** 2)
        # Let's try a slightly faster decay to adapt quicker to non-stationarity
        self.error_ema[arm] = self.ema_decay * self.error_ema[arm] + (1 - self.ema_decay) * (error ** 2)
        
        # Calculate Adaptive Q
        Q = self._get_adaptive_q(arm)
        
        # Prediction Step: P_pred = P + Q
        P_pred = self.P[arm] + Q
        
        # Update Step (Kalman)
        # K = P_pred x (x^T P_pred x + sigma^2)^-1
        # Assume sigma^2 = 1.0
        sigma2 = 1.0
        Px = P_pred @ context
        denom = context @ Px + sigma2
        K = Px / denom
        
        # Update Theta
        self.theta[arm] = self.theta[arm] + K * error
        
        # Update P (Joseph Form)
        # P_new = (I - K x^T) P_pred (I - K x^T)^T + K sigma2 K^T
        # To save computation, use the simpler form if stable:
        # P_new = P_pred - K x^T P_pred
        # Joseph form is safer:
        I = np.eye(self.d)
        KxT = np.outer(K, context)
        IKxT = I - KxT
        P_new = IKxT @ P_pred @ IKxT.T + np.outer(K, K) * sigma2
        
        # Ensure symmetry
        self.P[arm] = (P_new + P_new.T) / 2.0
