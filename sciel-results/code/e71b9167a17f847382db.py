import numpy as np
import time

class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        
        # State variables
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        # Start with moderate covariance to balance exploration
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        self.count = [0 for _ in range(n_arms)]
        
        # Adaptive Q parameters - faster adaptation with dynamic alpha
        self._innov_ema = [0.0 for _ in range(n_arms)]
        self._innov_ema_sq = [0.0 for _ in range(n_arms)]  # Track variance of innovations
        self._base_Q = 1e-4
        self._Q_max = 0.08
        
        # Exploration parameter
        self.alpha = 1.0
        self._alpha_min = 0.2
        
        # Deadline guard
        self._start_time = time.time()
        self._deadline = 112.0 # Seconds
        
        # Smooth covariance inflation instead of hard reset
        self._inflation_factor = [1.0 for _ in range(n_arms)]
        
        # Observation noise estimation
        self._sigma2_est = [0.1 for _ in range(n_arms)]
        self._sigma2_ema = [0.1 for _ in range(n_arms)]

    def _adaptive_Q(self, arm):
        """Compute adaptive process noise with dynamic adaptation rate."""
        if self.count[arm] < 5:
            return self._base_Q
        
        ema_val = self._innov_ema[arm]
        ema_sq = self._innov_ema_sq[arm]
        
        # Innovation-to-expected ratio: high means non-stationarity
        expected_innov = np.sqrt(ema_sq + 1e-8)
        ratio = abs(ema_val) / (expected_innov + 1e-8)
        
        # Dynamic alpha: faster adaptation when non-stationarity detected
        base_alpha = 0.08
        adaptive_alpha = base_alpha * (1.0 + 2.0 * min(ratio, 3.0))
        
        # Update EMAs with dynamic alpha
        self._innov_ema[arm] = (1.0 - adaptive_alpha) * self._innov_ema[arm] + adaptive_alpha * ema_val
        self._innov_ema_sq[arm] = (1.0 - adaptive_alpha) * self._innov_ema_sq[arm] + adaptive_alpha * ema_val * ema_val
        
        # Q scales with recent innovation magnitude, smoothed
        scaled = np.tanh(ema_val / 0.3)
        q = self._base_Q + (self._Q_max - self._base_Q) * scaled
        return q

    def _smooth_inflate(self, arm, innov_sq):
        """Smoothly inflate covariance based on recent innovation magnitude."""
        # Normalize innovation: how surprising is this observation?
        expected_var = self._sigma2_est[arm] + 1e-8
        z_score = np.sqrt(innov_sq / expected_var)
        
        # Inflation factor increases with surprise, capped
        target_inflation = 1.0 + 0.5 * np.tanh(z_score - 1.5)
        
        # Smooth transition
        self._inflation_factor[arm] = 0.85 * self._inflation_factor[arm] + 0.15 * target_inflation
        
        # Apply inflation to P
        if self._inflation_factor[arm] > 1.01:
            self.P[arm] = self.P[arm] * self._inflation_factor[arm]
            # Decay inflation back toward 1
            self._inflation_factor[arm] *= 0.95

    def select(self, context):
        # Deadline check
        if time.time() - self._start_time > self._deadline:
            # Fallback to greedy
            scores = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(scores))
        
        context = np.asarray(context, dtype=np.float64)
        scores = []
        for a in range(self.n):
            # Prediction
            pred = self.theta[a] @ context
            
            # Uncertainty
            Px = self.P[a] @ context
            var = context @ Px
            if var < 0:
                var = 0.0
            std = np.sqrt(var)
            
            # Adaptive exploration: UCB with count-aware bonus
            count_factor = 1.0 / (1.0 + np.sqrt(self.count[a]))
            alpha_t = max(self._alpha_min, self.alpha * (1.0 + self.t) ** (-0.08))
            
            # Bonus scaled by uncertainty relative to prediction magnitude
            pred_mag = max(abs(pred), 0.5)
            bonus_scale = min(std / pred_mag, 2.0)
            
            score = pred + alpha_t * std * count_factor * bonus_scale
            scores.append(score)
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.t += 1
        a = arm
        x = np.asarray(context, dtype=np.float64)
        r = reward
        
        # Deadline guard
        if time.time() - self._start_time > self._deadline:
            self.count[a] += 1
            return
        
        # 1. Predict (Process Noise)
        Q = self._adaptive_Q(a)
        P_pred = self.P[a] + Q * np.eye(self.d)
        
        # 2. Calculate Innovation
        y = r - self.theta[a] @ x
        
        # 3. Estimate observation noise from recent innovations
        sigma2 = self._sigma2_est[a]
        
        # 4. Update Step (Kalman)
        Px = P_pred @ x
        denom = x @ Px + sigma2
        if denom < 1e-12:
            denom = 1e-12
        K = Px / denom
        
        # Update Theta
        self.theta[a] = self.theta[a] + K * y
        
        # Update P (Joseph Form)
        I = np.eye(self.d)
        IKxT = I - np.outer(K, x)
        P_new = IKxT @ P_pred @ IKxT.T + np.outer(K, K) * sigma2
        
        # Ensure symmetry
        self.P[a] = (P_new + P_new.T) / 2.0
        
        # 5. Track prediction error for adaptive Q and sigma2 estimation
        innov_sq = (y * y) / denom
        
        # Update sigma2 estimate: innovations should be ~chi2(1) if model is correct
        # Expected value of innov_sq is ~1, so sigma2_est tracks the scale
        sigma2_alpha = 0.05
        self._sigma2_ema[a] = (1.0 - sigma2_alpha) * self._sigma2_ema[a] + sigma2_alpha * innov_sq
        # Clamp to reasonable range
        self._sigma2_est[a] = np.clip(self._sigma2_ema[a], 0.01, 1.0)
        
        # Smooth inflation based on innovation
        self._smooth_inflate(a, innov_sq)
        
        self.count[a] += 1
