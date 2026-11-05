# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Per-Arm Adaptive Kalman Filter with Context-Scaled Process Noise.

    Combines the robust Kalman filter structure with a Joseph Form covariance update
    for numerical stability. The process noise (Q) is dynamically modulated by:
    1. A base level of drift.
    2. An adaptive component based on recent innovation variance (EMA).
    3. A context-dependent component scaled by the norm of the current context vector.
    
    This allows the filter to automatically increase uncertainty (and thus exploration
    and adaptation speed) when the context magnitude is large or when recent prediction
    errors are high, while maintaining stability in low-noise, low-magnitude regimes.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self._start_time = time.time()
        self._deadline = 112.0

        # Per-arm state: theta (weight vector), P (covariance)
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        self.count = [0 for _ in range(n_arms)]

        # Observation noise
        self._R = 0.1

        # Exploration parameters
        self.alpha = 1.2
        self._alpha_min = 0.3

        # Adaptive Process Noise (Q) parameters
        # Q_t = Q_base + Q_adaptive + lambda_ctx * ||x||^2
        self._Q_base = 3e-4
        self._Q_max_adapt = 0.04
        self._lambda_ctx = 1e-3  # Scaling factor for context norm squared
        
        # EMA tracking for adaptive Q
        self._innov_ema = [0.0 for _ in range(n_arms)]
        self._innov_alpha = 0.12

        # Reset mechanism
        self._consec_large = [0 for _ in range(n_arms)]
        self._reset_threshold = 3
        self._reset_cov_scale = 5.0

    def _adaptive_Q(self, arm, context):
        """Compute adaptive process noise.
        
        Combines:
        1. Base drift.
        2. Adaptive term based on innovation history (tanh-saturated).
        3. Context-dependent term proportional to ||x||^2.
        """
        if self.count[arm] < 3:
            q_adapt = 0.0
        else:
            ema_val = self._innov_ema[arm]
            scaled = np.tanh(ema_val / 0.4)
            q_adapt = self._Q_max_adapt * scaled

        # Context scaling: larger contexts might imply higher variance in underlying signal
        ctx_norm_sq = np.dot(context, context)
        q_ctx = self._lambda_ctx * ctx_norm_sq

        return self._Q_base + q_adapt + q_ctx

    def _maybe_reset(self, arm, innov_sq):
        """If too many consecutive large innovations, reset covariance."""
        if innov_sq > 4.0:
            self._consec_large[arm] += 1
        else:
            self._consec_large[arm] = max(0, self._consec_large[arm] - 1)

        if self._consec_large[arm] >= self._reset_threshold:
            self.P[arm] = np.eye(self.d) * self._reset_cov_scale
            self._consec_large[arm] = 0

    def select(self, context):
        """Select arm using UCB with Kalman Filter estimates."""
        if time.time() - self._start_time > self._deadline:
            scores = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(scores))

        ctx = np.asarray(context, dtype=np.float64)
        scores = []
        for a in range(self.n):
            mu = self.theta[a] @ ctx
            Px = self.P[a] @ ctx
            var = ctx @ Px
            if var < 0:
                var = 0.0
            std = np.sqrt(var)

            # Adaptive exploration: count-scaled and time-decayed
            count_factor = 1.0 / (1.0 + np.sqrt(self.count[a]))
            alpha_t = max(self._alpha_min, self.alpha * (1.0 + self.t) ** (-0.1))

            score = mu + alpha_t * std * count_factor
            scores.append(score)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.t += 1
        a = arm
        x = np.asarray(context, dtype=np.float64)
        r = reward

        if time.time() - self._start_time > self._deadline:
            self.count[a] += 1
            return

        # 1. Predict Step: Update Covariance with Process Noise
        Q = self._adaptive_Q(a, x)
        P_pred = self.P[a] + Q * np.eye(self.d)
        
        # 2. Innovation Calculation
        y = r - self.theta[a] @ x

        # 3. Update Step (Kalman)
        # K = P_pred x (x^T P_pred x + R)^-1
        Px = P_pred @ x
        denom = x @ Px + self._R
        if denom < 1e-12:
            denom = 1e-12
        
        K = Px / denom
        
        # Update Theta
        self.theta[a] = self.theta[a] + K * y
        
        # Update P using Joseph Form for numerical stability
        # P_new = (I - K x^T) P_pred (I - K x^T)^T + K R K^T
        I = np.eye(self.d)
        IKxT = I - np.outer(K, x)
        P_new = IKxT @ P_pred @ IKxT.T + np.outer(K, K) * self._R
        
        # Ensure symmetry
        self.P[a] = (P_new + P_new.T) / 2.0
        
        # Track innovation for adaptive Q
        # Use normalized innovation squared as a measure of "surprise"
        innov_sq = (y * y) / denom
        self._innov_ema[a] = (1.0 - self._innov_alpha) * self._innov_ema[a] + self._innov_alpha * innov_sq
        
        # Check for reset
        self._maybe_reset(a, innov_sq)
        
        self.count[a] += 1
# EVOLVE-BLOCK-END
