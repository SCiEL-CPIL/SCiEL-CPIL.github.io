"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit using a per-arm Kalman filter
    with adaptive process noise, UCB exploration, and occasional resets.
    
    Key improvements over baseline:
    - Adaptive process noise via innovation EMA with tanh saturation.
    - Joseph-form covariance update for numerical stability.
    - Count-scaled UCB bonus that decays as we learn more about each arm.
    - Global alpha decay to reduce exploration over time.
    - Periodic soft-reset of covariance when prediction errors spike,
      allowing faster recovery from large weight shifts.
    - Tighter deadline guard for safety.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self._start_time = time.time()
        self._deadline = 112.0  # safety margin before 120s kill

        # Per-arm state: mean (theta) and covariance (P)
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        # Initialize covariance large to encourage exploration early
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        self.count = [0 for _ in range(n_arms)]

        # Adaptive process noise parameters
        self._base_Q = 3e-4
        self._Q_max = 0.04
        # EMA of squared innovations per arm
        self._innov_ema = [0.0 for _ in range(n_arms)]
        self._innov_alpha = 0.12  # EMA smoothing factor

        # Observation noise
        self._R = 0.1

        # UCB parameter (adaptive)
        self.alpha = 1.2
        self._alpha_min = 0.3
        self._alpha_decay_rate = 500.0

        # Reset mechanism: track consecutive large innovations
        self._consec_large = [0 for _ in range(n_arms)]
        self._reset_threshold = 3.0  # number of consecutive large innovations
        self._reset_cov_scale = 5.0  # scale to reset covariance to

    def _adaptive_Q(self, arm):
        """Compute adaptive process noise based on EMA of squared innovations."""
        if self.count[arm] < 3:
            return self._base_Q
        ema_val = self._innov_ema[arm]
        scaled = np.tanh(ema_val / 0.4)
        q = self._base_Q + (self._Q_max - self._base_Q) * scaled
        return q

    def _maybe_reset(self, arm, innov_sq):
        """If too many consecutive large innovations, reset covariance."""
        if innov_sq > 4.0:
            self._consec_large[arm] += 1
        else:
            self._consec_large[arm] = max(0, self._consec_large[arm] - 1)
        
        if self._consec_large[arm] >= self._reset_threshold:
            # Soft reset: inflate covariance to encourage re-exploration
            self.P[arm] = np.eye(self.d) * self._reset_cov_scale
            self._consec_large[arm] = 0

    def select(self, context):
        """Select arm using UCB with Kalman filter estimates."""
        # Deadline guard
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
            
            # Adaptive exploration: reduce bonus as we learn more
            count_factor = 1.0 / (1.0 + np.sqrt(self.count[a]))
            alpha_t = max(self._alpha_min, self.alpha * (1.0 + self.t) ** (-0.1))
            
            score = mu + alpha_t * std * count_factor
            scores.append(score)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        """Update Kalman filter for the selected arm with observed reward."""
        self.t += 1
        a = arm
        x = np.asarray(context, dtype=np.float64)
        r = reward

        # Deadline guard
        if time.time() - self._start_time > self._deadline:
            self.count[a] += 1
            return

        # Prediction step (random walk model)
        Q = self._adaptive_Q(a)
        P_pred = self.P[a] + Q * np.eye(self.d)

        # Update step
        y = r - self.theta[a] @ x

        P_pred_x = P_pred @ x
        S = x @ P_pred_x + self._R
        if S < 1e-12:
            S = 1e-12

        K = P_pred_x / S

        self.theta[a] = self.theta[a] + K * y

        IK = np.eye(self.d) - np.outer(K, x)
        self.P[a] = IK @ P_pred @ IK.T + Q * np.eye(self.d)
        self.P[a] = (self.P[a] + self.P[a].T) / 2.0

        # Track prediction error for adaptive Q
        innov_sq = y * y / S
        self._innov_ema[a] = (1.0 - self._innov_alpha) * self._innov_ema[a] + self._innov_alpha * innov_sq

        # Check for reset
        self._maybe_reset(a, innov_sq)

        self.count[a] += 1
# EVOLVE-BLOCK-END
