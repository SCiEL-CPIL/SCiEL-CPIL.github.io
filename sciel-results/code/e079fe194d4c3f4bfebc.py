"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit using a per-arm Kalman filter.
    
    Structural Design:
    - Uses a Hard Reset mechanism for non-stationarity detection (from inspirations),
      which is more aggressive and effective than soft inflation for large shifts.
    - Starts with high initial covariance (10.0) to ensure thorough initial exploration.
    
    Refinement Machinery:
    - Retains Adaptive Observation Noise (R) estimation from the current program,
      allowing the filter to dynamically adjust its gain based on recent error history.
    - Uses Count-scaled UCB to balance exploration across arms with different visit counts.
    - Joseph-form covariance update for numerical stability.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self._start_time = time.time()
        self._deadline = 115.0  # Safety margin before 150s kill

        # Per-arm state: mean (theta) and covariance (P)
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        # Moderate initial covariance for balanced exploration
        self.P = [np.eye(dim) * 5.0 for _ in range(n_arms)]
        self.count = [0 for _ in range(n_arms)]

        # Adaptive process noise parameters
        self._base_Q = 5e-4
        self._Q_max = 0.03
        # EMA of squared innovations per arm
        self._innov_ema = [0.0 for _ in range(n_arms)]
        self._innov_alpha = 0.10  # Smoothing factor

        # Adaptive observation noise: estimate R from innovations
        self._R_ema = [0.05 for _ in range(n_arms)]
        self._R_alpha = 0.08
        self._R_min = 0.01
        self._R_max = 0.5
        self._R_initial = 0.05

        # UCB parameter (adaptive)
        self.alpha = 1.0
        self._alpha_min = 0.4
        # Decay rate for exploration
        self._alpha_decay_rate = 0.08 

        # Reset mechanism: sliding window-based reset
        self._innov_window = [[] for _ in range(n_arms)]
        self._window_size = 10
        self._reset_threshold = 4  # Trigger reset if 4+ large innovations in window
        self._reset_cov_scale = 3.0  # Reset covariance to this scale
        self._large_innov_threshold = 3.0

    def _adaptive_Q(self, arm):
        """Compute adaptive process noise based on EMA of squared innovations."""
        if self.count[arm] < 3:
            return self._base_Q
        ema_val = self._innov_ema[arm]
        # Use tanh saturation for smooth, bounded Q
        scaled = np.tanh(ema_val / 0.3)
        q = self._base_Q + (self._Q_max - self._base_Q) * scaled
        return q

    def _adaptive_R(self, arm):
        """Estimate observation noise from innovation statistics with robustness."""
        if self.count[arm] < 5:
            return self._R_initial
        # Use the EMA of raw squared prediction errors, clipped for robustness
        r_est = max(self._R_min, min(self._R_max, self._R_ema[arm]))
        return r_est

    def _maybe_reset(self, arm, innov_sq):
        """If too many large innovations in sliding window, hard reset covariance."""
        window = self._innov_window[arm]
        window.append(innov_sq)
        if len(window) > self._window_size:
            window.pop(0)
        
        if len(window) >= self._window_size:
            large_count = sum(1 for v in window if v > self._large_innov_threshold)
            if large_count >= self._reset_threshold:
                # Hard reset: replace covariance to force re-exploration
                self.P[arm] = np.eye(self.d) * self._reset_cov_scale
                self._innov_window[arm] = []

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
            
            # Count-scaled UCB bonus with softer decay
            count_factor = 1.0 / (1.0 + 0.5 * np.sqrt(self.count[a]))
            # Time-decaying exploration
            alpha_t = max(self._alpha_min, self.alpha * (1.0 + self.t) ** (-self._alpha_decay_rate))
            
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

        # Use adaptive R for this arm
        R_eff = self._adaptive_R(a)

        P_pred_x = P_pred @ x
        S = x @ P_pred_x + R_eff
        if S < 1e-12:
            S = 1e-12

        K = P_pred_x / S

        self.theta[a] = self.theta[a] + K * y

        # Joseph form update for numerical stability
        IK = np.eye(self.d) - np.outer(K, x)
        self.P[a] = IK @ P_pred @ IK.T + np.outer(K, K) * R_eff
        self.P[a] = (self.P[a] + self.P[a].T) / 2.0

        # Periodic eigenvalue clipping for numerical stability
        if self.t % 50 == 0:
            eigvals = np.linalg.eigvalsh(self.P[a])
            min_eig = np.min(eigvals)
            if min_eig < 1e-8:
                self.P[a] = self.P[a] + (1e-8 - min_eig) * np.eye(self.d)

        # Track prediction error for adaptive Q
        innov_sq = y * y / S
        self._innov_ema[a] = (1.0 - self._innov_alpha) * self._innov_ema[a] + self._innov_alpha * innov_sq

        # Update adaptive R estimate from raw prediction error with clipping
        raw_err_sq = min(y * y, 1.0)  # Clip extreme errors
        self._R_ema[a] = (1.0 - self._R_alpha) * self._R_ema[a] + self._R_alpha * raw_err_sq

        # Check for reset
        self._maybe_reset(a, innov_sq)

        self.count[a] += 1
# EVOLVE-BLOCK-END
