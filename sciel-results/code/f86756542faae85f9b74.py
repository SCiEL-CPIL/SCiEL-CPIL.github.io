# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Per-arm Kalman filter with Huber-robust innovation handling and adaptive process noise.
    
    Improvements over baseline:
    1. Huber loss weighting on innovations to reduce sensitivity to outliers/regime shifts
    2. Adaptive process noise based on EMA of squared prediction errors
    3. Adaptive UCB coefficient that decays with time but remains responsive
    4. Joseph form covariance update for numerical stability
    5. Deadline guard for safety
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self.start_time = time.time()
        self.time_limit = 140.0  # safety margin before 150s kill

        # Per-arm state
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        self.count = [0 for _ in range(n_arms)]
        self.last_update_time = [0 for _ in range(n_arms)]

        # Adaptive process noise (scalar, scaled to identity)
        self.Q_scalar = [1e-3 for _ in range(n_arms)]
        self.base_Q = 1e-3
        
        # For adaptive Q: track baseline noise and recent error
        self.noise_baseline = [1.0 for _ in range(n_arms)]
        self.recent_error_ema = [0.0 for _ in range(n_arms)]
        self.error_count = [0 for _ in range(n_arms)]

        # Huber threshold parameter
        self.huber_delta = 1.5

        # Adaptive UCB coefficient
        self.beta_base = 2.0
        self.beta_min = 0.3
        self.beta_growth_rate = 0.05 # Growth of beta for unexplored arms

    def _get_adaptive_Q(self, arm):
        """Get adaptive process noise matrix for an arm."""
        # Update Q based on comparison of recent error to baseline
        if self.error_count[arm] >= 10:
            # If recent error is significantly higher than baseline, increase Q
            # ratio > 2.0 suggests a regime shift or high noise
            ratio = self.recent_error_ema[arm] / max(self.noise_baseline[arm], 1e-6)
            if ratio > 2.0:
                # Increase Q to allow faster tracking
                # Cap the increase to prevent instability
                target_Q = self.base_Q * min(ratio, 10.0)
                # Smooth transition to new Q
                self.Q_scalar[arm] = 0.9 * self.Q_scalar[arm] + 0.1 * target_Q
            else:
                # Decay Q back to baseline if error is normal
                self.Q_scalar[arm] = 0.95 * self.Q_scalar[arm] + 0.05 * self.base_Q
        return self.Q_scalar[arm] * np.eye(self.d)

    def _huber_weight(self, innovation, S):
        """Compute Huber loss weight for robust update.
        
        Returns a weight in (0, 1] that downweights large innovations.
        """
        if S <= 0:
            return 1.0
        abs_innov = abs(innovation)
        threshold = self.huber_delta * np.sqrt(S)
        if abs_innov <= threshold:
            return 1.0
        else:
            # Linear decay: weight = threshold / |innovation|
            return threshold / abs_innov

    def _adaptive_beta(self, arm):
        """Adaptive UCB coefficient that increases for unexplored arms."""
        # Time since last update for this arm
        time_since_update = self.t - self.last_update_time[arm]
        # Beta increases linearly with time since update, capped
        # This encourages exploration of arms that haven't been seen recently
        beta = self.beta_min + self.beta_growth_rate * time_since_update
        # Also decay global beta slightly to reduce overall exploration as t grows
        global_decay = np.exp(-0.0005 * self.t)
        return max(beta * global_decay, self.beta_min)

    def select(self, context):
        # Deadline guard
        if time.time() - self.start_time > self.time_limit:
            scores = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(scores))

        context = np.asarray(context, dtype=np.float64)
        scores = []

        for a in range(self.n):
            # Predict step: inflate covariance with process noise
            Q_a = self._get_adaptive_Q(a)
            P_pred = self.P[a] + Q_a

            # Prediction
            pred_reward = self.theta[a] @ context

            # UCB score: mean + beta * uncertainty
            uncertainty = np.sqrt(max(context @ P_pred @ context, 0.0))
            beta = self._adaptive_beta(a)
            score = pred_reward + beta * uncertainty

            scores.append(score)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Deadline guard
        if time.time() - self.start_time > self.time_limit:
            return

        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        self.t += 1
        self.last_update_time[arm] = self.t

        theta = self.theta[arm]
        P = self.P[arm]

        # Predict step
        Q_a = self._get_adaptive_Q(arm)
        P_pred = P + Q_a

        # Prediction
        pred_reward = theta @ context

        # Innovation
        innovation = reward - pred_reward

        # Update error tracking for adaptive Q
        sq_err = innovation * innovation
        if self.error_count[arm] == 0:
            self.recent_error_ema[arm] = sq_err
            self.noise_baseline[arm] = sq_err
        else:
            # Fast EMA for recent error
            alpha_fast = 0.2
            self.recent_error_ema[arm] = (1 - alpha_fast) * self.recent_error_ema[arm] + alpha_fast * sq_err
            # Slow EMA for baseline noise
            alpha_slow = 0.01
            self.noise_baseline[arm] = (1 - alpha_slow) * self.noise_baseline[arm] + alpha_slow * sq_err
        self.error_count[arm] += 1

        # Innovation variance
        S = context @ P_pred @ context + 1.0

        # Huber robustness weight
        w = self._huber_weight(innovation, S)

        # Effective innovation and noise
        # With Huber loss, we scale the innovation and adjust the noise
        S_eff = S / w
        K = P_pred @ context / S_eff

        # Update theta with weighted innovation
        self.theta[arm] = theta + K * (innovation * w)

        # Joseph form for numerical stability
        # Effective observation noise reflects Huber adjustment
        R_eff = 1.0 / w
        I_K_ctx = np.eye(self.d) - np.outer(K, context)
        self.P[arm] = I_K_ctx @ P_pred @ I_K_ctx.T + np.outer(K, K) * R_eff

        # Clamp covariance to prevent numerical drift
        self.P[arm] = np.maximum(self.P[arm], 1e-8 * np.eye(self.d))

        self.count[arm] += 1
# EVOLVE-BLOCK-END
