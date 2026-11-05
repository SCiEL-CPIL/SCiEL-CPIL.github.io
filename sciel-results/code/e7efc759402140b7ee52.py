# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit with:
    - Per-arm Kalman filter with adaptive process noise
    - Prediction-error-driven acceleration of forgetting
    - UCB exploration with adaptive coefficient
    - Joseph-form updates for numerical stability
    - Deadline guard and periodic numerical stabilization
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0  # Safety margin before 120s limit

        # Dual-mode Kalman filter state per arm
        # Stable mode: low process noise, high confidence
        self.theta_stable = [np.zeros(dim) for _ in range(n_arms)]
        self.P_stable = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        
        # Volatile mode: high process noise, fast adaptation
        self.theta_volatile = [np.zeros(dim) for _ in range(n_arms)]
        self.P_volatile = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        
        # Regime probabilities: P(volatile | history) per arm
        self.regime_prob = [0.1] * n_arms  # Start mostly in stable mode
        self.regime_alpha = 0.15  # Slow adaptation of regime probability
        
        self.count = [0] * n_arms

        # Process noise parameters for each mode
        self.q_stable_base = 0.02
        self.q_volatile_base = 0.5
        self.q_max = 5.0
        
        # Innovation tracking
        self.innov_ema_stable = [0.0] * n_arms
        self.innov_ema_volatile = [0.0] * n_arms
        self.innov_ema_alpha = 0.3
        self.innov_base = 1.0

        # UCB exploration parameters
        self.u_scale = 1.5
        self.count_scale = 0.4
        self.warmup_steps = 20
        self.jitter = 1e-6
        self.step_count = 0

    def _check_deadline(self):
        return (time.time() - self.start_time) > self.deadline

    def _update_regime(self, arm, innov_stable, innov_volatile):
        """
        Update regime probability based on which mode explains the innovation better.
        Uses a simple likelihood ratio approach.
        """
        # Compute innovation magnitudes relative to expected noise
        # Stable mode expects smaller innovations
        innov_mag_stable = abs(innov_stable)
        innov_mag_volatile = abs(innov_volatile)
        
        # Likelihood ratio: P(innov | stable) / P(innov | volatile)
        # Assume Gaussian noise with std proportional to sqrt(q)
        # For simplicity, use inverse squared innovation as proxy for likelihood
        # Lower innovation magnitude in a mode suggests that mode is more likely
        
        # Normalize innovations by their respective expected scales
        scale_stable = np.sqrt(self.q_stable_base + 1.0)
        scale_volatile = np.sqrt(self.q_volatile_base + 1.0)
        
        z_stable = innov_mag_stable / scale_stable
        z_volatile = innov_mag_volatile / scale_volatile
        
        # If stable mode has much lower z-score, it's more likely
        # Use a smooth switching function
        diff = z_volatile - z_stable  # Positive means stable is better
        # Sigmoid-like mapping: p_volatile decreases as diff increases
        p_volatile_new = 1.0 / (1.0 + np.exp(0.5 * diff))
        
        # Blend with previous probability
        self.regime_prob[arm] = (1.0 - self.regime_alpha) * self.regime_prob[arm] + \
                                  self.regime_alpha * p_volatile_new
        
        # Clip to avoid extreme values
        self.regime_prob[arm] = np.clip(self.regime_prob[arm], 0.01, 0.99)

    def _predict_stable(self, a):
        """Kalman predict step for stable mode."""
        q = self.q_stable_base
        self.P_stable[a] = self.P_stable[a] + q * np.eye(self.d) + self.jitter * np.eye(self.d)
        
    def _predict_volatile(self, a):
        """Kalman predict step for volatile mode."""
        q = self.q_volatile_base
        self.P_volatile[a] = self.P_volatile[a] + q * np.eye(self.d) + self.jitter * np.eye(self.d)

    def _update_stable(self, a, ctx, reward):
        """Kalman update step for stable mode."""
        P = self.P_stable[a]
        theta = self.theta_stable[a]
        pred = float(theta @ ctx)
        innov = reward - pred
        var = float(ctx @ P @ ctx) + 1.0 + self.jitter  # Assuming R=1
        K = P @ ctx / var          # Kalman gain
        self.theta_stable[a] = theta + (K * innov)
        # Joseph-form rank-1 update
        IK = np.eye(self.d) - np.outer(K, ctx)
        self.P_stable[a] = IK @ P @ IK.T
        # Symmetrise and clip for numerical safety
        self.P_stable[a] = 0.5 * (self.P_stable[a] + self.P_stable[a].T)
        np.fill_diagonal(self.P_stable[a], np.maximum(np.diag(self.P_stable[a]), 0.0))
        return innov
        
    def _update_volatile(self, a, ctx, reward):
        """Kalman update step for volatile mode."""
        P = self.P_volatile[a]
        theta = self.theta_volatile[a]
        pred = float(theta @ ctx)
        innov = reward - pred
        var = float(ctx @ P @ ctx) + 1.0 + self.jitter  # Assuming R=1
        K = P @ ctx / var          # Kalman gain
        self.theta_volatile[a] = theta + (K * innov)
        # Joseph-form rank-1 update
        IK = np.eye(self.d) - np.outer(K, ctx)
        self.P_volatile[a] = IK @ P @ IK.T
        # Symmetrise and clip for numerical safety
        self.P_volatile[a] = 0.5 * (self.P_volatile[a] + self.P_volatile[a].T)
        np.fill_diagonal(self.P_volatile[a], np.maximum(np.diag(self.P_volatile[a]), 0.0))
        return innov

    def _adaptive_u(self):
        """Adapt exploration coefficient."""
        if self.step_count < self.warmup_steps:
            return self.u_scale * 1.5
        else:
            # Decay alpha slightly as we get more data, but keep some exploration
            # for non-stationarity
            factor = 1.0 / (1.0 + self.step_count / 1000.0)
            return self.u_scale * max(factor, 0.5)
            
    def _get_blended_state(self, a):
        """Get blended theta and P from stable and volatile modes."""
        p_vol = self.regime_prob[a]
        p_stab = 1.0 - p_vol
        
        # Blend theta
        theta = p_stab * self.theta_stable[a] + p_vol * self.theta_volatile[a]
        
        # Blend P (approximate; true Bayesian fusion is more complex)
        P = p_stab * self.P_stable[a] + p_vol * self.P_volatile[a]
        
        return theta, P

    def select(self, context):
        ctx = np.asarray(context, dtype=float)

        # Deadline guard: if we are running out of wall-clock, commit to the
        # arm with the highest current mean estimate (no further exploration).
        if self._check_deadline():
            scores = []
            for a in range(self.n):
                theta, _ = self._get_blended_state(a)
                scores.append(float(theta @ ctx))
            return int(np.argmax(scores))

        self.step_count += 1
        u = self._adaptive_u()

        best = -np.inf
        best_arm = 0
        for a in range(self.n):
            theta, P = self._get_blended_state(a)
            mean = float(theta @ ctx)
            std = float(np.sqrt(max(ctx @ P @ ctx, 0.0)))
            c = self.count[a]
            bonus = u * std / (1.0 + self.count_scale * np.sqrt(c))
            if c == 0:
                bonus += 2.0  # encourage exploring unseen arms early
            score = mean + bonus
            if score > best:
                best = score
                best_arm = a
        return int(best_arm)

    def update(self, context, arm, reward):
        ctx = np.asarray(context, dtype=float)
        reward = float(reward)

        if self._check_deadline():
            return

        # Compute innovations for both modes before predict
        pred_stable = float(self.theta_stable[arm] @ ctx)
        pred_volatile = float(self.theta_volatile[arm] @ ctx)
        
        innov_stable = reward - pred_stable
        innov_volatile = reward - pred_volatile

        # Update regime probability based on which mode explains the innovation better
        self._update_regime(arm, innov_stable, innov_volatile)

        # Kalman predict step for both modes
        self._predict_stable(arm)
        self._predict_volatile(arm)

        # Kalman update step for both modes
        self._update_stable(arm, ctx, reward)
        self._update_volatile(arm, ctx, reward)

        self.count[arm] += 1

        # Periodic numerical stabilization every 50 steps
        if self.step_count % 50 == 0:
            for a in range(self.n):
                for P in [self.P_stable[a], self.P_volatile[a]]:
                    # Check for NaNs or infinities
                    if not np.all(np.isfinite(P)):
                        self.P_stable[a] = np.eye(self.d) * 10.0
                        self.theta_stable[a] = np.zeros(self.d)
                        self.P_volatile[a] = np.eye(self.d) * 10.0
                        self.theta_volatile[a] = np.zeros(self.d)
                        self.count[a] = 0
                        self.regime_prob[a] = 0.1
                        break
# EVOLVE-BLOCK-END
