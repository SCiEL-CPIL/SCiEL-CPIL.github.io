# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Non-stationary Linear Contextual Bandit.
    Combines a global exponentially-weighted LinUCB (shared theta across arms)
    with per-arm EWMA and a Thompson-sampling-style bonus. Forgetting factor
    adapts to detected reward drift.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 112.0

        # Global exponentially-weighted ridge regression
        # A_t = gamma * A_{t-1} + x x^T, b_t = gamma * b_{t-1} + r x
        self.gamma = 0.96
        self.A = np.eye(dim) * 1.0
        self.b = np.zeros(dim)
        self.A_inv = np.eye(dim) * 1.0

        # Per-arm EWMA of rewards
        self.ewma_alpha = 0.1
        self.ewma = np.zeros(n_arms)
        self.ewma_counts = np.zeros(n_arms)

        # Per-arm recent reward buffer for drift detection
        self.arm_buf = {a: [] for a in range(n_arms)}
        self.buf_len = 25

        # Exploration
        self.base_beta = 1.0
        self.current_beta = self.base_beta

        # Drift tracking
        self.prev_reward = None
        self.drift_signal = 0.0
        self.drift_ema = 0.0

        # Best arm fallback
        self.best_arm = 0
        self.arm_recent_avg = np.zeros(n_arms)

    def _within_deadline(self):
        return time.time() < self.deadline

    def _update_inverse(self, context, reward):
        """
        Update A_inv using Sherman-Morrison for rank-1 update after scaling.
        A_new = gamma * A_old + x x^T
        A_inv_new = ?
        We maintain A_inv = A^{-1}. After scaling A by gamma, A_inv scales by 1/gamma.
        Then rank-1 update: (A + x x^T)^{-1} = A^{-1} - A^{-1} x x^T A^{-1} / (1 + x^T A^{-1} x)
        """
        # Scale existing inverse
        self.A_inv = self.A_inv / self.gamma

        # Rank-1 update
        Ax = self.A_inv @ context
        denom = 1.0 + np.dot(context, Ax)
        if abs(denom) > 1e-10:
            self.A_inv -= np.outer(Ax, Ax) / denom
        else:
            # Fallback: recompute
            self.A = self.gamma * self.A + np.outer(context, context)
            self.b = self.gamma * self.b + reward * context
            try:
                self.A_inv = np.linalg.inv(self.A)
            except np.linalg.LinAlgError:
                self.A_inv = np.eye(self.d)

    def _compute_theta(self):
        """Compute theta = A^{-1} b."""
        return self.A_inv @ self.b

    def select(self, context):
        self.t += 1

        if not self._within_deadline():
            return self.best_arm

        # Adaptive beta: start high, decay, but floor to keep some exploration
        progress = min(self.t / 400.0, 1.0)
        self.current_beta = self.base_beta * (1.0 - 0.5 * progress)
        # Add drift-based exploration boost
        drift_boost = min(self.drift_ema * 2.0, 0.5)
        self.current_beta += drift_boost

        if self.t <= 3:
            return int(np.random.randint(self.n))

        theta = self._compute_theta(context) if False else self._compute_theta()
        ctx = np.asarray(context, dtype=np.float64)

        scores = np.zeros(self.n)
        for a in range(self.n):
            # Mean prediction: (theta + e_a)^T x, where e_a is arm indicator
            # For global model with arm-specific bias: use theta for all arms
            # but add per-arm EWMA as a prior/bias
            mean = np.dot(theta, ctx)
            # Variance: x^T A^{-1} x (same for all arms in global model)
            var = np.dot(ctx, self.A_inv @ ctx)
            ucb = mean + self.current_beta * np.sqrt(max(var, 0.0))

            # Blend with per-arm EWMA
            if self.ewma_counts[a] > 5:
                blend = min(0.35 * (self.ewma_counts[a] / 30.0), 0.35)
                scores[a] = (1.0 - blend) * ucb + blend * self.ewma[a]
            else:
                scores[a] = ucb

        arm = int(np.argmax(scores))
        return arm

    def update(self, context, arm, reward):
        if not self._within_deadline():
            return

        ctx = np.asarray(context, dtype=np.float64)

        # Update global exponentially-weighted regression
        self._update_inverse(ctx, reward)
        self.b = self.gamma * self.b + reward * ctx

        # Update per-arm EWMA
        self.ewma[arm] = (1.0 - self.ewma_alpha) * self.ewma[arm] + self.ewma_alpha * reward
        self.ewma_counts[arm] += 1

        # Drift detection: compare consecutive rewards
        if self.prev_reward is not None:
            diff = abs(reward - self.prev_reward)
            self.drift_signal = 0.8 * self.drift_signal + 0.2 * diff
        self.prev_reward = reward
        self.drift_ema = 0.9 * self.drift_ema + 0.1 * self.drift_signal

        # Adaptive forgetting: higher drift -> lower gamma (faster forgetting)
        # Map drift_signal to gamma in [0.92, 0.99]
        if self.t > 10:
            target_gamma = 0.99 - 0.07 * min(self.drift_signal / 1.0, 1.0)
            target_gamma = max(0.92, min(0.99, target_gamma))
            self.gamma = 0.9 * self.gamma + 0.1 * target_gamma

        # Track per-arm recent average for best_arm fallback
        self.arm_buf[arm].append(reward)
        if len(self.arm_buf[arm]) > self.buf_len:
            self.arm_buf[arm].pop(0)
        if self.arm_buf[arm]:
            self.arm_recent_avg[arm] = np.mean(self.arm_buf[arm])

        # Update best arm
        valid_arms = [a for a in range(self.n) if len(self.arm_buf[a]) >= 3]
        if valid_arms:
            self.best_arm = max(valid_arms, key=lambda a: self.arm_recent_avg[a])
        else:
            self.best_arm = arm
# EVOLVE-BLOCK-END
