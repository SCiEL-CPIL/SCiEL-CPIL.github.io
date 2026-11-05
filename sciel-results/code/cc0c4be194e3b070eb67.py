# EVOLVE-BLOCK-START
import time
import numpy as np


class Policy:
    """Per-arm Kalman filter (random-walk weights) with adaptive process noise.

    Each arm keeps a mean weight vector theta and a covariance P. Between
    observations the state performs a random-walk "predict" step adding
    process noise Q; an observation performs a rank-1 "update". Q adapts to
    an EMA of the squared innovation, so it reacts quickly to non-stationary
    weight changes but stays small when stable. The UCB exploration
    coefficient is adaptive (warmup boost, then slow decay with a floor).

    Refined with:
      - Huber-style robustification of the innovation (tuned delta)
      - Sliding-window inverse-covariance reset when the innovation residual
        exceeds a dynamic threshold
      - Context-norm adaptive process noise
      - Periodic numerical stabilization
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self.start_time = time.time()
        self.deadline = 115.0  # safety margin before the 120s kill

        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        # Start with large uncertainty so unexplored arms are favoured.
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        self.count = [0 for _ in range(n_arms)]

        # Adaptive process noise parameters
        self.q0 = 0.05          # base process noise
        self.q_max = 5.0        # cap on adaptive process noise
        self.innov_ema = [0.0] * n_arms   # EMA of squared innovation per arm
        self.innov_ema_alpha = 0.3
        self.innov_base = 1.0   # baseline squared-innov threshold

        # Context-norm adaptive scaling for Q
        self.ctx_norm_gamma = 0.05  # how strongly context norm scales Q
        self.ctx_norm_cap = 3.0     # cap on the context-norm multiplier

        # Huber robustification
        self.huber_delta = 2.0  # increased from 1.5 to reduce bias

        # Sliding-window reset mechanism
        self.reset_threshold_scale = 2.5  # multiplier on std for triggering reset
        self.reset_cov_scale = 10.0       # reset P to this times identity (same as init)
        self.reset_window = 8             # number of consecutive large innovations to trigger reset
        self.consecutive_large_innov = [0] * n_arms

        # UCB exploration parameters
        self.u_scale = 1.5      # UCB exploration multiplier
        self.count_scale = 0.4  # sqrt(count) dampening of the bonus
        self.warmup_steps = 20
        self.jitter = 1e-6      # numerical stabiliser for P
        self.step_count = 0

    def _check_deadline(self):
        return (time.time() - self.start_time) > self.deadline

    def _adaptive_q(self, arm, innov, ctx_norm_sq):
        """Adaptive process noise from an EMA of squared innovation,
        further scaled by context magnitude."""
        c = self.count[arm]
        if c < 3:
            # Early phase: moderate exploration via larger Q
            return self.q0 * 3.0

        # Update EMA of squared innovation
        self.innov_ema[arm] = (1.0 - self.innov_ema_alpha) * self.innov_ema[arm] + \
                               self.innov_ema_alpha * (innov * innov)

        innov_ema = self.innov_ema[arm]
        excess = max(0.0, innov_ema - self.innov_base)
        q_scale = 1.0 + 3.0 * (1.0 - np.exp(-1.0 * excess))
        q_scale = min(q_scale, 50.0)

        # Context-norm adaptive scaling
        ctx_mult = 1.0 + self.ctx_norm_gamma * ctx_norm_sq
        ctx_mult = min(ctx_mult, self.ctx_norm_cap)

        q = min(self.q_max, self.q0 * q_scale * ctx_mult)
        return q

    def _adaptive_u(self):
        """Adapt exploration coefficient: warmup boost, then slow decay with floor."""
        if self.step_count < self.warmup_steps:
            return self.u_scale * 1.5
        factor = 1.0 / (1.0 + self.step_count / 1000.0)
        return self.u_scale * max(factor, 0.5)

    def _huber_innov(self, innov, std):
        """Robustify innovation using Huber loss: clip large residuals."""
        if std <= 0:
            return innov
        ratio = abs(innov) / std
        if ratio <= self.huber_delta:
            return innov
        # For |innov|/std > delta, use linear scaling
        sign = 1.0 if innov > 0 else -1.0
        return sign * self.huber_delta * std

    def _predict(self, a, q):
        self.P[a] = self.P[a] + q * np.eye(self.d) + self.jitter * np.eye(self.d)

    def _update(self, a, ctx, reward):
        P = self.P[a]
        theta = self.theta[a]
        pred = float(theta @ ctx)
        innov_raw = reward - pred
        var = float(ctx @ P @ ctx) + 1.0 + self.jitter  # R=1 observation noise
        std = float(np.sqrt(var))
        # Robustify innovation
        innov = self._huber_innov(innov_raw, std)
        K = P @ ctx / var          # Kalman gain
        self.theta[a] = theta + (K * innov)
        # Joseph-form rank-1 update: P <- (I - K x') P (I - K x')'
        IK = np.eye(self.d) - np.outer(K, ctx)
        self.P[a] = IK @ P @ IK.T
        # Symmetrise and clip for numerical safety.
        self.P[a] = 0.5 * (self.P[a] + self.P[a].T)
        np.fill_diagonal(self.P[a], np.maximum(np.diag(self.P[a]), 0.0))
        self.count[a] += 1
        return innov_raw  # return raw innovation for tracking

    def _check_reset(self, a, innov_raw):
        """Check if we should reset covariance due to sustained large innovations."""
        P = self.P[a]
        var = float(np.trace(P)) / self.d  # average variance
        std = float(np.sqrt(max(var, 0.0)))
        threshold = self.reset_threshold_scale * std + 0.5

        if abs(innov_raw) > threshold:
            self.consecutive_large_innov[a] += 1
        else:
            self.consecutive_large_innov[a] = 0

        if self.consecutive_large_innov[a] >= self.reset_window:
            # Reset covariance to a moderate value
            self.P[a] = np.eye(self.d) * self.reset_cov_scale
            self.consecutive_large_innov[a] = 0
            return True
        return False

    def _stabilize(self):
        """Periodic numerical stabilization: check for NaN/Inf and reset."""
        for a in range(self.n):
            P = self.P[a]
            if not np.all(np.isfinite(P)) or not np.all(np.isfinite(self.theta[a])):
                self.P[a] = np.eye(self.d) * 10.0
                self.theta[a] = np.zeros(self.d)
                self.count[a] = 0
                self.innov_ema[a] = 0.0
                self.consecutive_large_innov[a] = 0

    def select(self, context):
        ctx = np.asarray(context, dtype=float)
        # Deadline guard: commit to the best mean estimate (no exploration).
        if self._check_deadline():
            return int(np.argmax([float(t @ ctx) for t in self.theta]))

        self.step_count += 1
        u = self._adaptive_u()

        best = -np.inf
        best_arm = 0
        for a in range(self.n):
            theta = self.theta[a]
            mean = float(theta @ ctx)
            std = float(np.sqrt(max(ctx @ self.P[a] @ ctx, 0.0)))
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

        self.t += 1
        # Compute innovation before predict, then adapt process noise.
        pred = float(self.theta[arm] @ ctx)
        innov = reward - pred
        ctx_norm_sq = float(ctx @ ctx)
        q = self._adaptive_q(arm, innov, ctx_norm_sq)
        self._predict(arm, q)
        self._update(arm, ctx, reward)
        # Check for covariance reset
        self._check_reset(arm, innov)

        # Periodic numerical stabilization every 100 steps
        if self.step_count % 100 == 0:
            self._stabilize()
# EVOLVE-BLOCK-END
