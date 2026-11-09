"""Non-stationary linear contextual bandit policy with multi-timescale adaptive forgetting and UCB exploration."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary LinUCB with multi-timescale estimation:
    - Short-window model for rapid detection of weight shifts
    - Long-window model for stable baseline estimation
    - Per-arm adaptive forgetting based on reward deviation from EMA
    - Deadline guard for safety
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.dim = dim

        # Short-window (fast adaptation) parameters
        self.gamma_short = 0.95
        self.A_short = [np.eye(dim) * 0.1 for _ in range(n_arms)]
        self.b_short = [np.zeros(dim) for _ in range(n_arms)]

        # Long-window (stable baseline) parameters
        self.gamma_long = 0.999
        self.A_long = [np.eye(dim) * 0.1 for _ in range(n_arms)]
        self.b_long = [np.zeros(dim) for _ in range(n_arms)]

        # Exploration parameter with decay
        self.alpha_init = 2.0
        self.alpha_decay = 0.001
        self.alpha = self.alpha_init

        # Per-arm EMA of rewards for non-stationarity detection
        self.reward_ema = [0.0 for _ in range(n_arms)]
        self.reward_ema_alpha = 0.2

        # Time tracking
        self.t = 0

        # Deadline guard
        self.start_time = time.time()
        self.deadline = 140.0  # 140s to leave 10s safety margin

    def _is_time_up(self):
        return time.time() > self.deadline

    def select(self, context):
        self.t += 1

        # Deadline guard: if time is running out, use long-window (more stable) model
        use_long_only = self._is_time_up()

        # Adaptive alpha decay to reduce over-exploration in later rounds
        self.alpha = max(0.5, self.alpha_init - self.alpha_decay * self.t)

        context = np.asarray(context, dtype=np.float64)

        scores = np.zeros(self.n)
        for a in range(self.n):
            if use_long_only:
                # Use long-window model only
                A = self.A_long[a] + np.eye(self.d) * 1e-6
                try:
                    inv_A = np.linalg.inv(A)
                    theta = inv_A @ self.b_long[a]
                except np.linalg.LinAlgError:
                    theta = np.zeros(self.d)
                    inv_A = np.eye(self.d)

                mean = theta @ context
                unc = np.sqrt(max(0.0, context @ inv_A @ context))
                scores[a] = mean + self.alpha * unc
            else:
                # Multi-timescale: combine short and long window estimates
                # Short window for rapid adaptation
                A_s = self.A_short[a] + np.eye(self.d) * 1e-6
                try:
                    inv_A_s = np.linalg.inv(A_s)
                    theta_s = inv_A_s @ self.b_short[a]
                except np.linalg.LinAlgError:
                    theta_s = np.zeros(self.d)
                    inv_A_s = np.eye(self.d)

                mean_s = theta_s @ context
                unc_s = np.sqrt(max(0.0, context @ inv_A_s @ context))

                # Long window for stable baseline
                A_l = self.A_long[a] + np.eye(self.d) * 1e-6
                try:
                    inv_A_l = np.linalg.inv(A_l)
                    theta_l = inv_A_l @ self.b_long[a]
                except np.linalg.LinAlgError:
                    theta_l = np.zeros(self.d)
                    inv_A_l = np.eye(self.d)

                mean_l = theta_l @ context
                unc_l = np.sqrt(max(0.0, context @ inv_A_l @ context))

                # Blend: prioritize short window for mean, use combined uncertainty
                # Weight based on how much the two models disagree
                disagreement = abs(mean_s - mean_l)
                # When models agree, trust long window more; when they disagree, trust short window
                short_weight = min(1.0, 0.5 + disagreement * 0.5)

                mean = short_weight * mean_s + (1 - short_weight) * mean_l
                # Uncertainty: use the larger of the two (conservative)
                unc = max(unc_s, unc_l * 0.8)

                scores[a] = mean + self.alpha * unc

        # Clip to reasonable range
        scores = np.clip(scores, -10.0, 10.0)

        # Small random tie-breaking
        if np.allclose(scores, scores[0]):
            return int(np.random.randint(self.n))

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)

        # Update EMA
        if self.t > 1:
            self.reward_ema[arm] = self.reward_ema_alpha * reward + (1 - self.reward_ema_alpha) * self.reward_ema[arm]
        else:
            self.reward_ema[arm] = reward

        # Detect non-stationarity: if reward deviates significantly from EMA, increase forgetting
        deviation = abs(reward - self.reward_ema[arm])
        # Adaptive gamma: higher deviation -> faster forgetting (lower gamma)
        gamma_short_target = max(0.90, 0.95 - deviation * 0.05)
        self.gamma_short = 0.8 * self.gamma_short + 0.2 * gamma_short_target

        # Update short-window model
        self.A_short[arm] = self.gamma_short * self.A_short[arm] + np.outer(context, context)
        self.b_short[arm] = self.gamma_short * self.b_short[arm] + reward * context

        # Update long-window model
        self.A_long[arm] = self.gamma_long * self.A_long[arm] + np.outer(context, context)
        self.b_long[arm] = self.gamma_long * self.b_long[arm] + reward * context

        # Periodic renormalization to prevent numerical issues
        if self.t % 100 == 0:
            for a in range(self.n):
                for A_list in [self.A_short, self.A_long]:
                    try:
                        eigvals = np.linalg.eigvalsh(A_list[a])
                        if np.min(eigvals) < 1e-4:
                            A_list[a] += np.eye(self.d) * 1e-4
                    except np.linalg.LinAlgError:
                        A_list[a] = np.eye(self.d) * 0.1
# EVOLVE-BLOCK-END
