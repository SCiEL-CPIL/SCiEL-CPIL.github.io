"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Multi-timescale sliding window LinUCB for non-stationary bandits."""

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.lam = 0.1  # Small regularization for stability

        # Short window: fast adaptation (gamma ~ 0.98)
        self.gamma_short = 0.98
        self.A_short = [self.lam * np.eye(dim) for _ in range(n_arms)]
        self.b_short = [np.zeros(dim) for _ in range(n_arms)]

        # Long window: stable baseline (gamma ~ 0.995)
        self.gamma_long = 0.995
        self.A_long = [self.lam * np.eye(dim) for _ in range(n_arms)]
        self.b_long = [np.zeros(dim) for _ in range(n_arms)]

        # UCB parameter
        self.alpha = 1.0

        # Deadline guard
        self.t0 = time.time()
        self.deadline = 145.0  # Safety margin before 150s kill

        # Track last arm for potential exploitation
        self.last_arm = None

    def _check_deadline(self):
        return (time.time() - self.t0) > self.deadline

    def _ucb_score_short(self, arm, context):
        A = self.A_short[arm]
        b = self.b_short[arm]
        inv_A = np.linalg.inv(A)
        theta = inv_A @ b
        mean = theta @ context
        var = context @ inv_A @ context
        return mean + self.alpha * np.sqrt(var)

    def _ucb_score_long(self, arm, context):
        A = self.A_long[arm]
        b = self.b_long[arm]
        inv_A = np.linalg.inv(A)
        theta = inv_A @ b
        mean = theta @ context
        var = context @ inv_A @ context
        return mean + self.alpha * 0.5 * np.sqrt(var)

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)

        # If past deadline, just return a fixed arm (safe fallback)
        if self._check_deadline():
            return 0

        scores_short = []
        scores_long = []
        for a in range(self.n):
            scores_short.append(self._ucb_score_short(a, context))
            scores_long.append(self._ucb_score_long(a, context))

        scores_short = np.array(scores_short)
        scores_long = np.array(scores_long)

        # Blend: primarily use short-window UCB, but regularize with long-window
        # Use short-window for exploration priority, long-window for exploitation stability
        blended = 0.7 * scores_short + 0.3 * scores_long

        return int(np.argmax(blended))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)

        # Update short window (fast forgetting)
        self.A_short[arm] = self.gamma_short * self.A_short[arm] + np.outer(context, context)
        self.b_short[arm] = self.gamma_short * self.b_short[arm] + reward * context

        # Update long window (slow forgetting)
        self.A_long[arm] = self.gamma_long * self.A_long[arm] + np.outer(context, context)
        self.b_long[arm] = self.gamma_long * self.b_long[arm] + reward * context

        self.last_arm = arm
# EVOLVE-BLOCK-END
