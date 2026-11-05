# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Sliding-window linear bandit with per-arm covariance-based UCB bonus."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.window = 50
        self.ridge = 1e-3
        self.beta0 = 1.0
        self.beta_decay = 0.999

        # Per-arm ring buffers
        self.ctx_buf = [np.zeros((self.window, dim)) for _ in range(n_arms)]
        self.rw_buf = [np.zeros(self.window) for _ in range(n_arms)]
        self.count = [0 for _ in range(n_arms)]
        self.ptr = [0 for _ in range(n_arms)]

        # Cached theta and inv(A) for fast select
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.inv_A = [np.eye(dim) for _ in range(n_arms)]
        self.A = [self.ridge * np.eye(dim) for _ in range(n_arms)]

        # Time tracking
        self.t0 = time.time()
        self.deadline = 140.0  # safety margin before 150s kill

    def _recompute(self, arm):
        """Recompute A, inv_A, and theta for a given arm from the sliding window."""
        c = self.count[arm]
        if c == 0:
            self.A[arm] = self.ridge * np.eye(self.d)
            self.inv_A[arm] = np.eye(self.d) / self.ridge
            self.theta[arm] = np.zeros(self.d)
            return

        # Gather the most recent c contexts and rewards
        start = (self.ptr[arm] - c) % self.window
        if start + c <= self.window:
            ctx = self.ctx_buf[arm][start:start + c]
            rw = self.rw_buf[arm][start:start + c]
        else:
            part1 = self.ctx_buf[arm][start:]
            part2 = self.ctx_buf[arm][:start + c - self.window]
            ctx = np.vstack([part1, part2])
            part1r = self.rw_buf[arm][start:]
            part2r = self.rw_buf[arm][:start + c - self.window]
            rw = np.concatenate([part1r, part2r])

        # Build A = sum(outer(x_i, x_i)) + ridge * I
        A = ctx.T @ ctx + self.ridge * np.eye(self.d)
        b = ctx.T @ rw
        self.A[arm] = A
        try:
            inv_A = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            inv_A = np.linalg.pinv(A)
        self.inv_A[arm] = inv_A
        self.theta[arm] = inv_A @ b

    def select(self, context):
        # Deadline guard
        if time.time() > self.deadline:
            return int(np.argmax(self.theta)) if self.count[0] > 0 else 0

        # Adaptive beta: starts high for exploration, decays over time
        total_rounds = sum(self.count)
        beta = self.beta0 * (1.0 + 0.5 * np.exp(-total_rounds / 20.0))

        scores = []
        for a in range(self.n):
            theta = self.theta[a]
            inv_A = self.inv_A[a]
            mean = theta @ context
            # Variance of the prediction
            var = context @ inv_A @ context
            scores.append(mean + beta * np.sqrt(var))
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Store in ring buffer
        self.ctx_buf[arm][self.ptr[arm]] = context
        self.rw_buf[arm][self.ptr[arm]] = reward
        self.ptr[arm] = (self.ptr[arm] + 1) % self.window
        self.count[arm] = min(self.count[arm] + 1, self.window)

        # Recompute only for the updated arm
        self._recompute(arm)
# EVOLVE-BLOCK-END
