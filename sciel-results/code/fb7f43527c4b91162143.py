"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Adaptive LinUCB with SMW inverse updates, low decay, and deadline guard."""

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        # Store inverse of A (A0 = I, so A0_inv = I)
        self.A_inv = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self._start_time = time.time()
        self._deadline = self._start_time + 140.0  # 10s safety margin under 150s
        self._alpha = 0.5
        self._decay = 0.95
        self._fallback_arm = 0

    def _get_theta(self, a):
        # theta = A_inv @ b
        return self.A_inv[a] @ self.b[a]

    def select(self, context):
        # Deadline guard: if near limit, return best-known arm to be safe
        if time.time() >= self._deadline:
            return self._fallback_arm

        scores = []
        for a in range(self.n):
            inv = self.A_inv[a]
            theta = inv @ self.b[a]
            mu = theta @ context
            # UCB bonus: alpha * sqrt(context^T A_inv context)
            val = context @ inv @ context
            if val < 0:
                val = 0.0
            scores.append(mu + self._alpha * np.sqrt(val))
        arm = int(np.argmax(scores))
        self._fallback_arm = arm
        return arm

    def update(self, context, arm, reward):
        # Apply decay to existing statistics
        self.b[arm] = self._decay * self.b[arm]
        # We need to also decay A_inv consistently. Since A = sum of outer products,
        # decay means A_new = decay*A_old + outer(x,x).
        # A_inv_new = (A_old + outer(x,x))^{-1} / decay  ... careful.
        # Let A' = decay * A_old. Then A_new = A' + outer(x,x).
        # A'_inv = A_old_inv / decay.
        # Then use SMW on A' + outer(x,x) with A'_inv.
        inv = self.A_inv[arm] / self._decay  # this is (decay*A_old)^{-1}
        x = context
        # SMW: (M + x x^T)^{-1} = M^{-1} - M^{-1} x x^T M^{-1} / (1 + x^T M^{-1} x)
        M_inv = inv
        M_inv_x = M_inv @ x
        denom = 1.0 + x @ M_inv_x
        if abs(denom) < 1e-12:
            denom = 1e-12
        self.A_inv[arm] = M_inv - (np.outer(M_inv_x, M_inv_x)) / denom
        self.b[arm] = self.b[arm] + reward * context
# EVOLVE-BLOCK-END
