"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Per-Arm RLS with context-adaptive forgetting factor and UCB exploration."""

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.A = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.A_inv = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.lam_base = 0.98
        self.gamma = 0.01
        self.alpha = 1.0
        self.start_time = time.time()
        self.deadline = 150.0 - 5.0  # 5s safety margin
        self._round = 0

    def select(self, context):
        # Deadline guard
        if time.time() > self.deadline:
            # Fallback: pick arm with highest mean reward so far
            means = [np.sum(self.b[a]) / max(1, self._round) for a in range(self.n)]
            return int(np.argmax(means))

        self._round += 1
        scores = []
        for a in range(self.n):
            pred = self.theta[a] @ context
            # Confidence term using A_inv
            conf = np.sqrt(self.alpha * (context @ self.A_inv[a] @ context))
            scores.append(pred + conf)
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        if time.time() > self.deadline:
            return

        # Context-adaptive forgetting factor
        ctx_norm_sq = float(context @ context)
        lam = self.lam_base + self.gamma * ctx_norm_sq
        lam = min(lam, 0.999)  # cap to avoid instability

        x = context
        A_inv = self.A_inv[arm]
        theta = self.theta[arm]

        # RLS update: A_inv_new = (1/lam) * A_inv - (1/lam) * A_inv @ x @ x^T @ A_inv / (lam + x^T @ A_inv @ x)
        # More standard RLS: 
        # K = A_inv @ x / (lam + x^T @ A_inv @ x)
        # A_inv_new = (A_inv - K @ x^T @ A_inv) / lam  ... but let's use the standard form
        # Actually, for discounted RLS:
        # A_t = lam * A_{t-1} + x x^T
        # A_inv_t = (1/lam) * A_inv_{t-1} - (1/lam^2) * A_inv_{t-1} x x^T A_inv_{t-1} / (1/lam + x^T A_inv_{t-1} x)
        
        # Let's use the simpler recursive form:
        # A_inv_new = (1/lam) * (A_inv - (A_inv @ x @ x^T @ A_inv) / (lam + x^T @ A_inv @ x))
        # Wait, that's not quite right. Let me derive it.
        # A_t = lam * A_{t-1} + x x^T
        # A_inv_t = A_t^{-1}
        # Using matrix inversion lemma: (A + u v^T)^{-1} = A^{-1} - A^{-1} u v^T A^{-1} / (1 + v^T A^{-1} u)
        # Here A_t = lam * A_{t-1} + x x^T
        # A_t^{-1} = (lam * A_{t-1} + x x^T)^{-1}
        # = (1/lam) * (A_{t-1} + (1/lam) x x^T)^{-1}
        # Let B = A_{t-1}, c = 1/lam
        # (B + c x x^T)^{-1} = B^{-1} - B^{-1} c x x^T B^{-1} / (1 + c x^T B^{-1} x)
        # So A_t^{-1} = (1/lam) * [A_inv_{t-1} - A_inv_{t-1} c x x^T A_inv_{t-1} / (1 + c x^T A_inv_{t-1} x)]
        # = (1/lam) * A_inv_{t-1} - (c/lam) * A_inv_{t-1} x x^T A_inv_{t-1} / (1 + c x^T A_inv_{t-1} x)
        # = (1/lam) * A_inv_{t-1} - (1/lam^2) * A_inv_{t-1} x x^T A_inv_{t-1} / (1 + (1/lam) x^T A_inv_{t-1} x)
        # = (1/lam) * A_inv_{t-1} - (1/lam^2) * A_inv_{t-1} x x^T A_inv_{t-1} / ((lam + x^T A_inv_{t-1} x) / lam)
        # = (1/lam) * A_inv_{t-1} - (1/lam) * A_inv_{t-1} x x^T A_inv_{t-1} / (lam + x^T A_inv_{t-1} x)
        
        Ax = A_inv @ x
        denom = lam + x @ Ax
        K = Ax / denom
        A_inv_new = (1.0 / lam) * (A_inv - np.outer(Ax, x) @ A_inv)
        # Wait, let me re-check. The formula is:
        # A_inv_new = (1/lam) * A_inv - (1/lam) * (A_inv @ x) (x^T @ A_inv) / (lam + x^T A_inv x)
        # = (1/lam) * (A_inv - A_inv @ x @ x^T @ A_inv / (lam + x^T A_inv x))
        
        # Let me just compute it directly:
        # A_inv_new = (1/lam) * A_inv - (1/lam^2) * np.outer(Ax, Ax) / (1 + (1/lam) * x @ Ax)
        # Hmm, let me use a cleaner formulation.
        
        # Actually, let's just do:
        # A_inv_new = (1/lam) * A_inv - (1/lam^2) * np.outer(Ax, Ax) / (1 + (1/lam) * (x @ Ax))
        # This is equivalent to what I had.
        
        # Let me re-derive more carefully.
        # We want A_t^{-1} where A_t = lam * A_{t-1} + x x^T
        # Let M = lam * A_{t-1} + x x^T
        # M^{-1} = (lam * A_{t-1})^{-1} - (lam * A_{t-1})^{-1} x x^T (lam * A_{t-1})^{-1} / (1 + x^T (lam * A_{t-1})^{-1} x)
        # = (1/lam) A_{t-1}^{-1} - (1/lam^2) A_{t-1}^{-1} x x^T A_{t-1}^{-1} / (1 + (1/lam) x^T A_{t-1}^{-1} x)
        
        A_inv_old = A_inv
        Ax_old = A_inv_old @ x
        scalar = x @ Ax_old
        denom = 1.0 + scalar / lam
        A_inv_new = (1.0 / lam) * A_inv_old - (1.0 / (lam ** 2)) * np.outer(Ax_old, Ax_old) / denom
        
        # Update theta: theta_t = (1/lam) theta_{t-1} + (1/lam) A_inv_new x (reward - x^T theta_{t-1})
        # Wait, theta_t = A_t^{-1} b_t where b_t = lam * b_{t-1} + reward * x
        # theta_t = A_t^{-1} (lam * b_{t-1} + reward * x)
        # = A_t^{-1} lam * A_{t-1} theta_{t-1} + A_t^{-1} reward * x
        # = (A_t^{-1} lam A_{t-1}) theta_{t-1} + reward * A_t^{-1} x
        # Note that A_t^{-1} lam A_{t-1} is not simply (1/lam) A_t^{-1} A_t ... 
        # Let's just compute theta directly: theta = A_inv_new @ b_new
        # b_new = lam * b_old + reward * x
        
        b_new = lam * self.b[arm] + reward * x
        theta_new = A_inv_new @ b_new
        
        self.A_inv[arm] = A_inv_new
        self.b[arm] = b_new
        self.theta[arm] = theta_new
# EVOLVE-BLOCK-END
