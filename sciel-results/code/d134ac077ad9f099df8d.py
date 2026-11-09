"""Recurrent Linear Bandit with SMW rank-1 inverse updates and deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Policy:
    """Recurrent linear bandit: context augmented with temporal history,
    SMW rank-1 inverse updates, adaptive forgetting, deadline guard."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.d_aug = dim + n_arms + 1          # raw + one_hot(prev_arm) + prev_reward
        self.A_inv = [np.eye(self.d_aug) for _ in range(n_arms)]
        self.b = [np.zeros(self.d_aug) for _ in range(n_arms)]
        self.theta = [np.zeros(self.d_aug) for _ in range(n_arms)]
        self.gamma = 0.99                       # base forgetting (decay)
        self.alpha = 1.0                        # ridge / exploration scale
        self.prev_reward = 0.0
        self.prev_arm = None
        self._t0 = time.time()
        self._deadline = 145.0                  # safety margin before 150s kill
        self._step = 0
        self._reward_scale = 1.0
        self._last_resid = 0.0

    def _aug(self, context):
        ctx = np.asarray(context, dtype=np.float64)
        aug = np.empty(self.d_aug, dtype=np.float64)
        aug[:self.d] = ctx
        if self.prev_arm is not None:
            aug[self.d + self.prev_arm] = 1.0
        aug[-1] = self.prev_reward
        return aug

    def select(self, context):
        # Deadline guard: if close to the limit, return a stable greedy arm.
        if time.time() > self._deadline:
            return self._safe_arm(context)

        aug = self._aug(context)
        best_arm = 0
        best_score = -np.inf
        for a in range(self.n):
            Ai = self.A_inv[a]
            # mean prediction
            mu = self.theta[a] @ aug
            # variance term via quadratic form (use cached theta = A_inv b)
            var = aug @ (Ai @ aug)
            if var < 0.0:
                var = 0.0
            score = mu + self.alpha * np.sqrt(var)
            if score > best_score:
                best_score = score
                best_arm = a
        self._step += 1
        return int(best_arm)

    def _safe_arm(self, context):
        aug = self._aug(context)
        best_arm = 0
        best_mu = -np.inf
        for a in range(self.n):
            mu = self.theta[a] @ aug
            if mu > best_mu:
                best_mu = mu
                best_arm = a
        return int(best_arm)

    def update(self, context, arm, reward):
        # Deadline guard: avoid heavy work near the limit.
        if time.time() > self._deadline:
            self.prev_reward = reward
            self.prev_arm = arm
            return

        aug = self._aug(context)
        Ai = self.A_inv[arm]
        b = self.b[arm]

        # Prediction residual for adaptive forgetting
        pred = self.theta[arm] @ aug
        resid = abs(reward - pred)
        self._last_resid = resid
        # Track reward scale for normalizing residual
        self._reward_scale = max(self._reward_scale * 0.999 + 1e-6,
                                 abs(reward) + 1e-6)

        # --- Sherman-Morrison-Woodbury rank-1 update:
        # A_new = gamma * A_old + x x^T
        # A_inv_new = A_inv_old / gamma - (A_inv_old x x^T A_inv_old / gamma)
        #            / (1 + x^T A_inv_old x / gamma)
        # We fold the decay into the inverse directly.
        # Let A_old_inv = Ai. We want inv(gamma*A_old + x x^T).
        # = (1/gamma) * inv(A_old + (1/gamma) x x^T)
        # SMW: inv(M + c x x^T) = inv(M) - c inv(M) x x^T inv(M) / (1 + c x^T inv(M) x)
        # with M = A_old, c = 1/gamma.
        inv_gamma = 1.0 / self.gamma
        c = inv_gamma
        M_inv = Ai
        Mx = M_inv @ aug
        denom = 1.0 + c * (aug @ Mx)
        if denom <= 1e-12:
            denom = 1e-12
        # inv(M + c x x^T)
        M_plus = M_inv - (c / denom) * np.outer(Mx, Mx)
        Ai_new = inv_gamma * M_plus

        # Adaptive forgetting: if residual is large relative to scale,
        # reset more aggressively (shrink the inverse toward identity-ish).
        norm_resid = resid / (self._reward_scale + 1e-6)
        if norm_resid > 2.0:
            # Strong drift: blend toward a near-prior inverse.
            # Identity scaled so the effective prior is reasonable.
            reset = np.eye(self.d_aug)
            Ai_new = 0.5 * Ai_new + 0.5 * reset
            # Also decay b to reduce stale influence
            b = 0.5 * b + reward * aug

        # Update b with forgetting
        b = self.gamma * b + reward * aug

        # Update theta = A_inv b
        theta_new = Ai_new @ b

        self.A_inv[arm] = Ai_new
        self.b[arm] = b
        self.theta[arm] = theta_new

        # Periodic numerical stabilization: re-orthogonalize / symmetrize
        if self._step % 200 == 0:
            for a in range(self.n):
                Ai = self.A_inv[a]
                # Symmetrize to remove drift
                Ai = 0.5 * (Ai + Ai.T)
                self.A_inv[a] = Ai
                self.theta[a] = Ai @ self.b[a]

        self.prev_reward = reward
        self.prev_arm = arm
# EVOLVE-BLOCK-END
