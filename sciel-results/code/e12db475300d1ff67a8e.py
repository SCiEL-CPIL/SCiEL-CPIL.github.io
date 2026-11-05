# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Global linear bandit with one-hot arm indicators and adaptive forgetting."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.d_aug = dim + n_arms
        self.ridge = 0.15
        self.V = np.eye(self.d_aug) * self.ridge
        self.V_inv = np.eye(self.d_aug) / self.ridge
        self.b = np.zeros(self.d_aug)
        self.t = 0
        self.total_reward = 0.0
        self.base_gamma = 0.95
        self.gamma_min = 0.75
        self.gamma_max = 0.995
        self.gamma_schedule_len = 200
        self.C = 1.0
        self.C_min = 0.5
        self.C_max = 2.0
        self.start_time = time.time()
        self.deadline = 140.0
        self.one_hot = [np.zeros(self.d_aug) for _ in range(n_arms)]
        for a in range(n_arms):
            self.one_hot[a][dim + a] = 1.0
        self.recompute_interval = 50

    def _get_gamma(self):
        if self.t < self.gamma_schedule_len:
            frac = self.t / self.gamma_schedule_len
            return self.base_gamma + (self.gamma_max - self.base_gamma) * frac
        return self.gamma_max

    def _get_C(self):
        if self.t < 50:
            return self.C
        frac = min(1.0, (self.t - 50) / 500.0)
        return self.C_max - (self.C_max - self.C_min) * frac

    def _recompute_V_inv(self):
        self.V_inv = np.linalg.inv(self.V)

    def select(self, context):
        self.t += 1
        elapsed = time.time() - self.start_time
        if elapsed > self.deadline:
            theta = self.V_inv @ self.b
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                x_aug = np.concatenate([context, self.one_hot[a]])
                score = theta @ x_aug
                if score > best_score:
                    best_score = score
                    best_arm = a
            return int(best_arm)

        C = self._get_C()
        theta = self.V_inv @ self.b
        scores = []
        for a in range(self.n):
            x_aug = np.concatenate([context, self.one_hot[a]])
            mu = theta @ x_aug
            v = x_aug @ self.V_inv @ x_aug
            bonus = C * np.sqrt(max(v, 0.0))
            scores.append(mu + bonus)
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.total_reward += reward
        gamma = self._get_gamma()
        x_aug = np.concatenate([context, self.one_hot[arm]])

        # Update V: V_new = gamma * V_old + x x^T
        self.V = gamma * self.V + np.outer(x_aug, x_aug)

        # Update V_inv via Sherman-Morrison
        W_inv = self.V_inv / gamma
        W_inv_x = W_inv @ x_aug
        denom = 1.0 + x_aug @ W_inv_x
        if abs(denom) > 1e-12:
            self.V_inv = W_inv - np.outer(W_inv_x, x_aug) @ W_inv / denom
        else:
            self._recompute_V_inv()

        # Periodic recompute to prevent drift
        if self.t % self.recompute_interval == 0:
            self._recompute_V_inv()

        # Update b
        self.b = gamma * self.b + reward * x_aug
# EVOLVE-BLOCK-END
