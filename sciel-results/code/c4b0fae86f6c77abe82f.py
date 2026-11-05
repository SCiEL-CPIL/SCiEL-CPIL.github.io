"""Rewritten program for non-stationary linear contextual bandit."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit with:
    - Per-arm ridge regression with exponential forgetting (discounted least squares)
    - Adaptive forgetting rate based on prediction residuals
    - UCB-style exploration with arm-specific uncertainty
    - Deadline guard for safety
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.dim = dim

        # Per-arm parameters: V = (sum gamma^t x x^T + lambda I)^{-1}, b = sum gamma^t r x
        self.V = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]

        # Forgetting
        self.gamma = 0.95
        self.gamma_min = 0.85
        self.gamma_max = 0.995
        self.lam = 1.0  # Ridge regularization

        # Exploration
        self.alpha = 1.5  # UCB coefficient
        self.alpha_min = 0.6
        self.alpha_max = 3.5

        # Residual tracking for adaptive gamma
        self.residual_ema = np.zeros(n_arms)
        self.residual_alpha = 0.15
        self.max_residual = 1e-8

        # Deadline guard
        self.start_time = time.time()
        self.deadline = 118.0
        self.round = 0

        # Fallback arm tracking
        self.arm_mean_reward = [0.0] * n_arms
        self.arm_count = [0] * n_arms
        self.best_arm = 0
        self.best_reward = -np.inf

        # Reward tracking per arm for fallback
        self.recent_rewards = [[] for _ in range(n_arms)]
        self.recent_reward_window = 20

    def _check_deadline(self):
        if time.time() - self.start_time >= self.deadline:
            return True
        return False

    def _update_adaptive_gamma(self, arm, residual):
        """Adapt gamma based on prediction residual magnitude."""
        # Larger residuals suggest non-stationarity -> faster forgetting
        self.residual_ema[arm] = (1 - self.residual_alpha) * self.residual_ema[arm] + \
                                 self.residual_alpha * abs(residual)
        
        # Map residual to gamma: high residual -> low gamma (fast forgetting)
        # More sensitive thresholds for quicker response to regime changes
        r = self.residual_ema[arm]
        if r < 0.15:
            self.gamma = self.gamma_max
        elif r < 0.6:
            self.gamma = self.gamma_max - (self.gamma_max - 0.96) * ((r - 0.15) / 0.45)
        elif r < 1.8:
            self.gamma = 0.96 - (0.96 - 0.88) * ((r - 0.6) / 1.2)
        elif r < 3.5:
            self.gamma = 0.88 - (0.88 - self.gamma_min) * ((r - 1.8) / 1.7)
        else:
            self.gamma = self.gamma_min

    def _update_adaptive_alpha(self):
        """Adapt exploration coefficient over time."""
        r = self.round
        if r < 40:
            self.alpha = self.alpha_max
        elif r < 120:
            self.alpha = self.alpha_max - (self.alpha_max - 1.8) * (r - 40) / 80.0
        elif r < 350:
            self.alpha = 1.8 - (1.8 - 1.2) * (r - 120) / 230.0
        else:
            self.alpha = max(self.alpha_min, 1.2 - 0.3 * (r - 350) / 1000.0)

    def select(self, context):
        self.round += 1

        if self._check_deadline():
            return self.best_arm

        context = np.asarray(context, dtype=np.float64).flatten()

        self._update_adaptive_alpha()

        scores = np.zeros(self.n)
        for a in range(self.n):
            mean = self.theta[a] @ context
            # Uncertainty: context^T V_a context
            var = context @ self.V[a] @ context
            scores[a] = mean + self.alpha * np.sqrt(max(var, 1e-10))

        # Small random tiebreaker
        scores += np.random.normal(0, 0.08, self.n)

        arm = int(np.argmax(scores))
        return arm

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64).flatten()
        reward = float(reward)

        # Track for fallback
        self.arm_count[arm] += 1
        self.recent_rewards[arm].append(reward)
        if len(self.recent_rewards[arm]) > self.recent_reward_window:
            self.recent_rewards[arm].pop(0)
        self.arm_mean_reward[arm] = np.mean(self.recent_rewards[arm]) if self.recent_rewards[arm] else 0.0
        if self.arm_count[arm] >= 3:
            if self.arm_mean_reward[arm] > self.best_reward:
                self.best_reward = self.arm_mean_reward[arm]
                self.best_arm = arm

        # Prediction before update
        pred = self.theta[arm] @ context
        residual = reward - pred

        # Adapt gamma
        self._update_adaptive_gamma(arm, residual)
        gamma = self.gamma

        # Update V and b with forgetting
        # Old: V = (gamma * A_old + I)^{-1} where A_old was the accumulated A
        # We maintain V directly. The update is:
        # A_new = gamma * A_old + x x^T
        # V_new = A_new^{-1} = (gamma * A_old + x x^T)^{-1}
        # Using matrix inversion lemma on (gamma * A_old + x x^T):
        # Let M = gamma * A_old, then V_new = (M + x x^T)^{-1}
        # = M^{-1} - M^{-1} x x^T M^{-1} / (1 + x^T M^{-1} x)
        # M^{-1} = V_old / gamma
        # V_new = V_old/gamma - (V_old/gamma) x x^T (V_old/gamma) / (1 + x^T V_old/gamma x)
        # = V_old/gamma - V_old x x^T V_old / (gamma * (1 + x^T V_old x / gamma))
        # = V_old/gamma - V_old x x^T V_old / (gamma + x^T V_old x)

        V = self.V[arm]
        b = self.b[arm]

        Vx = V @ context
        denom = gamma + context @ Vx
        if denom < 1e-10:
            denom = 1e-10
        V_new = V / gamma - np.outer(Vx, Vx) / denom
        b_new = gamma * b + reward * context

        # Ensure symmetry and numerical stability
        V_new = (V_new + V_new.T) / 2.0

        self.V[arm] = V_new
        self.b[arm] = b_new
        self.theta[arm] = V_new @ b_new
# EVOLVE-BLOCK-END
