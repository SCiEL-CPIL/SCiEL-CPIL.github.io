"""Non-stationary linear contextual bandit policy with adaptive forgetting and UCB exploration."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary LinUCB with:
    - Adaptive forgetting factor based on reward volatility
    - Per-arm exponential moving average for baseline detection
    - Time-aware regularization to handle drift
    - Efficient linear algebra via solve instead of explicit inverse
    - Deadline guard for safety
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.dim = dim

        # Core matrices for LinUCB
        # Use moderate initial regularization for stability
        self.A = [np.eye(dim) * 0.1 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]

        # Forgetting factor: start moderate, adapt based on perceived non-stationarity
        self.base_gamma = 0.99
        self.gamma = 0.99

        # Exploration parameter
        self.alpha = 1.2

        # Tracking for non-stationarity detection
        self.reward_history = [[] for _ in range(n_arms)]
        self.max_history_len = 30

        # Per-arm EMA of rewards to detect shifts
        self.reward_ema = [0.0 for _ in range(n_arms)]
        self.reward_ema_alpha = 0.4

        # Per-arm reward change tracking for drift detection
        self.prev_reward = [None for _ in range(n_arms)]
        self.reward_change_ema = [0.0 for _ in range(n_arms)]

        # Time tracking
        self.t = 0
        self.start_time = time.time()
        self.deadline = 145.0  # Safety margin before 150s limit

    def _is_time_up(self):
        """Check if we're approaching the deadline."""
        return (time.time() - self.start_time) > self.deadline

    def select(self, context):
        self.t += 1

        # Deadline guard: if we're close to timeout, use greedy with best known arm
        if self._is_time_up():
            # Use the arm with highest recent average reward
            best_arm = 0
            best_mean = -np.inf
            for a in range(self.n):
                if self.reward_history[a]:
                    mean_r = np.mean(self.reward_history[a][-10:])
                    if mean_r > best_mean:
                        best_mean = mean_r
                        best_arm = a
            return int(best_arm)

        # Adaptive forgetting: track volatility per arm to detect drift
        recent_rewards = []
        for a in range(self.n):
            if self.reward_history[a]:
                recent_rewards.extend(self.reward_history[a][-5:])

        if len(recent_rewards) > 10:
            vol = np.std(recent_rewards)
            # Higher volatility -> lower gamma (faster forgetting)
            gamma_target = 0.999 - 0.049 * min(vol / 1.0, 1.0)
            gamma_target = max(0.95, min(0.999, gamma_target))
            # Smoothly adapt gamma
            self.gamma = 0.9 * self.gamma + 0.1 * gamma_target
        else:
            self.gamma = self.base_gamma

        context = np.asarray(context, dtype=np.float64)

        scores = np.zeros(self.n)
        for a in range(self.n):
            # Use solve for numerical stability and efficiency
            A = self.A[a] + np.eye(self.d) * 1e-6
            try:
                # Solve A @ theta = b instead of computing inverse
                theta = np.linalg.solve(A, self.b[a])
                # For uncertainty, we need x^T A^{-1} x
                # Solve A @ y = x, then uncertainty = sqrt(x^T y)
                y = np.linalg.solve(A, context)
                unc_sq = context @ y
                unc = np.sqrt(max(0.0, unc_sq))
            except np.linalg.LinAlgError:
                theta = np.zeros(self.d)
                unc = 0.0

            mean = theta @ context
            scores[a] = mean + self.alpha * unc

        # Clip to reasonable range to avoid extreme values
        scores = np.clip(scores, -10.0, 10.0)

        # Small random tie-breaking for stability
        if np.allclose(scores, scores[0]):
            return int(np.random.randint(self.n))

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)

        # Update per-arm reward change tracking
        if self.prev_reward[arm] is not None:
            change = abs(reward - self.prev_reward[arm])
            self.reward_change_ema[arm] = 0.3 * change + 0.7 * self.reward_change_ema[arm]
        self.prev_reward[arm] = reward

        # Update EMA
        if self.reward_history[arm]:
            self.reward_ema[arm] = self.reward_ema_alpha * reward + (1 - self.reward_ema_alpha) * self.reward_ema[arm]
        else:
            self.reward_ema[arm] = reward

        # Apply forgetting and update
        gamma = self.gamma
        self.A[arm] = gamma * self.A[arm] + np.outer(context, context)
        self.b[arm] = gamma * self.b[arm] + reward * context

        # Maintain reward history for volatility detection
        self.reward_history[arm].append(reward)
        if len(self.reward_history[arm]) > self.max_history_len:
            self.reward_history[arm].pop(0)

        # Periodic renormalization to prevent numerical issues
        if self.t % 50 == 0:
            for a in range(self.n):
                # Ensure A stays well-conditioned
                try:
                    eigvals = np.linalg.eigvalsh(self.A[a])
                    min_eig = np.min(eigvals)
                    if min_eig < 1e-3:
                        self.A[a] += np.eye(self.d) * (1e-3 - min_eig + 1e-6)
                except np.linalg.LinAlgError:
                    self.A[a] = np.eye(self.d) * 0.1
# EVOLVE-BLOCK-END
