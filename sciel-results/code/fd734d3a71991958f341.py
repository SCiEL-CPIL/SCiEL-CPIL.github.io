"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """LinUCB with adaptive forgetting via Cholesky and EWMA non-stationarity detection."""

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0  # Safety margin before 120s limit

        # Initialize A and b for each arm
        self.A = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]

        # Cholesky factors for numerical stability
        self.L = [np.eye(dim) for _ in range(n_arms)]
        self.L_valid = [True] * n_arms

        # Adaptive forgetting: base and EWMA-based adjustment
        self.base_forgetting = 0.995
        self.min_forgetting = 0.95
        self.max_forgetting = 0.999

        # EWMA for non-stationarity detection
        self.reward_ewma = [0.0] * n_arms
        self.reward_var_ewma = [1.0] * n_arms
        self.ewma_alpha = 0.1  # Fast tracking

        # Track recent rewards for variance estimation
        self.recent_rewards = [[] for _ in range(n_arms)]
        self.max_recent = 20

        # Exploration parameter
        self.alpha = 1.0

        # Per-arm visit counts
        self.visits = [0] * n_arms

    def _update_cholesky(self, arm):
        """Update Cholesky factor for the given arm."""
        try:
            self.L[arm] = np.linalg.cholesky(self.A[arm])
            self.L_valid[arm] = True
        except np.linalg.LinAlgError:
            # If Cholesky fails, add small regularization
            self.A[arm] += 1e-6 * np.eye(self.d)
            try:
                self.L[arm] = np.linalg.cholesky(self.A[arm])
                self.L_valid[arm] = True
            except np.linalg.LinAlgError:
                self.L_valid[arm] = False

    def _compute_forgetting(self, arm):
        """Compute adaptive forgetting factor based on EWMA variance."""
        if self.t < 10:
            return self.base_forgetting

        # Estimate variance of recent rewards for this arm
        recent = self.recent_rewards[arm]
        if len(recent) >= 5:
            var = np.var(recent)
            # Higher variance suggests more non-stationarity -> lower forgetting (faster adaptation)
            # Map variance to forgetting factor
            if var > 0.5:
                return self.min_forgetting
            elif var > 0.1:
                return 0.97
            elif var > 0.01:
                return 0.99
            else:
                return self.max_forgetting
        else:
            return self.base_forgetting

    def select(self, context):
        # Check deadline
        if time.time() > self.deadline:
            # Return a safe default arm
            return 0

        scores = []
        for a in range(self.n):
            if self.L_valid[a]:
                # Use Cholesky for stable solve
                L = self.L[a]
                # Solve L theta = b
                try:
                    theta = np.linalg.solve(L.T, np.linalg.solve(L, self.b[a]))
                except np.linalg.LinAlgError:
                    theta = np.linalg.inv(self.A[a]) @ self.b[a]
                # Variance: context^T A^{-1} context = ||L^{-1} context||^2
                try:
                    v = np.linalg.solve(L, context)
                    var = np.dot(v, v)
                except np.linalg.LinAlgError:
                    inv = np.linalg.inv(self.A[a])
                    var = context @ inv @ context
            else:
                inv = np.linalg.inv(self.A[a])
                theta = inv @ self.b[a]
                var = context @ inv @ context

            # Adaptive alpha based on visit count
            visit_bonus = 1.0 / (1.0 + np.sqrt(self.visits[a]))
            scores.append(theta @ context + self.alpha * visit_bonus * np.sqrt(max(var, 0)))

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.t += 1

        # Update visit count
        self.visits[arm] += 1

        # Update recent rewards
        self.recent_rewards[arm].append(reward)
        if len(self.recent_rewards[arm]) > self.max_recent:
            self.recent_rewards[arm].pop(0)

        # Update EWMA
        self.reward_ewma[arm] = (1 - self.ewma_alpha) * self.reward_ewma[arm] + self.ewma_alpha * reward
        diff = reward - self.reward_ewma[arm]
        self.reward_var_ewma[arm] = (1 - self.ewma_alpha) * self.reward_var_ewma[arm] + self.ewma_alpha * (diff ** 2)

        # Compute adaptive forgetting factor
        forgetting = self._compute_forgetting(arm)

        # Update A and b with forgetting
        self.A[arm] = forgetting * self.A[arm] + np.outer(context, context)
        self.b[arm] = forgetting * self.b[arm] + reward * context

        # Update Cholesky factor
        self._update_cholesky(arm)

        # Adapt alpha over time
        self.alpha = 1.0 / (1.0 + 0.01 * self.t)
# EVOLVE-BLOCK-END
