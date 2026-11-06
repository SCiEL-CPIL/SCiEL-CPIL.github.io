"""Non-stationary linear contextual bandit with adaptive RLS and time-aware exploration."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit using:
    1. Per-arm Recursive Least Squares (RLS) with exponential forgetting
    2. Adaptive forgetting factor based on recent prediction errors
    3. Time-aware exploration bonus that increases when arm hasn't been played
    4. Deadline guard to avoid timeout
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0

        # Per-arm RLS state
        self.P = [np.eye(dim) / 1.0 for _ in range(n_arms)]  # Inverse covariance
        self.theta = [np.zeros(dim) for _ in range(n_arms)]   # Parameter estimates

        # Adaptive forgetting
        self.base_forgetting = 0.98
        self.min_forgetting = 0.95
        self.max_forgetting = 0.999
        self.error_window = 50
        self.recent_errors = [np.zeros(self.error_window) for _ in range(n_arms)]
        self.error_idx = [0] * n_arms

        # Time-aware exploration: track last play time for each arm
        self.last_played = [-1000] * n_arms  # Initialize far in the past

        # Exploration parameters
        self.base_alpha = 1.0
        self.max_alpha = 5.0
        self.time_decay = 0.95  # How quickly exploration bonus decays with time since last play

        # Deadline guard
        self.start_time = time.time()
        self.deadline = 145.0  # Safety margin before 150s

        # Track overall reward trend
        self.reward_history = []
        self.reward_window = 100

    def _check_deadline(self):
        """Return True if we should stop computing and use greedy."""
        return (time.time() - self.start_time) > self.deadline

    def _compute_forgetting_factor(self, arm):
        """Adaptive forgetting factor based on recent prediction errors."""
        if self.t < 10:
            return self.base_forgetting
        
        # Use recent errors to estimate non-stationarity
        n_recent = min(self.t, self.error_window)
        if n_recent == 0:
            return self.base_forgetting
        
        # Get recent errors for this arm
        recent = self.recent_errors[arm]
        start = (self.error_idx[arm] - n_recent) % self.error_window
        if start >= 0:
            errors = recent[start:start + n_recent]
        else:
            errors = np.concatenate([recent[start:], recent[:start + n_recent]])
        
        if len(errors) == 0:
            return self.base_forgetting
        
        mean_err = np.mean(np.abs(errors))
        
        # Higher error -> lower forgetting (faster adaptation)
        # Map mean error to forgetting factor
        if mean_err < 0.1:
            factor = self.max_forgetting
        elif mean_err > 1.0:
            factor = self.min_forgetting
        else:
            # Linear interpolation
            t_norm = (mean_err - 0.1) / 0.9
            factor = self.max_forgetting - t_norm * (self.max_forgetting - self.min_forgetting)
        
        return np.clip(factor, self.min_forgetting, self.max_forgetting)

    def _compute_exploration_bonus(self, arm, context):
        """Compute exploration bonus combining uncertainty and time since last play."""
        # Uncertainty-based bonus
        P = self.P[arm]
        var = context @ P @ context
        uncertainty_bonus = np.sqrt(max(var, 0))
        
        # Time-based bonus: increase if arm hasn't been played recently
        time_since_play = self.t - self.last_played[arm]
        time_bonus = 1.0 / (1.0 + 0.05 * time_since_play)
        
        # Combine: base alpha * uncertainty * time_factor
        alpha = self.base_alpha * (0.5 + 0.5 * time_bonus)
        alpha = min(alpha, self.max_alpha)
        
        return alpha * uncertainty_bonus

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)
        self.t += 1

        # Deadline guard: if close to timeout, use greedy
        if self._check_deadline():
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                score = self.theta[a] @ context
                if score > best_score:
                    best_score = score
                    best_arm = a
            return int(best_arm)

        scores = np.zeros(self.n)
        for a in range(self.n):
            # Predicted reward
            pred = self.theta[a] @ context
            # Exploration bonus
            bonus = self._compute_exploration_bonus(a, context)
            scores[a] = pred + bonus

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        
        # Deadline guard: skip update if close to timeout
        if self._check_deadline():
            return

        # Compute prediction error
        pred = self.theta[arm] @ context
        error = reward - pred

        # Store error for adaptive forgetting
        self.recent_errors[arm][self.error_idx[arm]] = error
        self.error_idx[arm] = (self.error_idx[arm] + 1) % self.error_window

        # Update last played time
        self.last_played[arm] = self.t

        # Adaptive forgetting factor
        lambda_ = self._compute_forgetting_factor(arm)

        # RLS update
        P = self.P[arm]
        theta = self.theta[arm]

        # P = (1/lambda) * P - (P @ x @ (lambda + x^T @ P @ x)^-1 @ x^T @ P) / lambda
        # More stable form:
        Px = P @ context
        denom = lambda_ + context @ Px
        if denom < 1e-10:
            denom = 1e-10

        gain = Px / denom
        theta_new = theta + gain * error
        P_new = (P - np.outer(gain, context)) / lambda_

        # Ensure P remains positive definite (regularize if needed)
        # Simple check: if trace becomes too large, reset
        if np.trace(P_new) > 1e6:
            P_new = np.eye(self.d) / 1.0
            theta_new = np.zeros(self.d)

        self.P[arm] = P_new
        self.theta[arm] = theta_new

        # Track reward history for potential trend detection
        self.reward_history.append(reward)
        if len(self.reward_history) > self.reward_window:
            self.reward_history.pop(0)
# EVOLVE-BLOCK-END
