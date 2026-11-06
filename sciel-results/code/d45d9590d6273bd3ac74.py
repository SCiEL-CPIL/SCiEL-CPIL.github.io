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
        self.start_time = time.time()
        self.deadline = self.start_time + 145.0  # 5s safety margin before 150s

        # Per-arm RLS state
        self.P = [np.eye(dim) / 1.0 for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]

        # Forgetting factor (adaptive)
        self.base_forgetting = 0.99
        self.forgetting = self.base_forgetting

        # Track last play time per arm for exploration bonus
        self.last_played = [0] * n_arms

        # Track recent prediction errors for adaptive forgetting
        self.recent_errors = []
        self.error_window = 50

        # Exploration parameter
        self.alpha = 1.0

    def _check_deadline(self):
        """Check if we're approaching the deadline."""
        return time.time() > self.deadline

    def _update_forgetting(self, prediction_error):
        """Adaptively adjust forgetting factor based on recent prediction errors."""
        self.recent_errors.append(abs(prediction_error))
        if len(self.recent_errors) > self.error_window:
            self.recent_errors.pop(0)

        # If recent errors are large, decrease forgetting (forget faster)
        if len(self.recent_errors) >= 10:
            mean_error = np.mean(self.recent_errors)
            # Scale forgetting factor: higher error -> lower forgetting (faster adaptation)
            self.forgetting = max(0.95, min(0.999, self.base_forgetting - 0.01 * mean_error))
        else:
            self.forgetting = self.base_forgetting

    def select(self, context):
        """Select arm using UCB with time-aware exploration bonus."""
        if self._check_deadline():
            # Return best arm based on current estimates with minimal exploration
            scores = []
            for a in range(self.n):
                scores.append(self.theta[a] @ context)
            return int(np.argmax(scores))

        context = np.asarray(context, dtype=np.float64)
        scores = []
        for a in range(self.n):
            # Predicted reward
            mu = self.theta[a] @ context

            # Uncertainty estimate using P matrix
            # Var = context^T P context
            var = context @ self.P[a] @ context
            std = np.sqrt(max(var, 1e-10))

            # Time-aware exploration bonus: increase if arm hasn't been played recently
            time_since_played = self.t - self.last_played[a]
            time_bonus = 0.1 * np.log1p(time_since_played)

            # UCB score
            score = mu + self.alpha * std + time_bonus
            scores.append(score)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        """Update RLS estimate for the played arm."""
        if self._check_deadline():
            return

        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)

        # Compute prediction error
        prediction = self.theta[arm] @ context
        error = reward - prediction

        # Update adaptive forgetting factor
        self._update_forgetting(error)

        # RLS update with forgetting
        # P_new = (1/forgetting) * P - (1/forgetting) * P x (x^T P x + 1/forgetting)^-1 x^T P
        # theta_new = theta + P_new * x * error

        P = self.P[arm]
        theta = self.theta[arm]
        lam = self.forgetting

        # Compute gain
        Px = P @ context
        denom = 1.0 / lam + context @ Px
        gain = Px / denom

        # Update theta
        self.theta[arm] = theta + gain * error

        # Update P
        self.P[arm] = (P - np.outer(gain, context)) / lam

        # Update last played time
        self.last_played[arm] = self.t
        self.t += 1

        # Decay alpha slowly to reduce exploration over time
        self.alpha = max(0.5, 1.0 - 0.001 * self.t)
# EVOLVE-BLOCK-END
