"""
Non-stationary linear contextual bandit policy using Recursive Least Squares (RLS).
Structural change: RLS maintains explicit inverse covariance (P) for O(d^2) updates
with a time-varying forgetting factor modulated by squared prediction error.
This allows rapid re-initialization during regime changes and stable convergence
otherwise, outperforming fixed or heuristically bounded forgetting.
"""

import numpy as np
import time


class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim

        # RLS: P is the inverse covariance (precision) matrix per arm
        # theta is the weight vector per arm
        # Initialize P to large identity (high uncertainty) and theta to zero
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]

        # Forgetting factor parameters
        self.lam_base = 0.97
        self.lam_min = 0.85   # Fast forgetting during high error
        self.lam_max = 0.995  # Slow forgetting during low error

        # Error tracking for adaptive forgetting
        self.sq_errors = [0.0] * n_arms  # EMA of squared errors per arm
        self.sq_error_alpha = 0.3        # EMA decay for error tracking

        # Exploration: UCB with adaptive alpha
        self.alpha_base = 1.5
        self.arm_counts = [0] * n_arms
        self.total_rounds = 0

        # Deadline guard
        self._start_time = time.time()
        self._deadline = self._start_time + 140.0

    def _get_lambda(self, arm):
        """
        Compute adaptive forgetting factor based on recent squared prediction error.
        High error -> low lambda (fast forgetting, rapid re-initialization)
        Low error  -> high lambda (slow forgetting, stable convergence)
        """
        err = self.sq_errors[arm]
        # Normalize: err_scale controls sensitivity
        err_scale = 0.1
        normalized = err / err_scale

        # Smooth sigmoid-like mapping from normalized error to [lam_min, lam_max]
        # When normalized is small -> lam_max (slow forgetting)
        # When normalized is large -> lam_min (fast forgetting)
        # Use a smooth inverse-sigmoid
        t = 1.0 / (1.0 + np.exp(-10.0 * (normalized - 1.0)))
        # t is 0 when normalized << 1, 1 when normalized >> 1
        # We want high error -> low lambda, so:
        lam = self.lam_max - t * (self.lam_max - self.lam_min)
        return np.clip(lam, self.lam_min, self.lam_max)

    def _get_alpha(self):
        """
        Adaptive exploration parameter.
        """
        if self.total_rounds < 30:
            return self.alpha_base + 1.0  # More exploration early

        # Decrease exploration over time
        decay = 1.0 / (1.0 + self.total_rounds / 200.0)
        return self.alpha_base * decay + 0.5

    def select(self, context):
        # Deadline check
        if time.time() > self._deadline:
            return int(np.random.randint(self.n))

        context = np.asarray(context, dtype=np.float64)
        alpha = self._get_alpha()
        scores = []

        for a in range(self.n):
            # Mean prediction
            mu = self.theta[a] @ context

            # Uncertainty: sqrt(context^T P context)
            p_context = self.P[a] @ context
            var = context @ p_context
            std = np.sqrt(max(0.0, var))

            # UCB score
            score = mu + alpha * std
            scores.append(score)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)

        # Deadline check
        if time.time() > self._deadline:
            return

        self.total_rounds += 1
        self.arm_counts[arm] += 1

        # Compute prediction error
        pred = self.theta[arm] @ context
        error = reward - pred
        sq_err = error * error

        # Update EMA of squared error for this arm
        self.sq_errors[arm] = (1.0 - self.sq_error_alpha) * self.sq_errors[arm] + self.sq_error_alpha * sq_err

        # Get adaptive forgetting factor
        lam = self._get_lambda(arm)

        # RLS update
        # K = P x / (lam + x^T P x)
        P = self.P[arm]
        theta = self.theta[arm]

        Px = P @ context
        denom = lam + context @ Px
        if denom < 1e-12:
            denom = 1e-12

        K = Px / denom

        # Update theta
        self.theta[arm] = theta + K * error

        # Update P: P = (P - K x^T P) / lam
        # Using matrix identity: P_new = (1/lam)(P - K x^T P)
        self.P[arm] = (P - np.outer(K, context) @ P) / lam

        # Ensure P remains symmetric positive definite (numerical stability)
        P_new = self.P[arm]
        P_new = (P_new + P_new.T) * 0.5  # Symmetrize

        # Add small regularization to prevent singularity
        P_new += 1e-6 * np.eye(self.d)

        self.P[arm] = P_new
