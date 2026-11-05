# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear bandit with Bayesian Online Changepoint Detection (BOCPD).
    
    For each arm, we maintain:
      - A linear regression model (A, b) for the current regime
      - A hazard model for regime changes (piecewise constant)
      - A posterior over the age of the current regime (hazard-based)
    
    When a regime change is likely, we reset the linear model for that arm.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self._t0 = time.time()
        self._deadline = 150.0  # seconds

        # Per-arm state
        self.A = [np.eye(dim) * 1e-3 for _ in range(n_arms)]  # regularized
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.age = [0 for _ in range(n_arms)]  # current regime age
        self.hazard = [0.05 for _ in range(n_arms)]  # base change probability
        self.n_observations = [0 for _ in range(n_arms)]
        
        # BOCPD: posterior over age (truncated to max_age for efficiency)
        self.max_age = 100
        self.age_posterior = [np.ones(self.max_age + 1) / (self.max_age + 1) for _ in range(n_arms)]
        
        # Track prediction errors for adaptive hazard
        self.pred_err = [0.0 for _ in range(n_arms)]
        self.err_count = [0 for _ in range(n_arms)]
        
        # Exploration bonus
        self.beta = 1.0

    def _update_age_posterior(self, arm):
        """
        Update the posterior over the age of the current regime for a given arm.
        Uses a simple hazard model: P(change at next step | age) = hazard * (1 + age * 0.01)
        """
        h = self.hazard[arm]
        post = self.age_posterior[arm].copy()
        new_post = np.zeros(self.max_age + 1)
        
        # No change: age increases by 1
        for a in range(self.max_age):
            new_post[a + 1] += post[a] * (1 - h * (1 + a * 0.01))
        
        # Change: age resets to 0
        change_prob = sum(post[a] * h * (1 + a * 0.01) for a in range(self.max_age + 1))
        new_post[0] += change_prob
        
        # Normalize
        total = new_post.sum()
        if total > 1e-12:
            new_post /= total
        else:
            new_post = np.ones(self.max_age + 1) / (self.max_age + 1)
        
        self.age_posterior[arm] = new_post

    def _should_reset(self, arm):
        """
        Decide whether to reset the linear model based on the posterior probability
        that a change has occurred recently.
        """
        # Probability that a change occurred in the last few steps
        recent_change_prob = self.age_posterior[arm][0] + 0.5 * self.age_posterior[arm][1] + 0.25 * self.age_posterior[arm][2]
        
        # Also consider if the model is becoming stale (high age)
        expected_age = sum(a * self.age_posterior[arm][a] for a in range(self.max_age + 1))
        
        # Reset if recent change is likely or model is very stale
        return recent_change_prob > 0.3 or expected_age > 50

    def select(self, context):
        # Deadline guard
        if time.time() - self._t0 > self._deadline - 5.0:
            # Return a reasonable default
            return int(np.argmax([self.b[a] @ context for a in range(self.n)]))

        scores = []
        for a in range(self.n):
            # Compute theta
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            
            # Mean prediction
            mean = theta @ context
            
            # Uncertainty bonus
            var = context @ A_inv @ context
            bonus = self.beta * np.sqrt(var)
            
            scores.append(mean + bonus)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Deadline guard
        if time.time() - self._t0 > self._deadline - 5.0:
            return

        # Update the age posterior for this arm
        self._update_age_posterior(arm)
        
        # Check if we should reset
        if self._should_reset(arm):
            # Reset the linear model
            self.A[arm] = np.eye(self.d) * 1e-3
            self.b[arm] = np.zeros(self.d)
            self.age[arm] = 0
            self.n_observations[arm] = 0
            self.age_posterior[arm] = np.ones(self.max_age + 1) / (self.max_age + 1)
            self.pred_err[arm] = 0.0
            self.err_count[arm] = 0
            return

        # Update the linear model
        self.A[arm] = self.A[arm] + np.outer(context, context)
        self.b[arm] = self.b[arm] + reward * context
        self.age[arm] += 1
        self.n_observations[arm] += 1

        # Track prediction error for adaptive hazard
        A_inv = np.linalg.inv(self.A[arm])
        theta = A_inv @ self.b[arm]
        pred = theta @ context
        err = abs(reward - pred)
        self.pred_err[arm] = 0.9 * self.pred_err[arm] + 0.1 * err
        self.err_count[arm] += 1
        
        # Adapt hazard: increase if errors are large
        if self.err_count[arm] > 10:
            base_hazard = 0.05
            err_factor = min(self.pred_err[arm] / 1.0, 2.0)
            self.hazard[arm] = base_hazard * (1 + err_factor)
# EVOLVE-BLOCK-END
