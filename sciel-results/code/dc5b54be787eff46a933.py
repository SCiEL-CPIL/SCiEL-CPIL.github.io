"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """LinUCB with Bayesian Online Changepoint Detection (BOCPD) for non-stationary linear bandits."""

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        # Maintain A^{-1} directly for O(d^2) updates instead of O(d^3) inversions
        self.A_inv = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # BOCPD parameters
        self.max_age = 200  # Maximum age to track
        self.hazard_rate = 0.02  # Base probability of change per step
        self.age_posteriors = [np.zeros(self.max_age + 1) for _ in range(n_arms)]
        for a in range(n_arms):
            self.age_posteriors[a][0] = 1.0  # Initially all arms at age 0
        
        # Track prediction errors for changepoint detection
        self.pred_errors = [np.zeros(self.max_age + 1) for _ in range(n_arms)]
        self.error_variance = [1.0 for _ in range(n_arms)]
        
        # Exploration parameter
        self.C = 0.5
        self.round_count = 0
        
        # Deadline guard: track start time
        self._start_time = time.time()
        self._deadline = 120.0  # seconds
        self._last_update_time = time.time()

    def _reset_arm(self, arm):
        """Reset the linear regression state for an arm."""
        self.A_inv[arm] = np.eye(self.d)
        self.b[arm] = np.zeros(self.d)
        self.pred_errors[arm] = np.zeros(self.max_age + 1)
        self.error_variance[arm] = 1.0

    def select(self, context):
        # Deadline guard: if we're getting close to the limit, be conservative
        elapsed = time.time() - self._start_time
        if elapsed > self._deadline - 5.0:
            # Return a simple heuristic to avoid computation overhead
            return int(np.argmax(self.b) @ context)
        
        scores = []
        for a in range(self.n):
            inv = self.A_inv[a]
            theta = inv @ self.b[a]
            scores.append(theta @ context + self.C * np.sqrt(context @ inv @ context))
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.round_count += 1
        
        # Check deadline before doing expensive work
        if time.time() - self._start_time > self._deadline - 2.0:
            return
        
        # Get current age posterior for this arm
        posterior = self.age_posteriors[arm].copy()
        max_age = self.max_age
        
        # Compute prediction for current state
        inv = self.A_inv[arm]
        theta = inv @ self.b[arm]
        pred = theta @ context
        residual = reward - pred
        
        # Update prediction errors for each possible age
        for age in range(1, max_age + 1):
            if posterior[age - 1] > 1e-10:  # Only compute if this age has non-negligible probability
                self.pred_errors[arm][age] = residual
        
        # Compute likelihood of observing this reward given each age
        # Use a Gaussian likelihood with variance estimated from recent errors
        likelihoods = np.zeros(max_age + 1)
        for age in range(1, max_age + 1):
            if posterior[age - 1] > 1e-10:
                # Estimate variance from errors at this age
                errors_at_age = self.pred_errors[arm][age]
                variance = self.error_variance[arm] + 0.1  # Add small regularization
                # Gaussian likelihood
                likelihoods[age] = np.exp(-0.5 * (errors_at_age ** 2) / variance) / np.sqrt(2 * np.pi * variance)
        
        # Normalize likelihoods
        total_likelihood = likelihoods.sum()
        if total_likelihood < 1e-300:
            likelihoods = np.ones(max_age + 1) / (max_age + 1)
        else:
            likelihoods /= total_likelihood
        
        # Predictive update: predict age before observing
        # P(age_t | age_{t-1}) with hazard rate
        predicted = np.zeros(max_age + 1)
        for age in range(1, max_age + 1):
            # Either stay at same age (no change) or change to new age
            predicted[age] = posterior[age] * (1 - self.hazard_rate)
            if age > 1:
                predicted[age] += posterior[age - 1] * self.hazard_rate
        
        # Bayesian update
        updated = predicted * likelihoods
        updated_sum = updated.sum()
        if updated_sum < 1e-300:
            updated = np.ones(max_age + 1) / (max_age + 1)
        else:
            updated /= updated_sum
        
        self.age_posteriors[arm] = updated
        
        # Detect changepoint: if probability of new regime (age 1) is high, reset
        changepoint_prob = updated[1]
        if changepoint_prob > 0.3:
            # Reset the arm's state
            self._reset_arm(arm)
            # Reset age posterior to age 1
            self.age_posteriors[arm] = np.zeros(self.max_age + 1)
            self.age_posteriors[arm][1] = 1.0
        
        # Update linear regression with forgetting based on age
        # Older regimes get more aggressive forgetting
        age_weight = np.sum(updated * np.arange(1, max_age + 2))
        # Map age to forgetting factor: younger = less forgetting, older = more forgetting
        forgetting_factor = max(0.85, 1.0 - 0.005 * age_weight)
        
        # Update b with forgetting
        self.b[arm] = forgetting_factor * self.b[arm] + reward * context
        
        # Update A^{-1} using Sherman-Morrison with forgetting
        V = self.A_inv[arm]
        g = forgetting_factor
        x = context
        Vx = V @ x
        denom = (1.0 / g) + x @ Vx
        self.A_inv[arm] = (1.0 / g) * V - (1.0 / (g * g)) * np.outer(Vx, Vx) / denom
        
        # Symmetrize to avoid numerical drift
        self.A_inv[arm] = 0.5 * (self.A_inv[arm] + self.A_inv[arm].T)
        
        # Update error variance estimate
        self.error_variance[arm] = 0.95 * self.error_variance[arm] + 0.05 * (residual ** 2)
# EVOLVE-BLOCK-END
