"""Bayesian Linear Bandit with Adaptive Residual-Based Forgetting."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Thompson Sampling for non-stationary linear bandits.
    
    Uses a Bayesian framework where the posterior over weights is Gaussian.
    Adaptive forgetting is triggered when prediction residuals are large,
    indicating a potential regime shift in the underlying weights.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t0 = time.time()
        self.deadline = 145.0  # Safety margin before 150s kill
        
        # Posterior parameters: A[a] = V^{-1}, b[a] = V^{-1} * mu
        # We maintain A and b directly, and use Sherman-Morrison for updates
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # For adaptive forgetting
        self.residuals = [[] for _ in range(n_arms)]
        self.forget_factor = 0.99
        self.residual_threshold = 2.0
        self.window_size = 20
        
        # Track last prediction for each arm
        self.last_pred = [None] * n_arms

    def _mahalanobis(self, arm, context):
        """Compute context @ A[arm] @ context (the variance term)."""
        return context @ self.A[arm] @ context

    def select(self, context):
        # Check deadline
        if time.time() - self.t0 > self.deadline:
            # Fallback: pick the arm with highest mean estimate
            means = [self.b[a] @ context for a in range(self.n)]
            return int(np.argmax(means))
        
        scores = []
        for a in range(self.n):
            theta = self.A[a] @ self.b[a]
            mean = theta @ context
            variance = self._mahalanobis(a, context)
            # Thompson Sampling: sample from N(mean, variance)
            sample = mean + np.sqrt(max(variance, 1e-8)) * np.random.randn()
            scores.append(sample)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Check deadline
        if time.time() - self.t0 > self.deadline:
            return
        
        # Compute residual
        theta = self.A[arm] @ self.b[arm]
        pred = theta @ context
        residual = reward - pred
        
        # Track residuals for adaptive forgetting
        self.residuals[arm].append(residual)
        if len(self.residuals[arm]) > self.window_size:
            self.residuals[arm].pop(0)
        
        # Check if we should forget more aggressively
        if len(self.residuals[arm]) >= 5:
            recent_res = np.array(self.residuals[arm][-5:])
            if np.std(recent_res) > self.residual_threshold:
                # Increase forgetting (decrease forget_factor toward 0.9)
                self.forget_factor = max(0.9, self.forget_factor - 0.005)
            else:
                # Decrease forgetting (increase forget_factor toward 0.995)
                self.forget_factor = min(0.995, self.forget_factor + 0.002)
        
        # Update posterior using Sherman-Morrison-Woodbury
        # A_new = (A_old - outer(x,x) * A_old @ (I + x @ A_old @ x)^{-1} @ A_old)^{-1}
        # More efficiently: use the standard Bayesian linear regression update
        
        x = context
        A_old = self.A[arm]
        b_old = self.b[arm]
        
        # Compute a = A_old @ x
        a = A_old @ x
        # Compute the scalar denominator
        denom = 1.0 + x @ a
        # Update A: A_new = A_old - outer(a, a) / denom
        self.A[arm] = A_old - np.outer(a, a) / denom
        # Update b: b_new = b_old + a * reward / denom
        self.b[arm] = b_old + a * reward / denom
        
        # Apply adaptive forgetting: shrink A and b toward prior
        # This effectively increases the prior weight
        self.A[arm] = self.forget_factor * self.A[arm] + (1 - self.forget_factor) * np.eye(self.d)
        self.b[arm] = self.forget_factor * self.b[arm]
        
        # Store prediction for next residual computation
        self.last_pred[arm] = pred
# EVOLVE-BLOCK-END
