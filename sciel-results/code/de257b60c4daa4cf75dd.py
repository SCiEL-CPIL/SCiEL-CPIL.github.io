import numpy as np
import time
from collections import deque
import scipy.linalg

class Policy:
    """
    Adaptive LinUCB for non-stationary linear contextual bandits.
    Combines:
    - Structural idea: Per-arm adaptive forgetting factor (lambda) based on local MSE.
    - Refinement machinery: Global adaptive alpha (exploration) based on global MSE,
      and a strict deadline guard for robustness.
    """
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Cached inverses
        self.inv_A = [np.eye(dim) for _ in range(n_arms)]
        
        # Adaptive Forgetting Parameters
        self.lambda_min = 0.90
        self.lambda_max = 0.995
        self.error_windows = [deque(maxlen=30) for _ in range(n_arms)]
        
        # Exploration
        self.alpha_base = 1.0
        self.alpha_min = 0.5
        self.alpha_max = 2.0
        self.alpha = self.alpha_base
        
        # Timing
        self.start_time = time.time()
        self.deadline = 110.0  # Safety margin before 120s limit
        
        # Global step
        self.t = 0
        
        # Global error tracking for alpha adaptation
        self.global_errors = deque(maxlen=100)

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def _recompute_inv(self, arm):
        """Recompute inverse of A[arm] and cache it."""
        try:
            self.inv_A[arm] = scipy.linalg.inv(self.A[arm])
        except np.linalg.LinAlgError:
            self.inv_A[arm] = np.eye(self.d)

    def _get_lambda(self, arm):
        """Determine forgetting factor based on recent errors for this arm."""
        if len(self.error_windows[arm]) == 0:
            return self.lambda_max
        
        errors = np.array(self.error_windows[arm])
        mse = np.mean(errors**2)
        
        # If MSE is small, we are stable, so forget slowly (lambda close to 1)
        # If MSE is large, we are unstable, so forget fast (lambda lower)
        
        # Scale factor: assume typical reward variance is around 1.0
        # Using a saturating function to map MSE to lambda
        # lambda = lambda_max - (lambda_max - lambda_min) * (mse / (mse + scale))
        scale = 0.1
        norm_mse = mse / (mse + scale)
        lam = self.lambda_max - norm_mse * (self.lambda_max - self.lambda_min)
        
        return max(self.lambda_min, min(self.lambda_max, lam))

    def _update_alpha(self):
        """Dynamically adjust alpha based on global performance."""
        if len(self.global_errors) < 10:
            return
        
        errors = np.array(self.global_errors)
        mse = np.mean(errors**2)
        
        # If global MSE is high, we might be exploring too little or model is bad.
        # If global MSE is low, we can reduce exploration to exploit.
        
        # Target alpha:
        # High MSE -> Higher Alpha (more exploration to find better model)
        # Low MSE -> Lower Alpha (exploit current model)
        
        # Map MSE to alpha
        # Assume MSE range [0, 2]
        # alpha = alpha_min + (alpha_max - alpha_min) * (mse / (mse + 0.5))
        
        # However, if MSE is very low, we want alpha to be small.
        # If MSE is high, we want alpha to be larger.
        
        # Let's use a simpler heuristic:
        # If recent errors are decreasing, keep alpha or decrease slightly.
        # If recent errors are increasing, increase alpha.
        
        # For simplicity and stability, let's modulate alpha based on the magnitude of recent errors.
        # alpha = alpha_base * (1.0 + 0.5 * min(mse, 1.0))
        # This allows alpha to go up to 1.5x base if MSE is high.
        
        # But we also want to decay alpha over time if performance is good.
        # Let's stick to error-based modulation for now.
        
        target_alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * (mse / (mse + 0.2))
        
        # Smoothly move towards target
        self.alpha = 0.9 * self.alpha + 0.1 * target_alpha
        self.alpha = max(self.alpha_min, min(self.alpha_max, self.alpha))

    def select(self, context):
        if not self._check_deadline():
            # If we are near the deadline, just pick the arm with highest mean prediction
            # to avoid expensive computations or just pick a random one?
            # Better to pick the best known arm.
            scores = []
            for a in range(self.n):
                inv_A = self.inv_A[a]
                theta = inv_A @ self.b[a]
                mu = theta @ context
                scores.append(mu)
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            inv_A = self.inv_A[a]
            theta = inv_A @ self.b[a]
            mu = theta @ context
            var = context @ inv_A @ context
            bonus = self.alpha * np.sqrt(var)
            scores.append(mu + bonus)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        if not self._check_deadline():
            return

        self.t += 1
        
        # 1. Calculate residual using the cached inverse (model before this update)
        inv_A_old = self.inv_A[arm]
        theta_old = inv_A_old @ self.b[arm]
        pred = theta_old @ context
        residual = reward - pred
        
        # Store residual
        self.error_windows[arm].append(residual)
        self.global_errors.append(residual)
        
        # 2. Determine current forgetting factor
        lam = self._get_lambda(arm)
        
        # 3. Update matrices
        self.A[arm] = lam * self.A[arm] + np.outer(context, context)
        self.b[arm] = lam * self.b[arm] + reward * context
        
        # 4. Recompute inverse
        self._recompute_inv(arm)
        
        # 5. Update alpha
        self._update_alpha()
