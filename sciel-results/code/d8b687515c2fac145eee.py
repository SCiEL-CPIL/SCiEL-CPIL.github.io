"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

class Policy:
    """
    Adaptive LinUCB with per-arm adaptive forgetting and drift detection.
    
    Combines:
    1. Per-arm lambda adaptation based on prediction error (from current best).
    2. Drift detection to trigger aggressive forgetting (from inspiration 1).
    3. Adaptive alpha decay and per-arm exploration bonus (from inspiration 1).
    """

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.dim = dim
        
        # Initialization of matrices with moderate regularization
        self.A = [np.eye(dim) * 0.1 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Per-arm adaptive forgetting parameters
        # Start with high memory (lambda close to 1)
        self.lambdas = [0.98 for _ in range(n_arms)]
        self.lambda_min = 0.85
        self.lambda_max = 0.99
        self.lambda_base = 0.98
        
        # Tracking for error and drift
        # Store recent squared errors to compute average error
        self.error_hist = [deque(maxlen=20) for _ in range(n_arms)]
        # Store recent rewards for drift detection (mean comparison)
        self.reward_hist = [deque(maxlen=10) for _ in range(n_arms)]
        
        # Exploration parameters
        self.base_alpha = 1.0
        self.t = 0
        
        # Counters for adaptive alpha
        self.count = [0] * n_arms
        self.total_reward = [0.0] * n_arms
        
        # Time management
        self.start_time = time.time()
        self.deadline = self.start_time + 130.0  # Safety margin before 150s kill

    def _solve(self, A, b):
        """Solve A x = b efficiently with fallback."""
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, b, rcond=None)[0]

    def _update_adaptive_params(self, arm, residual_sq, reward):
        """
        Update the forgetting factor for the specified arm based on 
        prediction error and drift detection.
        """
        # 1. Update error history
        self.error_hist[arm].append(residual_sq)
        
        # 2. Update reward history for drift detection
        self.reward_hist[arm].append(reward)
        
        # 3. Detect Drift
        # If we have enough history, check if current reward deviates significantly from recent average
        is_drift = False
        if len(self.reward_hist[arm]) >= 3:
            recent_rewards = list(self.reward_hist[arm])
            # Compare last reward to average of previous ones
            prev_avg = np.mean(recent_rewards[:-1])
            current_reward = recent_rewards[-1]
            # Simple heuristic: deviation > 0.5 * (1 + abs(mean))
            if abs(current_reward - prev_avg) > 0.5 * (1.0 + abs(prev_avg)):
                is_drift = True
        
        # 4. Determine new Lambda
        avg_err_sq = np.mean(self.error_hist[arm]) if self.error_hist[arm] else 0.0
        
        # Base lambda adjustment based on error magnitude
        # Higher error -> lower lambda (more forgetting)
        # Formula: lambda = base / (1 + k * err)
        # We tune 'k' (sensitivity) implicitly by clamping
        new_lambda = self.lambda_base / (1.0 + 1.0 * avg_err_sq)
        
        # If drift is detected, force a more aggressive forgetting
        if is_drift:
            new_lambda = min(new_lambda, 0.90)
            
        # Clamp to valid range
        self.lambdas[arm] = max(self.lambda_min, min(self.lambda_max, new_lambda))

    def _select_arm(self, context):
        """Select arm using LinUCB with per-arm adaptive parameters."""
        context = np.asarray(context, dtype=np.float64)
        self.t += 1

        # Adaptive global alpha: decay over time to reduce exploration
        alpha_global = self.base_alpha * max(0.5, 1.0 - 0.3 * min(1.0, self.t / 300.0))

        scores = []
        for a in range(self.n):
            A = self.A[a]
            b = self.b[a]
            
            # Solve for theta
            theta = self._solve(A, b)
            mu = theta @ context
            
            # Compute variance term: context^T A^{-1} context
            try:
                A_inv_context = np.linalg.solve(A, context)
                v = context @ A_inv_context
            except np.linalg.LinAlgError:
                v = 1.0
            
            # Per-arm adaptive alpha: boost exploration for rarely pulled arms
            alpha_arm = alpha_global * (1.0 + 0.5 / (1.0 + self.count[a]))
            
            scores.append(mu + alpha_arm * np.sqrt(max(v, 0.0)))

        best_arm = int(np.argmax(scores))
        return best_arm

    def select(self, context):
        # Deadline guard
        if time.time() > self.deadline:
            # Fallback: return arm with highest historical average reward
            if any(self.count[a] > 0 for a in range(self.n)):
                avgs = [self.total_reward[a] / max(self.count[a], 1) for a in range(self.n)]
                return int(np.argmax(avgs))
            return 0

        return self._select_arm(context)

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        self.t += 1

        # 1. Compute prediction to get residual
        A = self.A[arm]
        b = self.b[arm]
        theta = self._solve(A, b)
        mu_pred = theta @ context
        residual = reward - mu_pred
        residual_sq = residual ** 2

        # 2. Update adaptive forgetting factor
        self._update_adaptive_params(arm, residual_sq, reward)

        # 3. Update A and b for the pulled arm using the new lambda
        lam = self.lambdas[arm]
        self.A[arm] = lam * self.A[arm] + np.outer(context, context)
        self.b[arm] = lam * self.b[arm] + reward * context

        # 4. Update counters for adaptive alpha
        self.count[arm] += 1
        self.total_reward[arm] = lam * self.total_reward[arm] + reward


# EVOLVE-BLOCK-END
