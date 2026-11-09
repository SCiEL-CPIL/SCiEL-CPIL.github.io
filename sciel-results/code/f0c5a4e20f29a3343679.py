"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

class Policy:
    """Adaptive LinUCB with per-arm adaptive forgetting based on prediction error and drift detection."""

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.dim = dim
        
        # Initialization of matrices
        # We maintain A_inv explicitly for faster UCB calculation and updates
        # Start with a moderate regularization to allow quick adaptation but stable start
        reg = 0.5 
        self.A_inv = [np.eye(dim) / reg for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Per-arm adaptive forgetting parameters
        self.lambdas = [0.98 for _ in range(n_arms)]
        self.lambda_min = 0.70
        self.lambda_max = 0.99
        self.lambda_base = 0.98
        
        # History for error and drift tracking
        self.error_hist = [deque(maxlen=20) for _ in range(n_arms)]
        self.reward_hist = [deque(maxlen=20) for _ in range(n_arms)]
        
        # Exploration parameter
        self.alpha = 1.2
        
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 140.0  # Safety margin

    def _select_arm(self, context):
        """Select arm using LinUCB with per-arm adaptive parameters."""
        context = np.asarray(context, dtype=np.float64)
        self.t += 1

        scores = []
        for a in range(self.n):
            A_inv = self.A_inv[a]
            b = self.b[a]
            
            # theta = A_inv @ b
            theta = A_inv @ b
            mu = theta @ context
            
            # Compute variance term: context^T A_inv context
            # v = context @ A_inv @ context
            v = context @ A_inv @ context
            
            # UCB score
            score = mu + self.alpha * np.sqrt(max(v, 0.0))
            scores.append(score)

        best_arm = int(np.argmax(scores))
        return best_arm

    def select(self, context):
        # Deadline guard
        if time.time() > self.deadline:
            return 0

        arm = self._select_arm(context)
        return arm

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        self.t += 1

        # 1. Compute prediction for the pulled arm to get residual
        A_inv = self.A_inv[arm]
        b = self.b[arm]
        theta = A_inv @ b
        mu_pred = theta @ context
        residual = reward - mu_pred
        residual_sq = residual ** 2

        # 2. Update error history and adaptive forgetting factor
        self.error_hist[arm].append(residual_sq)
        self.reward_hist[arm].append(reward)
        
        # Drift Detection: Check if recent mean reward differs significantly from older mean
        drift_detected = False
        if len(self.reward_hist[arm]) >= 10:
            recent_mean = np.mean(list(self.reward_hist[arm])[-5:])
            older_mean = np.mean(list(self.reward_hist[arm])[:-5])
            # If the change is larger than 0.5 standard deviations of the history or a fixed threshold
            std_reward = np.std(list(self.reward_hist[arm]))
            if abs(recent_mean - older_mean) > 0.5 * std_reward + 0.1:
                drift_detected = True

        if drift_detected:
            # Aggressive forgetting on drift
            new_lambda = self.lambda_min 
        else:
            # Standard adaptive forgetting based on error
            if len(self.error_hist[arm]) > 0:
                avg_err_sq = np.mean(self.error_hist[arm])
                # lambda = base / (1 + k * err)
                new_lambda = self.lambda_base / (1.0 + 1.0 * avg_err_sq)
                self.lambdas[arm] = max(self.lambda_min, min(self.lambda_max, new_lambda))
            else:
                self.lambdas[arm] = self.lambda_base
        
        lam = self.lambdas[arm]

        # 3. Update A_inv and b using Sherman-Morrison-Woodbury with decay
        # A_new = lam * A_old + xx^T
        # A_inv_new = (1/lam) * (A_inv_old - (A_inv_old @ x @ x^T @ A_inv_old) / (lam + x^T @ A_inv_old @ x))
        # Note: The formula for (alpha A + xx^T)^-1 is:
        # 1/alpha * (A^-1 - (A^-1 x x^T A^-1) / (alpha + x^T A^-1 x))
        # Here alpha = lam, A = A_old. So A^-1 is A_inv_old.
        
        x = context
        A_inv_old = self.A_inv[arm]
        
        # Calculate denominator
        A_inv_x = A_inv_old @ x
        denom = lam + x @ A_inv_x
        
        # Update A_inv
        # A_inv_new = (1/lam) * (A_inv_old - (A_inv_x * A_inv_x.T) / denom)
        # To avoid creating large temporary matrices if possible, but d is small so it's fine.
        correction = np.outer(A_inv_x, A_inv_x) / denom
        self.A_inv[arm] = (1.0 / lam) * (A_inv_old - correction)
        
        # Update b
        # b_new = lam * b_old + r * x
        self.b[arm] = lam * self.b[arm] + reward * context

# EVOLVE-BLOCK-END
