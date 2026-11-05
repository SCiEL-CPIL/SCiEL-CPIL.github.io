"""
Rewritten program: Non-stationary linear contextual bandit with Adaptive RLS Thompson Sampling.

Structural Change:
- Switched from Dual-Timescale Ridge Regression to Single-Model Recursive Least Squares (RLS).
- Maintains the inverse covariance matrix P directly for O(d^2) updates and sampling.
- Implements adaptive forgetting factor (lambda) based on prediction error magnitude.
- Uses Thompson Sampling for exploration, leveraging the exact posterior covariance P.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t_start = time.time()
        self.deadline = self.t_start + 115.0

        # RLS Parameters
        # theta: weight vectors
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        # P: inverse covariance matrices (initially large to encourage exploration)
        # Using 100 * I ensures high initial uncertainty
        self.P = [np.eye(dim) * 100.0 for _ in range(n_arms)]
        
        # Adaptive Forgetting
        # lambda is the forgetting factor. 
        # High lambda (close to 1) = slow forgetting (stable).
        # Low lambda (e.g., 0.9) = fast forgetting (responsive).
        self.lambda_base = 0.98
        self.lambda_min = 0.85
        self.lambda_max = 0.999
        self.lambdas = [self.lambda_base] * n_arms
        
        # Error tracking for adaptation
        self.recent_errors = [[] for _ in range(n_arms)]
        self.error_window = 20

        # Pre-compute Cholesky factors for P for fast sampling
        self._L_P = [np.linalg.cholesky(self.P[a]) for a in range(n_arms)]

    def _update_chol_P(self, arm):
        """Update Cholesky factor of P with fallback."""
        try:
            return np.linalg.cholesky(self.P[arm] + 1e-9 * np.eye(self.d))
        except np.linalg.LinAlgError:
            # If P becomes ill-conditioned, re-initialize with high uncertainty
            self.P[arm] = np.eye(self.d) * 10.0
            return np.linalg.cholesky(self.P[arm])

    def select(self, context):
        # Deadline guard
        if time.time() > self.deadline:
            # Fallback to greedy
            context = np.asarray(context, dtype=np.float64)
            scores = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(scores))

        context = np.asarray(context, dtype=np.float64)
        scores = []

        for a in range(self.n):
            try:
                # Thompson Sampling: theta_sample ~ N(theta, P)
                # We have P, so we need L such that L L^T = P
                # theta_sample = theta + L * z
                L = self._L_P[a]
                z = np.random.randn(self.d)
                noise = L @ z
                theta_sample = self.theta[a] + noise
                scores.append(theta_sample @ context)
            except Exception:
                # Fallback to greedy if sampling fails
                scores.append(self.theta[a] @ context)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Deadline guard
        if time.time() > self.deadline:
            return

        context = np.asarray(context, dtype=np.float64)
        
        # 1. Compute Prediction Error
        pred = self.theta[arm] @ context
        error = reward - pred
        
        # 2. Adaptive Forgetting
        # Track recent absolute errors
        self.recent_errors[arm].append(abs(error))
        if len(self.recent_errors[arm]) > self.error_window:
            self.recent_errors[arm].pop(0)
        
        # Heuristic: If recent errors are high, we are likely in a non-stationary regime.
        # Decrease lambda (forget faster). If errors are low, increase lambda (trust history).
        avg_err = np.mean(self.recent_errors[arm])
        
        # Normalize error to adjust lambda
        # If avg_err is large, target lambda_min. If small, target lambda_max.
        # Use a simple mapping. Assume typical reward scale is ~1-10.
        if avg_err > 5.0:
            target_lambda = self.lambda_min
        elif avg_err > 2.0:
            target_lambda = 0.92
        elif avg_err > 1.0:
            target_lambda = 0.96
        else:
            target_lambda = self.lambda_max
            
        # Smoothly adjust lambda
        self.lambdas[arm] += 0.1 * (target_lambda - self.lambdas[arm])
        self.lambdas[arm] = np.clip(self.lambdas[arm], self.lambda_min, self.lambda_max)
        
        lam = self.lambdas[arm]
        P = self.P[arm]
        theta = self.theta[arm]

        # 3. RLS Update
        # P_new = (1/lam) * P_old + x x^T
        # K = P_new x / (1 + x^T P_new x)  <-- This is the Kalman Gain form for RLS
        # Actually, standard RLS update for P:
        # P = (1/lam) * P + x x^T  is NOT quite right for the inverse update.
        # Let's use the standard RLS formula for the inverse covariance P:
        # P_t = (1/lambda) * P_{t-1} + x_t x_t^T
        # Wait, the standard RLS for minimizing sum lambda^t (y - x^T theta)^2
        # leads to:
        # P_t = (1/lambda) P_{t-1} + x_t x_t^T
        # theta_t = theta_{t-1} + P_t x_t (y_t - x_t^T theta_{t-1})
        
        # Let's verify this.
        # If P = A^{-1}, and A_t = lambda A_{t-1} + x x^T
        # P_t = A_t^{-1} = (lambda A_{t-1} + x x^T)^{-1}
        # Using matrix inversion lemma:
        # (lambda A + x x^T)^{-1} = (1/lambda) A^{-1} - (1/lambda) A^{-1} x x^T A^{-1} (1/lambda + x^T A^{-1} x)^{-1}
        # P_t = (1/lambda) P_{t-1} - (1/lambda) P_{t-1} x x^T P_{t-1} (1/lambda + x^T P_{t-1} x)^{-1}
        # This is the correct update for P.
        
        # Let's implement the correct P update.
        P_old = P
        x = context
        
        # Compute scalar denominator
        # x^T P_old x
        P_old_x = P_old @ x
        denom = 1.0 / lam + x @ P_old_x
        
        # Update P
        # P_new = (1/lambda) * P_old - (1/lambda^2) * P_old x x^T P_old / denom
        # Note: The term is (1/lambda) * P_old * x * (x^T * (1/lambda) * P_old) / denom ?
        # Let's re-derive carefully.
        # P_t = (lambda P_{t-1}^{-1} + x x^T)^{-1}
        # Let M = lambda P_{t-1}^{-1}. Then P_t = (M + x x^T)^{-1}
        # (M + x x^T)^{-1} = M^{-1} - M^{-1} x x^T M^{-1} / (1 + x^T M^{-1} x)
        # M^{-1} = P_{t-1} / lambda
        # P_t = P_{t-1}/lambda - (P_{t-1}/lambda) x x^T (P_{t-1}/lambda) / (1 + x^T (P_{t-1}/lambda) x)
        # P_t = P_{t-1}/lambda - (1/lambda^2) P_{t-1} x x^T P_{t-1} / (1 + (1/lambda) x^T P_{t-1} x)
        
        # This is computationally efficient:
        # 1. P_old_x = P_old @ x (d)
        # 2. x_P_old_x = x @ P_old_x (1)
        # 3. denom = 1 + x_P_old_x / lambda
        # 4. P_new = P_old / lambda - (1 / (lambda * denom)) * np.outer(P_old_x, P_old_x)
        
        x_P_old_x = x @ P_old_x
        denom = 1.0 + x_P_old_x / lam
        
        # Avoid division by zero
        if denom < 1e-8:
            denom = 1e-8
            
        P_new = P_old / lam
        correction = (1.0 / (lam * denom)) * np.outer(P_old_x, P_old_x)
        P_new -= correction
        
        # Ensure symmetry and positive definiteness
        P_new = (P_new + P_new.T) / 2.0
        
        self.P[arm] = P_new
        
        # Update Theta
        # theta_t = theta_{t-1} + P_new x (y_t - x^T theta_{t-1})
        # Note: Some formulations use P_old, some P_new. 
        # Standard RLS: theta_t = theta_{t-1} + K_t (y_t - x^T theta_{t-1})
        # where K_t = P_t x / (1 + x^T P_t x) ?? No, K is derived from P_t.
        # Let's stick to the simple update:
        # theta_new = theta_old + P_new @ x * error
        # Wait, the gain is usually calculated with the *new* P?
        # Let's check:
        # theta_t = theta_{t-1} + P_t x (y - x^T theta_{t-1})
        # This is correct if P_t is the inverse of the current information matrix.
        
        self.theta[arm] = theta + (P_new @ x) * error
        
        # 4. Update Cholesky for Sampling
        self._L_P[arm] = self._update_chol_P(arm)

# EVOLVE-BLOCK-END
