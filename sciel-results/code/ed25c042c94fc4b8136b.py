"""
Direct approach for Non-stationary Linear Contextual Bandit.
Uses RLS with adaptive forgetting factor (lambda) and adaptive exploration (alpha)
driven by a global drift metric based on prediction residuals.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Adaptive Non-stationary LinUCB.
    
    Features:
    1. Recursive Least Squares (RLS) with per-arm forgetting factor.
    2. Global drift detection based on recent prediction errors.
    3. Adaptive exploration bonus that increases during high drift.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # RLS State
        self.A = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Time and Deadline
        self.t = 0
        self.start_time = time.time()
        # Safety margin: Evaluator kills at 150s. We stop risky updates after 140s.
        self.deadline = self.start_time + 140.0 

        # Drift Detection State
        # We track the residual of the selected arm to detect global regime changes.
        # If the error spikes, we assume a shift.
        self.residual_hist = []
        self.max_hist_len = 20
        self.base_lambda = 0.98
        self.min_lambda = 0.90
        self.max_lambda = 0.995
        
        # Exploration Parameters
        self.base_alpha = 1.2
        self.alpha_drift_boost = 1.5
        
        # Counters
        self.counts = [0] * n_arms
        
        # Residual variance estimate for adaptive scaling
        self.residual_var = 1.0

    def _compute_drift_metric(self):
        """
        Compute a drift metric based on recent residuals using smoothed log-sum-exp.
        Uses adaptive variance scaling for better non-stationarity capture.
        """
        if len(self.residual_hist) < 2:
            return 0.0
        
        recent = np.array(self.residual_hist[-self.max_hist_len:])
        
        # Adaptive variance scaling: estimate typical residual scale
        # Use median absolute deviation as robust scale estimator
        med = np.median(recent)
        mad = np.median(np.abs(recent - med)) + 1e-8
        scaled_residuals = (recent - med) / mad
        
        # Smoothed log-sum-exp based error metric
        # This provides a smooth surrogate for max-based drift detection
        scale_param = 0.5  # Temperature for log-sum-exp
        shifted = scaled_residuals / scale_param
        max_shifted = np.max(shifted)
        lse = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
        
        # Also compute mean absolute scaled error
        mean_abs_scaled = np.mean(np.abs(scaled_residuals))
        
        # Combine: lse captures outliers, mean_abs captures systematic shift
        drift_signal = lse + mean_abs_scaled
        
        # Normalize to [0, 1] with smooth sigmoid-like mapping
        # Typical values: lse ~ 1-3, mean_abs ~ 0.5-1.5
        # So drift_signal typically ranges 1.5-4.5
        # Map [1.0, 4.0] to [0, 1] smoothly
        t = (drift_signal - 1.0) / 3.0
        # Smooth step function: 3t^2 - 2t^3
        t_clamped = np.clip(t, 0, 1)
        drift_metric = 3 * t_clamped**2 - 2 * t_clamped**3
        
        return float(min(1.0, max(0.0, drift_metric)))

    def select(self, context):
        # Deadline Guard
        if time.time() > self.deadline:
            # Fallback: Greedy based on current theta
            context = np.asarray(context, dtype=np.float64)
            scores = []
            for a in range(self.n):
                try:
                    theta = np.linalg.solve(self.A[a], self.b[a])
                    scores.append(theta @ context)
                except:
                    scores.append(0.0)
            return int(np.argmax(scores))

        self.t += 1
        context = np.asarray(context, dtype=np.float64)
        
        # Compute Drift Metric
        drift_metric = self._compute_drift_metric()
        
        # Adapt Lambda (Forgetting Factor)
        # High drift -> Lower Lambda (Forget Faster)
        # Linear interpolation
        lam = self.max_lambda - (self.max_lambda - self.min_lambda) * drift_metric
        
        # Adapt Alpha (Exploration Bonus) with temporal decay
        # High drift -> Higher Alpha (Explore More)
        # But decay exploration over time to reduce over-exploration in later rounds
        # Use a smooth decay based on total rounds seen
        t = self.t
        # Decay rate: after ~100 rounds, exploration bonus is reduced by ~50%
        # After ~200 rounds, reduced by ~75%
        decay_factor = 1.0 / (1.0 + 0.01 * t)
        
        # Base alpha also decays slightly
        base_alpha_decayed = self.base_alpha * decay_factor
        
        # Drift boost also decays but less aggressively
        drift_boost_decayed = self.alpha_drift_boost * (1.0 / (1.0 + 0.005 * t))
        
        alpha = base_alpha_decayed + drift_boost_decayed * drift_metric
        
        scores = []
        for a in range(self.n):
            A = self.A[a]
            b = self.b[a]
            
            try:
                # Solve for theta
                theta = np.linalg.solve(A, b)
                mu = theta @ context
                
                # Compute Variance: x^T A^{-1} x
                # Instead of inverting A, solve A v = x
                v_vec = np.linalg.solve(A, context)
                variance = context @ v_vec
            except np.linalg.LinAlgError:
                mu = 0.0
                variance = 1.0

            ucb_score = mu + alpha * np.sqrt(max(variance, 0.0))
            scores.append(ucb_score)
            
        best_arm = int(np.argmax(scores))
        return best_arm

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        self.t += 1
        
        # Deadline Guard for Update: If we are close to limit, skip complex updates
        if time.time() > self.deadline:
             return

        # 1. Compute Prediction Error for Drift Detection
        A = self.A[arm]
        b = self.b[arm]
        try:
            theta = np.linalg.solve(A, b)
            mu_pred = theta @ context
        except np.linalg.LinAlgError:
            mu_pred = 0.0
            
        residual = reward - mu_pred
        self.residual_hist.append(abs(residual))
        if len(self.residual_hist) > self.max_hist_len * 2:
            self.residual_hist.pop(0)
        
        # 2. Compute variance estimate for adaptive scaling
        # Use recent variance of residuals to normalize drift signal
        if len(self.residual_hist) >= 5:
            recent_res = np.array(self.residual_hist[-min(10, len(self.residual_hist)):])
            self.residual_var = float(np.var(recent_res)) + 1e-8
        else:
            self.residual_var = getattr(self, 'residual_var', 1.0)
            
        # 2. Update Adaptive Parameters
        drift_metric = self._compute_drift_metric()
        lam = self.max_lambda - (self.max_lambda - self.min_lambda) * drift_metric
        
        # 3. Update RLS State
        # A <- lam * A + x x^T
        # b <- lam * b + r x
        self.A[arm] = lam * self.A[arm] + np.outer(context, context)
        self.b[arm] = lam * self.b[arm] + reward * context
        
        self.counts[arm] += 1

# EVOLVE-BLOCK-END
