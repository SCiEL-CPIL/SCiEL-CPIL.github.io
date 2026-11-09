"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

class Policy:
    """
    Non-stationary Linear Contextual Bandit with Multi-Timescale Adaptive Forgetting.
    
    This policy uses a dual-timescale LinUCB approach:
    - Short window (fast forgetting) tracks rapid weight shifts.
    - Long window (slow forgetting) provides a stable baseline.
    - The final prediction is a weighted blend of the two estimates.
    - Adaptive alpha increases exploration during high-reward-variance periods.
    - Per-arm adaptive lambda adjusts forgetting rate based on recent residuals.
    
    Key features:
    1. Dual-timescale estimation (fast + slow) for robust tracking of non-stationarity.
    2. Per-arm adaptive lambda based on residual magnitude and variance.
    3. Global drift detection to trigger aggressive forgetting across all arms.
    4. Adaptive alpha that increases during drift and decreases in stable periods.
    5. Strict time limit guard to ensure termination within evaluator constraints.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Initialize linear regression matrices for both timescales
        self.reg = 0.1
        # Fast timescale
        self.A_fast = [np.eye(dim) * self.reg for _ in range(n_arms)]
        self.b_fast = [np.zeros(dim) for _ in range(n_arms)]
        # Slow timescale
        self.A_slow = [np.eye(dim) * self.reg for _ in range(n_arms)]
        self.b_slow = [np.zeros(dim) for _ in range(n_arms)]
        
        # Adaptive forgetting parameters
        self.lambda_fast_base = 0.90
        self.lambda_slow_base = 0.97
        self.lambdas_fast = [self.lambda_fast_base for _ in range(n_arms)]
        self.lambdas_slow = [self.lambda_slow_base for _ in range(n_arms)]
        self.lambda_fast_min = 0.75
        self.lambda_fast_max = 0.95
        self.lambda_slow_min = 0.90
        self.lambda_slow_max = 0.995
        self.lambda_adapt_rate = 0.1
        
        # Residual tracking for each arm
        self.residuals = [deque(maxlen=10) for _ in range(n_arms)]
        
        # Global drift tracking
        self.recent_rewards = deque(maxlen=20)
        self.global_reward_avg = 0.0
        self.global_reward_var = 1.0
        
        # Adaptive exploration coefficient
        self.alpha_base = 1.2
        self.alpha = 1.2
        self.alpha_min = 0.8
        self.alpha_max = 2.5
        self.alpha_adapt_rate = 0.15
        
        # Blend weight between fast and slow estimates
        # Higher w means more trust in fast estimate
        self.w_fast = 0.6
        self.w_fast_min = 0.3
        self.w_fast_max = 0.8
        self.w_adapt_rate = 0.1
        
        # Counters
        self.t = 0
        self.counts = [0] * n_arms
        
        # Time management
        self.start_time = time.time()
        self.deadline = self.start_time + 145.0  # Safety margin: 150s limit, stop at 145s

    def _solve(self, A, b):
        """Solve A x = b with fallback for singular matrices."""
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, b, rcond=None)[0]

    def _update_lambda(self, arm, residual_sq):
        """
        Update the forgetting factors for a specific arm based on recent errors.
        High or volatile errors -> lower lambda (faster forgetting).
        Low and stable errors -> higher lambda (slower forgetting, more memory).
        """
        self.residuals[arm].append(residual_sq)
        
        if len(self.residuals[arm]) < 3:
            return

        recent_res = list(self.residuals[arm])
        mean_res = np.mean(recent_res)
        std_res = np.std(recent_res) if len(recent_res) > 2 else 0.0
        
        error_metric = mean_res + std_res
        
        sens = 0.05
        adjustment = -sens * error_metric
        
        # Update fast lambda
        new_lambda_fast = self.lambdas_fast[arm] + self.lambda_adapt_rate * adjustment
        new_lambda_fast = max(self.lambda_fast_min, min(self.lambda_fast_max, new_lambda_fast))
        self.lambdas_fast[arm] = new_lambda_fast
        
        # Update slow lambda (less sensitive)
        new_lambda_slow = self.lambdas_slow[arm] + 0.5 * self.lambda_adapt_rate * adjustment
        new_lambda_slow = max(self.lambda_slow_min, min(self.lambda_slow_max, new_lambda_slow))
        self.lambdas_slow[arm] = new_lambda_slow

    def _update_alpha(self, is_drift, reward_var):
        """
        Adapt the exploration coefficient based on drift detection and reward variance.
        During drift or high variance, increase alpha to explore more.
        In stable periods, decrease alpha to exploit more.
        """
        if is_drift:
            self.alpha = min(self.alpha_max, self.alpha + self.alpha_adapt_rate)
        elif reward_var > 0.5:
            self.alpha = min(self.alpha_max, self.alpha + 0.5 * self.alpha_adapt_rate)
        else:
            self.alpha = max(self.alpha_base, self.alpha * (1.0 - 0.01))
        
        self.alpha = max(self.alpha_min, min(self.alpha_max, self.alpha))

    def _update_w_fast(self, residual_sq, is_drift):
        """
        Adapt the blend weight between fast and slow estimates.
        During drift or high residuals, trust the fast estimate more.
        """
        if is_drift or residual_sq > 1.0:
            self.w_fast = min(self.w_fast_max, self.w_fast + self.w_adapt_rate)
        else:
            self.w_fast = max(self.w_fast_min, self.w_fast - 0.5 * self.w_adapt_rate)

    def _check_global_drift(self, reward):
        """
        Detect global drift by comparing current reward to recent history.
        Returns (True if drift, current variance).
        """
        self.recent_rewards.append(reward)
        
        if len(self.recent_rewards) < 5:
            return False, 0.0
            
        current_mean = np.mean(self.recent_rewards)
        current_var = np.var(self.recent_rewards)
        
        self.global_reward_avg = 0.9 * self.global_reward_avg + 0.1 * current_mean
        self.global_reward_var = 0.9 * self.global_reward_var + 0.1 * current_var
        
        drift_threshold = 0.5 * (1.0 + np.sqrt(self.global_reward_var + 1e-6))
        
        if abs(current_mean - self.global_reward_avg) > drift_threshold:
            return True, current_var
            
        return False, current_var

    def _select_arm(self, context):
        """Select arm using dual-timescale LinUCB with adaptive parameters."""
        context = np.asarray(context, dtype=np.float64)
        self.t += 1

        scores = []
        for a in range(self.n):
            # Fast estimate
            theta_fast = self._solve(self.A_fast[a], self.b_fast[a])
            mu_fast = theta_fast @ context
            
            # Slow estimate
            theta_slow = self._solve(self.A_slow[a], self.b_slow[a])
            mu_slow = theta_slow @ context
            
            # Blended prediction
            mu = self.w_fast * mu_fast + (1.0 - self.w_fast) * mu_slow
            
            # Compute variance terms for both timescales
            try:
                A_inv_fast_context = np.linalg.solve(self.A_fast[a], context)
                v_fast = context @ A_inv_fast_context
            except np.linalg.LinAlgError:
                v_fast = 1.0
            
            try:
                A_inv_slow_context = np.linalg.solve(self.A_slow[a], context)
                v_slow = context @ A_inv_slow_context
            except np.linalg.LinAlgError:
                v_slow = 1.0
            
            # Blended variance
            v = self.w_fast * v_fast + (1.0 - self.w_fast) * v_slow
            
            bonus = self.alpha * np.sqrt(max(v, 0.0))
            
            scores.append(mu + bonus)

        best_arm = int(np.argmax(scores))
        return best_arm

    def select(self, context):
        if time.time() > self.deadline:
            return 0
        return self._select_arm(context)

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        
        # 1. Compute prediction error using blended estimate
        theta_fast = self._solve(self.A_fast[arm], self.b_fast[arm])
        theta_slow = self._solve(self.A_slow[arm], self.b_slow[arm])
        mu_pred = self.w_fast * (theta_fast @ context) + (1.0 - self.w_fast) * (theta_slow @ context)
        residual = reward - mu_pred
        residual_sq = residual ** 2

        # 2. Update adaptive lambdas for this arm
        self._update_lambda(arm, residual_sq)

        # 3. Check for global drift
        is_global_drift, reward_var = self._check_global_drift(reward)
        
        # 4. Update adaptive alpha and blend weight
        self._update_alpha(is_global_drift, reward_var)
        self._update_w_fast(residual_sq, is_global_drift)
        
        if is_global_drift:
            # Aggressively forget for all arms
            for a in range(self.n):
                self.lambdas_fast[a] = max(self.lambda_fast_min, self.lambdas_fast[a] * 0.95)
                self.lambdas_slow[a] = max(self.lambda_slow_min, self.lambdas_slow[a] * 0.98)

        # 5. Update linear models for the pulled arm
        lam_fast = self.lambdas_fast[arm]
        lam_slow = self.lambdas_slow[arm]
        self.A_fast[arm] = lam_fast * self.A_fast[arm] + np.outer(context, context)
        self.b_fast[arm] = lam_fast * self.b_fast[arm] + reward * context
        self.A_slow[arm] = lam_slow * self.A_slow[arm] + np.outer(context, context)
        self.b_slow[arm] = lam_slow * self.b_slow[arm] + reward * context
        
        # 6. Update count
        self.counts[arm] += 1


# EVOLVE-BLOCK-END
