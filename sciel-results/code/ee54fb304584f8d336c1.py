"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

class Policy:
    """
    Adaptive LinUCB for non-stationary linear bandits.
    
    Improvements over the current program:
    1. Robust IQR-based error scaling for adaptive lambda and alpha.
    2. Smoothstep curves for parameter adaptation to prevent abrupt jumps.
    3. Symmetrization of inv_A to reduce numerical drift.
    4. Lower minimum lambda (0.8) for faster regime change detection.
    5. Slightly wider alpha range and better initial values.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Initialize per-arm linear models
        self.inv_A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        
        # Adaptive forgetting factor parameters
        self.lam_min = 0.80
        self.lam_max = 0.998
        
        # Buffer for recent squared errors per arm
        self.error_buffers = [deque(maxlen=20) for _ in range(n_arms)]
        
        # UCB exploration parameter
        self.alpha_min = 0.3
        self.alpha_max = 2.5
        self.alpha = 1.2
        
        # Global error tracking for adaptive alpha
        self.global_errors = deque(maxlen=40)
        
        # Deadline guard
        self._start_time = time.time()
        self._deadline = self._start_time + 140.0
        
        # Step counter
        self.t = 0

    def _elapsed(self):
        return time.time() - self._start_time

    def _robust_scale(self, errors):
        """
        Compute a robust scale estimate from a list of squared errors.
        Uses IQR-based approach: 1.5 * (Q3 - Q1) or fallback to mean.
        """
        if len(errors) < 5:
            # Fallback to mean for small samples
            return max(np.mean(errors), 0.01)
        
        errs = np.array(errors)
        q1 = np.percentile(errs, 25)
        q3 = np.percentile(errs, 75)
        iqr = q3 - q1
        
        if iqr < 1e-6:
            # Degenerate case, use mean
            return max(np.mean(errs), 0.01)
        
        # Scale factor: use IQR as a measure of typical error magnitude
        # Multiply by a constant to map to a reasonable range
        return max(1.5 * iqr, 0.01)

    def _smoothstep(self, x):
        """
        Smoothstep function: 3x^2 - 2x^3 for x in [0, 1].
        Provides a smooth transition from 0 to 1.
        """
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    def _get_lambda(self, arm):
        """
        Calculate the adaptive forgetting factor for a specific arm
        based on recent squared prediction errors using robust scaling.
        """
        buf = self.error_buffers[arm]
        if len(buf) == 0:
            return self.lam_max
            
        errs = list(buf)
        scale = self._robust_scale(errs)
        
        # Normalize MSE by robust scale
        mse = np.mean(errs)
        norm = mse / (mse + scale)
        
        # Smooth mapping to [lam_min, lam_max]
        norm_smooth = self._smoothstep(norm)
        lam = self.lam_max - norm_smooth * (self.lam_max - self.lam_min)
        
        return max(self.lam_min, min(self.lam_max, lam))

    def _update_alpha(self):
        """
        Adapt exploration parameter alpha based on global error level
        using robust scaling.
        """
        if len(self.global_errors) < 5:
            return
            
        errs = list(self.global_errors)
        scale = self._robust_scale(errs)
        mse = np.mean(errs)
        
        # Normalize by robust scale
        norm = mse / (mse + scale)
        norm_smooth = self._smoothstep(norm)
        
        target = self.alpha_min + (self.alpha_max - self.alpha_min) * norm_smooth
        
        # Smooth update to avoid jitter
        self.alpha = 0.85 * self.alpha + 0.15 * target
        self.alpha = max(self.alpha_min, min(self.alpha_max, self.alpha))

    def _update_theta(self, arm, context, reward, lam):
        """
        Update inv_A using Sherman-Morrison-Woodbury identity and update theta cache.
        With symmetrization to reduce numerical drift.
        """
        x = context
        inv_B = (1.0 / lam) * self.inv_A[arm]
        
        # Compute scalar denominator
        tmp = inv_B @ x
        denom = 1.0 + x @ tmp
        
        # Numerator term: (inv_B @ x) (inv_B @ x).T
        numerator = np.outer(tmp, tmp)
        
        inv_A_new = inv_B - numerator / denom
        
        # Symmetrize to reduce numerical drift
        inv_A_new = 0.5 * (inv_A_new + inv_A_new.T)
        
        self.inv_A[arm] = inv_A_new
        
        # Update b
        b_new = lam * self.b[arm] + reward * x
        
        # Update theta cache
        self.theta[arm] = inv_A_new @ b_new
        self.b[arm] = b_new

    def select(self, context):
        # Deadline check
        if self._elapsed() > self._deadline:
            scores = []
            for a in range(self.n):
                scores.append(float(self.theta[a] @ context))
            return int(np.argmax(scores))

        context = np.asarray(context, dtype=np.float64)
        scores = []
        
        for a in range(self.n):
            mu = float(self.theta[a] @ context)
            var = float(context @ self.inv_A[a] @ context)
            std = np.sqrt(max(var, 0.0))
            score = mu + self.alpha * std
            scores.append(score)
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        
        # Deadline check
        if self._elapsed() > self._deadline:
            return
        
        self.t += 1
        
        # 1. Compute residual using current model BEFORE update
        pred = float(self.theta[arm] @ context)
        residual = reward - pred
        residual_sq = residual ** 2
        
        # 2. Update error buffers
        self.error_buffers[arm].append(residual_sq)
        self.global_errors.append(residual_sq)
        
        # 3. Determine adaptive parameters
        lam = self._get_lambda(arm)
        self._update_alpha()
        
        # 4. Update model matrices and theta cache
        self._update_theta(arm, context, reward, lam)

# EVOLVE-BLOCK-END
