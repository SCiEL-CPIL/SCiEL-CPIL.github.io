"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

class Policy:
    """
    Adaptive LinUCB for non-stationary linear bandits.
    
    Combines:
    1. Per-arm adaptive forgetting factor (lambda) based on recent MSE.
    2. Adaptive exploration parameter (alpha) based on global MSE.
    3. Efficient matrix inverse updates via Sherman-Morrison-Woodbury.
    4. Cached theta vectors updated incrementally for fast select().
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Initialize per-arm linear models
        # inv_A is the inverse covariance matrix, b is the moment vector
        self.inv_A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        # Cached theta = inv_A @ b, updated incrementally
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        
        # Adaptive forgetting factor parameters
        # Lambda controls how much old data is retained.
        # High lambda (close to 1) = long memory. Low lambda = short memory.
        self.lam_min = 0.90
        self.lam_max = 0.998
        
        # Buffer for recent squared errors per arm to determine local stability
        self.error_buffers = [deque(maxlen=20) for _ in range(n_arms)]
        
        # UCB exploration parameter
        # Alpha controls the width of the confidence interval.
        self.alpha_min = 0.5
        self.alpha_max = 1.8
        self.alpha = 1.0
        
        # Global error tracking for adaptive alpha
        self.global_errors = deque(maxlen=40)
        
        # Deadline guard
        self._start_time = time.time()
        self._deadline = self._start_time + 140.0  # 10s safety margin before 150s kill
        
        # Step counter
        self.t = 0

    def _elapsed(self):
        return time.time() - self._start_time

    def _get_lambda(self, arm):
        """
        Calculate the adaptive forgetting factor for a specific arm
        based on recent squared prediction errors (MSE).
        Smooth mapping: high MSE -> low lambda (fast forgetting).
        """
        buf = self.error_buffers[arm]
        if len(buf) == 0:
            return self.lam_max
            
        # Convert deque to array for efficient MSE calculation
        errs = np.array(list(buf))
        mse = np.mean(errs)
        
        # Smooth mapping to [lam_min, lam_max]
        # If mse is 0, norm=0, lam=lam_max (remember everything)
        # If mse is large, norm->1, lam->lam_min (forget fast)
        mse_scale = 0.15
        norm = mse / (mse + mse_scale)
        lam = self.lam_max - norm * (self.lam_max - self.lam_min)
        
        return max(self.lam_min, min(self.lam_max, lam))

    def _update_alpha(self):
        """
        Adapt exploration parameter alpha based on global error level.
        High global error -> more exploration (higher alpha).
        """
        if len(self.global_errors) < 10:
            return
            
        errs = np.array(list(self.global_errors))
        mse = np.mean(errs)
        
        # Target alpha: more exploration when errors are high
        # Scale factor 0.3 means if mse > 0.3, we push towards alpha_max
        target = self.alpha_min + (self.alpha_max - self.alpha_min) * (mse / (mse + 0.3))
        
        # Conservative smooth update to avoid jitter
        self.alpha = 0.85 * self.alpha + 0.15 * target
        self.alpha = max(self.alpha_min, min(self.alpha_max, self.alpha))

    def _update_model(self, arm, context, reward, lam):
        """
        Update inv_A, b, and theta using Sherman-Morrison-Woodbury.
        
        A_new = lam * A_old + x x^T
        b_new = lam * b_old + r * x
        
        inv_A_new = inv(lam * A_old + x x^T)
        Let B = lam * A_old. inv_B = (1/lam) * inv_A_old.
        inv_A_new = inv_B - (inv_B @ x)(inv_B @ x)^T / (1 + x^T inv_B x)
        
        theta_new = inv_A_new @ b_new
        """
        x = context
        r = reward
        
        # Compute inv_B = (1/lam) * inv_A_old
        inv_B = (1.0 / lam) * self.inv_A[arm]
        
        # Compute tmp = inv_B @ x
        tmp = inv_B @ x
        
        # Denominator
        denom = 1.0 + float(x @ tmp)
        
        # Update inv_A
        self.inv_A[arm] = inv_B - np.outer(tmp, tmp) / denom
        
        # Update b
        self.b[arm] = lam * self.b[arm] + r * x
        
        # Update theta = inv_A @ b
        self.theta[arm] = self.inv_A[arm] @ self.b[arm]

    def select(self, context):
        # Deadline check
        if self._elapsed() > self._deadline:
            # Fallback: greedy based on current estimates to avoid hanging
            scores = []
            for a in range(self.n):
                scores.append(float(self.theta[a] @ context))
            return int(np.argmax(scores))

        context = np.asarray(context, dtype=np.float64)
        scores = np.empty(self.n)
        
        for a in range(self.n):
            # Mean prediction using cached theta
            mu = float(self.theta[a] @ context)
            
            # Variance term
            var = float(context @ self.inv_A[a] @ context)
            std = np.sqrt(max(var, 0.0))
            
            # UCB score
            scores[a] = mu + self.alpha * std
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        
        # Deadline check
        if self._elapsed() > self._deadline:
            return
        
        self.t += 1
        
        # 1. Compute residual using cached theta BEFORE update
        pred = float(self.theta[arm] @ context)
        residual = reward - pred
        residual_sq = residual ** 2
        
        # 2. Update error buffers
        self.error_buffers[arm].append(residual_sq)
        self.global_errors.append(residual_sq)
        
        # 3. Determine adaptive parameters
        lam = self._get_lambda(arm)
        self._update_alpha()
        
        # 4. Update model: inv_A, b, and theta
        self._update_model(arm, context, reward, lam)

# EVOLVE-BLOCK-END
