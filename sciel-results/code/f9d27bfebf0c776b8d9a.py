"""Non-stationary linear contextual bandit with adaptive Thompson Sampling and maintained Cholesky."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Adaptive Thompson Sampling for non-stationary linear bandits.
    
    Combines:
    - Thompson Sampling for exploration (sampling from posterior)
    - Maintained Cholesky factorization for efficient inverse and sampling
    - Smooth adaptive forgetting via EMA of Mahalanobis residuals
    - Time-aware safety margin
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.I = np.eye(dim)
        # Per-arm precision and cross-moment
        self.A = [self.I.copy() for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Maintained Cholesky factors L such that A = L @ L.T
        # Initialize with identity: L = I
        self.L = [np.eye(dim) for _ in range(n_arms)]
        
        # Adaptive forgetting parameters
        self.base_decay = 0.98       # mild forgetting each step (lower for better tracking)
        self.shift_decay = 0.5       # strong forgetting on regime shift
        self.shift_threshold = 2.0   # residual/uncertainty threshold
        self.min_decay = 0.3         # floor for decay factor
        self.t = 0
        
        # Time management
        self.start_time = time.time()
        self.deadline = self.start_time + 145.0  # Leave 5s margin
        
        # EMA of normalized residuals for smooth adaptive decay
        self.residual_ema = 0.0
        self.residual_alpha = 0.1    # EMA smoothing factor
        
        # Noise estimate
        self.noise_est = 1.0
        self.noise_alpha = 0.9

    def _safe(self):
        return time.time() >= self.deadline

    def _update_cholesky(self, arm, x, reward):
        """
        Update A and b for the given arm, and maintain Cholesky factor L.
        A_new = A_old + x x^T
        b_new = b_old + reward * x
        """
        A = self.A[arm]
        b = self.b[arm]
        L = self.L[arm]
        d = self.d
        
        # Update A and b
        A += np.outer(x, x)
        b += reward * x
        
        # Update Cholesky factor using the formula for rank-1 update:
        # If A = L L^T, then A + x x^T = L_new L_new^T
        # We can compute L_new using the Cholesky update formula.
        
        # Method: Compute w = L^{-1} x, then update
        # L_new = L * (I + w w^T)^{1/2} in a specific form
        
        # More efficient: use the standard Cholesky update algorithm
        # L_new[i,j] = L[i,j] for j < i
        # L_new[i,i] = sqrt(L[i,i]^2 + w_i^2)
        # L_new[i,j] = (L[i,j] + w_i w_j) / L_new[i,i] for j > i
        
        # First compute w = L^{-1} x
        w = np.linalg.solve(L, x)
        
        L_new = np.empty_like(L)
        for i in range(d):
            L_new[i, i] = np.sqrt(L[i, i]**2 + w[i]**2)
            for j in range(i):
                L_new[i, j] = L[i, j]
            for j in range(i+1, d):
                L_new[i, j] = (L[i, j] + w[i] * w[j]) / L_new[i, i]
        
        self.L[arm] = L_new

    def _apply_forgetting(self, arm, decay):
        """
        Apply forgetting factor to A and b, and update Cholesky factor.
        A_new = decay * A_old
        b_new = decay * b_old
        L_new = sqrt(decay) * L_old
        """
        self.A[arm] *= decay
        self.b[arm] *= decay
        self.L[arm] *= np.sqrt(decay)

    def _compute_adaptive_decay(self, normalized_residual):
        """
        Compute decay factor based on prediction error using EMA.
        High normalized residual -> strong forgetting (lower decay).
        Low normalized residual -> mild forgetting (higher decay).
        """
        # Update EMA of normalized residuals
        self.residual_ema = (1 - self.residual_alpha) * self.residual_ema + self.residual_alpha * normalized_residual
        
        # Use EMA for smooth adaptation
        if self.residual_ema > self.shift_threshold:
            # Strong shift: use shift_decay or even lower if very large error
            excess = (self.residual_ema - self.shift_threshold) / self.shift_threshold
            decay = self.shift_decay - 0.2 * min(excess, 1.0)
            return max(decay, self.min_decay)
        else:
            # Mild forgetting, slightly modulated by residual
            proximity = self.residual_ema / self.shift_threshold
            decay = self.base_decay - 0.02 * proximity
            return max(decay, self.min_decay)

    def select(self, context):
        context = np.asarray(context, dtype=float)
        
        if self._safe():
            # Fallback: return greedy arm based on current theta
            scores = []
            for a in range(self.n):
                L = self.L[a]
                # theta = A^{-1} b = L^{-T} L^{-1} b
                # Solve L y = b, then L^T theta = y
                y = np.linalg.solve(L, self.b[a])
                theta = np.linalg.solve(L.T, y)
                mu = theta @ context
                scores.append(mu)
            return int(np.argmax(scores))
        
        # Thompson Sampling: sample from posterior for each arm
        samples = []
        for a in range(self.n):
            L = self.L[a]
            d = self.d
            
            # theta_mean = A^{-1} b = L^{-T} L^{-1} b
            y = np.linalg.solve(L, self.b[a])
            theta_mean = np.linalg.solve(L.T, y)
            mu = theta_mean @ context
            
            # Sample theta ~ N(theta_mean, A^{-1})
            # A^{-1} = L^{-T} L^{-1}
            # So theta = theta_mean + L^{-T} z where z ~ N(0, I)
            # Equivalently: theta = theta_mean + w where L^T w = z
            # Or: theta = theta_mean + L^{-T} z
            
            z = np.random.randn(d)
            # Solve L^T w = z, then theta_sample = theta_mean + w
            w = np.linalg.solve(L.T, z)
            theta_sample = theta_mean + w
            sample_score = theta_sample @ context
            
            samples.append(sample_score)
        
        return int(np.argmax(samples))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=float)
        self.t += 1
        
        if self._safe():
            # Minimal update to maintain state
            self.A[arm] = self.A[arm] + np.outer(context, context)
            self.b[arm] = self.b[arm] + reward * context
            return
        
        # Compute current model prediction and uncertainty for this arm
        L = self.L[arm]
        d = self.d
        
        # theta = A^{-1} b = L^{-T} L^{-1} b
        y = np.linalg.solve(L, self.b[arm])
        theta = np.linalg.solve(L.T, y)
        mu = theta @ context
        
        # Compute variance: context^T A^{-1} context
        # A^{-1} = L^{-T} L^{-1}
        # Let v = L^{-1} context, then var = v^T v
        v = np.linalg.solve(L, context)
        var = np.dot(v, v)
        std = np.sqrt(max(var, 1e-12))
        
        # Update noise estimate
        error = reward - mu
        self.noise_est = self.noise_alpha * self.noise_est + (1 - self.noise_alpha) * (error ** 2)
        noise_std = np.sqrt(max(self.noise_est, 1e-8))
        
        # Normalized residual: how surprised are we?
        if noise_std > 1e-8:
            normalized_residual = abs(reward - mu) / noise_std
        else:
            normalized_residual = abs(reward - mu)
        
        # Decide decay factor based on normalized residual
        decay = self._compute_adaptive_decay(normalized_residual)
        
        # Apply forgetting
        self._apply_forgetting(arm, decay)
        
        # Add new observation and update Cholesky
        self._update_cholesky(arm, context, reward)
# EVOLVE-BLOCK-END
