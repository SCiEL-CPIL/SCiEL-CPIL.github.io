"""Non-stationary linear contextual bandit with adaptive Thompson Sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary Linear Contextual Bandit using Thompson Sampling with
    adaptive forgetting and Cholesky-based uncertainty estimation.
    
    Strategy:
    1. Maintain a recency-weighted posterior for each arm via exponential forgetting.
    2. Use Thompson Sampling for action selection.
    3. Adaptive forgetting: detect regime shifts via z-score of prediction error,
       apply strong decay when shift detected, mild decay otherwise.
    4. Cholesky-based uncertainty computation for numerical stability.
    5. Adaptive regularization that increases after regime shifts.
    6. Deadline guard to ensure we finish within time limits.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self._t0 = time.time()
        self._deadline = 115.0  # safety margin before 120s limit
        
        # Initialize per-arm matrices for ridge regression
        # Start with moderate regularization for stability
        init_lambda = 1.0
        self.A = [np.eye(dim) * init_lambda for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Forgetting parameters
        self.mild_decay = 0.99
        self.strong_decay = 0.3
        self.regime_threshold = 2.5  # z-score threshold for regime shift detection
        
        # Adaptive parameters
        self.exploration_scale = 1.0
        self.base_lambda = 1.0
        self.shift_lambda = 5.0  # extra regularization after regime shift
        
        # Track prediction errors with exponential moving average
        self.ema_errors = [0.0] * n_arms
        self.ema_error_alpha = 0.1  # EMA smoothing factor
        
        # Track number of observations per arm
        self.arm_counts = [0] * n_arms
        
        # Track recent regime shifts
        self.recent_shifts = [0] * n_arms
        self.shift_decay = 0.95

    def _compute_theta_and_std(self, a, context):
        """Compute theta and std using Cholesky decomposition for stability."""
        try:
            # Ensure positive definite by adding small jitter if needed
            A = self.A[a]
            b = self.b[a]
            
            # Try Cholesky factorization
            L = np.linalg.cholesky(A)
            # Solve L * y = b, then L^T * theta = y
            y = np.linalg.solve(L, b)
            theta = np.linalg.solve(L.T, y)
            
            # Compute variance: context^T * A^{-1} * context
            # A^{-1} = L^{-T} * L^{-1}
            # context^T * A^{-1} * context = ||L^{-1} * context||^2
            z = np.linalg.solve(L, context)
            var = np.dot(z, z)
            std = np.sqrt(max(var, 0.0))
            
            return theta, std
        except np.linalg.LinAlgError:
            # Fallback: add more regularization and retry
            A_reg = self.A[a] + np.eye(self.d) * 1.0
            try:
                L = np.linalg.cholesky(A_reg)
                y = np.linalg.solve(L, self.b[a])
                theta = np.linalg.solve(L.T, y)
                z = np.linalg.solve(L, context)
                var = np.dot(z, z)
                std = np.sqrt(max(var, 0.0))
                return theta, std
            except np.linalg.LinAlgError:
                return np.zeros(self.d), 1.0

    def select(self, context):
        # Deadline check
        if time.time() - self._t0 > self._deadline:
            return 0
        
        context = np.asarray(context, dtype=np.float64)
        
        # Thompson Sampling: sample from posterior for each arm
        scores = []
        for a in range(self.n):
            theta, std = self._compute_theta_and_std(a, context)
            
            # Adaptive exploration: increase exploration for arms with low counts
            # or after recent regime shifts
            count_bonus = max(0, 5 - self.arm_counts[a]) * 0.1
            shift_bonus = self.recent_shifts[a] * 0.5
            
            effective_std = std * (1.0 + count_bonus + shift_bonus)
            
            # Sample from posterior: theta ~ N(theta, std^2)
            sampled_score = theta @ context + self.exploration_scale * effective_std * np.random.randn()
            scores.append(sampled_score)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Deadline check
        if time.time() - self._t0 > self._deadline:
            return
        
        context = np.asarray(context, dtype=np.float64)
        
        # Compute prediction and uncertainty before update
        theta, std = self._compute_theta_and_std(arm, context)
        pred = theta @ context
        
        # Compute residual and z-score
        residual = reward - pred
        z_score = abs(residual) / max(std, 1e-6)
        
        # Update EMA of errors
        self.ema_errors[arm] = (1 - self.ema_error_alpha) * self.ema_errors[arm] + self.ema_error_alpha * residual
        
        # Adaptive forgetting: strong decay if regime shift detected
        if z_score > self.regime_threshold:
            decay = self.strong_decay
            self.recent_shifts[arm] = min(self.recent_shifts[arm] + 1.0, 5.0)
        else:
            decay = self.mild_decay
        
        # Decay recent shift counter
        self.recent_shifts[arm] *= self.shift_decay
        
        # Adaptive regularization: increase lambda after regime shifts
        adaptive_lambda = self.base_lambda + self.recent_shifts[arm] * (self.shift_lambda - self.base_lambda)
        
        # Update with forgetting
        self.A[arm] = decay * self.A[arm] + np.outer(context, context) + np.eye(self.d) * adaptive_lambda * (1 - decay)
        self.b[arm] = decay * self.b[arm] + reward * context
        
        # Increment arm count
        self.arm_counts[arm] += 1
# EVOLVE-BLOCK-END
