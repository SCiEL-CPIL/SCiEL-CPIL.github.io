"""Non-stationary linear contextual bandit with adaptive Thompson Sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary Linear Contextual Bandit using Thompson Sampling with
    event-driven adaptive forgetting.
    
    Strategy:
    1. Maintain a recency-weighted posterior (sliding window via forgetting factor)
       for each arm.
    2. Use Thompson Sampling for action selection to balance exploration/exploitation.
    3. Adaptive forgetting: when prediction error exceeds uncertainty (regime shift),
       apply strong decay to rapidly discard stale model; otherwise mild decay.
    4. Use an adaptive noise floor (lambda) to prevent numerical instability and
       allow exploration in early stages or after regime shifts.
    5. Deadline guard to ensure we finish within time limits.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self._t0 = time.time()
        self._deadline = 115.0  # safety margin before 120s limit
        
        # Initialize per-arm matrices for ridge regression
        self.A = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Forgetting parameters
        self.mild_decay = 0.995
        self.strong_decay = 0.5
        self.regime_threshold = 3.0  # z-score threshold for regime shift detection
        
        # Exploration bonus scaling
        self.exploration_scale = 1.0
        
        # Track prediction errors for adaptive forgetting
        self.pred_errors = [0.0] * n_arms
        self.uncertainties = [1.0] * n_arms

    def select(self, context):
        # Deadline check
        if time.time() - self._t0 > self._deadline:
            return 0
        
        context = np.asarray(context, dtype=np.float64)
        
        # Thompson Sampling: sample from posterior for each arm
        scores = []
        for a in range(self.n):
            # Compute theta = A^{-1} b
            try:
                theta = np.linalg.solve(self.A[a], self.b[a])
            except np.linalg.LinAlgError:
                theta = np.zeros(self.d)
            
            # Compute uncertainty for this context
            try:
                inv_A = np.linalg.inv(self.A[a])
                var = context @ inv_A @ context
                std = np.sqrt(max(var, 0.0))
            except np.linalg.LinAlgError:
                std = 1.0
            
            # Update uncertainty estimate
            self.uncertainties[a] = std
            
            # Sample from posterior: theta ~ N(theta, std^2)
            sampled_score = theta @ context + self.exploration_scale * std * np.random.randn()
            scores.append(sampled_score)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Deadline check
        if time.time() - self._t0 > self._deadline:
            return
        
        context = np.asarray(context, dtype=np.float64)
        
        # Compute prediction and uncertainty before update
        try:
            theta = np.linalg.solve(self.A[arm], self.b[arm])
            pred = theta @ context
            inv_A = np.linalg.inv(self.A[arm])
            var = context @ inv_A @ context
            std = np.sqrt(max(var, 0.0))
        except np.linalg.LinAlgError:
            pred = 0.0
            std = 1.0
        
        # Compute residual and z-score
        residual = reward - pred
        z_score = abs(residual) / max(std, 1e-6)
        
        # Adaptive forgetting: strong decay if regime shift detected
        if z_score > self.regime_threshold:
            decay = self.strong_decay
        else:
            decay = self.mild_decay
        
        # Update with forgetting
        self.A[arm] = decay * self.A[arm] + np.outer(context, context)
        self.b[arm] = decay * self.b[arm] + reward * context
        
        # Track prediction error for potential future use
        self.pred_errors[arm] = residual
# EVOLVE-BLOCK-END
