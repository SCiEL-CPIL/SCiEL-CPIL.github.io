"""
Refined Non-stationary Linear Contextual Bandit Policy.
Uses Per-Arm Drift Detection with Exponentially Weighted Moving Averages (EWMA)
for adaptive forgetting factors in Recursive Least Squares (RLS).
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Adaptive Non-stationary LinUCB with Per-Arm Drift Detection.
    
    Improvements over global drift:
    1. Tracks residual EWMA per arm to detect local weight changes.
    2. Adapts forgetting factor per arm based on local drift signal.
    3. Handles 'stale' arms by applying a minimum forgetting boost if idle.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # RLS State
        # Initialize with identity matrix scaled by a small value to avoid singularities
        # but allow fast initial learning.
        self.A = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Time and Deadline
        self.t = 0
        self.start_time = time.time()
        # Safety margin: Evaluator kills at 150s. We stop risky updates after 140s.
        self.deadline = self.start_time + 140.0 

        # Drift Detection State (Per-Arm)
        # EWMA of squared residuals for each arm
        self.residual_ewma = [1.0 for _ in range(n_arms)]
        self.ewma_alpha = 0.1  # Learning rate for EWMA (higher = more responsive)
        
        # Last time each arm was played
        self.last_played = [-1000] * n_arms
        
        # Forgetting Factor Parameters
        self.base_lambda = 0.98
        self.min_lambda = 0.85
        self.max_lambda = 0.995
        
        # Exploration Parameters
        self.base_alpha = 1.2
        self.alpha_drift_boost = 1.5
        
        # Counters
        self.counts = [0] * n_arms
        
        # Staleness threshold: If arm not played for this many rounds, apply drift boost
        self.stale_threshold = 10

    def _get_arm_drift_signal(self, arm):
        """
        Compute a drift signal for a specific arm based on its residual EWMA.
        Returns a value in [0, 1] where 1 indicates high drift/error.
        """
        # The residual_ewma represents the typical magnitude of error.
        # We need to normalize this to a drift signal.
        # If the error is consistently high, drift is high.
        # However, in the beginning, errors are high due to lack of data.
        # We can use the count to scale the drift signal.
        
        count = self.counts[arm]
        if count < 5:
            return 0.5 # Moderate exploration early on
            
        # Normalize the EWMA. 
        # Typical Gaussian noise variance is around 1.0 in these tasks.
        # If EWMA is much larger than expected noise, it's drift.
        # Let's assume baseline noise std is ~1.0, so variance ~1.0.
        # Drift signal = min(1, (EWMA - 1.0) / 2.0)
        # But we want to be responsive to increases, not just absolute levels.
        
        # A better metric: Compare current residual to EWMA.
        # But we compute this in update. Here we just return the stored 'drift_level'
        # which is updated in update().
        
        return self._current_drift_levels[arm]

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # RLS State
        self.A = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Time and Deadline
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 140.0 

        # Drift Detection State (Per-Arm)
        self.residual_ewma = [1.0 for _ in range(n_arms)]
        self.ewma_alpha = 0.2  # Higher alpha for faster adaptation
        self._current_drift_levels = [0.0] * n_arms
        
        # Last time each arm was played
        self.last_played = [-1000] * n_arms
        
        # Forgetting Factor Parameters
        self.base_lambda = 0.98
        self.min_lambda = 0.85
        self.max_lambda = 0.995
        
        # Exploration Parameters
        self.base_alpha = 1.2
        self.alpha_drift_boost = 1.5
        
        # Counters
        self.counts = [0] * n_arms
        
        # Staleness threshold
        self.stale_threshold = 5

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
        
        # Compute global drift metric for alpha adaptation
        # Use max of individual drift levels for a conservative exploration boost
        global_drift = max(self._current_drift_levels)
        
        # Adapt Alpha (Exploration Bonus)
        # Decay exploration over time
        decay_factor = 1.0 / (1.0 + 0.01 * self.t)
        base_alpha_decayed = self.base_alpha * decay_factor
        drift_boost_decayed = self.alpha_drift_boost * (1.0 / (1.0 + 0.005 * self.t))
        
        alpha = base_alpha_decayed + drift_boost_decayed * global_drift
        
        scores = []
        for a in range(self.n):
            A = self.A[a]
            b = self.b[a]
            
            try:
                # Solve for theta
                theta = np.linalg.solve(A, b)
                mu = theta @ context
                
                # Compute Variance: x^T A^{-1} x
                v_vec = np.linalg.solve(A, context)
                variance = context @ v_vec
            except np.linalg.LinAlgError:
                mu = 0.0
                variance = 1.0

            # Add a small boost to alpha if the arm is stale
            if self.t - self.last_played[a] > self.stale_threshold:
                alpha_arm = alpha * 1.2
            else:
                alpha_arm = alpha

            ucb_score = mu + alpha_arm * np.sqrt(max(variance, 0.0))
            scores.append(ucb_score)
            
        best_arm = int(np.argmax(scores))
        return best_arm

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        self.t += 1
        
        # Deadline Guard for Update
        if time.time() > self.deadline:
            return

        # 1. Compute Prediction Error
        A = self.A[arm]
        b = self.b[arm]
        try:
            theta = np.linalg.solve(A, b)
            mu_pred = theta @ context
        except np.linalg.LinAlgError:
            mu_pred = 0.0
            
        residual = reward - mu_pred
        sq_residual = residual ** 2
        
        # 2. Update EWMA of squared residuals
        self.residual_ewma[arm] = (1 - self.ewma_alpha) * self.residual_ewma[arm] + self.ewma_alpha * sq_residual
        
        # 3. Compute Drift Signal for this arm
        # Normalize the EWMA. Assume baseline noise variance is ~1.0.
        # If EWMA is significantly higher, there is drift.
        # Use a smooth mapping.
        # drift_signal = 1 - exp(-EWMA / scale)
        # Or simply clip the ratio.
        # Let's use: if EWMA > 1.5, start increasing drift.
        # drift_level = min(1.0, max(0.0, (self.residual_ewma[arm] - 1.0) / 2.0))
        
        # To make it smoother and more responsive to *changes*, we can look at the ratio
        # of recent error to past error, but EWMA already does this.
        # Let's refine the mapping:
        # If EWMA is 1.0 (noise), drift = 0.
        # If EWMA is 3.0 (high error), drift = 1.
        # Linear interpolation between 1.0 and 3.0.
        drift_level = (self.residual_ewma[arm] - 1.0) / 2.0
        drift_level = float(np.clip(drift_level, 0.0, 1.0))
        
        # Smooth the drift level to avoid oscillation
        self._current_drift_levels[arm] = 0.8 * self._current_drift_levels[arm] + 0.2 * drift_level
        
        # 4. Compute Forgetting Factor
        # High drift -> Lower Lambda
        # Also apply staleness boost: if arm was not played recently, force some forgetting
        staleness = self.t - self.last_played[arm]
        if staleness > self.stale_threshold:
            # Apply a minimum drift level for stale arms
            stale_drift_boost = min(0.5, (staleness - self.stale_threshold) * 0.05)
            effective_drift = max(self._current_drift_levels[arm], stale_drift_boost)
        else:
            effective_drift = self._current_drift_levels[arm]
            
        lam = self.max_lambda - (self.max_lambda - self.min_lambda) * effective_drift
        
        # 5. Update RLS State
        # A <- lam * A + x x^T
        # b <- lam * b + r x
        self.A[arm] = lam * self.A[arm] + np.outer(context, context)
        self.b[arm] = lam * self.b[arm] + reward * context
        
        self.counts[arm] += 1
        self.last_played[arm] = self.t

# EVOLVE-BLOCK-END
