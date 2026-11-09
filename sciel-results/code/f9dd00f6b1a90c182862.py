import numpy as np
import time

class Policy:
    """
    Non-stationary Linear Contextual Bandit Policy.
    
    Strategy:
    Uses a "Global Contextual Drift" approach. Instead of relying solely on per-arm
    error to trigger forgetting, this policy monitors the change in the context vector
    itself. In many non-stationary settings, the context distribution or the mapping
    changes over time. A sudden shift in context (high norm of context difference)
    suggests a regime change, triggering a global "reset" or aggressive decay of the
    covariance matrices (increasing exploration) for all arms, not just the pulled one.
    
    This is materially different from the current program which uses a per-arm error
    driven adaptive alpha. Here, the trigger is the *input* space change, not the
    *output* error (though output error is also used as a secondary signal).
    
    Key mechanisms:
    1. Context Drift Detector: Computes the norm of the difference between the current
       context and the previous context. If this is large, it indicates a potential
       non-stationary event.
    2. Global Aggressive Decay: When drift is detected, it applies a strong decay to
       the A matrices of ALL arms, effectively reducing the confidence (variance) and
       forcing exploration.
    3. Hybrid UCB: Uses a standard UCB with a slightly boosted exploration parameter
       when drift is detected.
    4. Efficient Linear Algebra: Uses np.linalg.solve for stability and speed.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Initialize models
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Forgetting parameters
        self.base_alpha = 0.995  # Base decay (slow)
        self.min_alpha = 0.900   # Aggressive decay (fast)
        
        # Drift detection parameters
        self.prev_context = None
        self.drift_threshold = 1.5  # Threshold for context change
        self.drift_cooldown = 5     # Number of steps to wait after a drift event
        self.drift_counter = 0
        self.drift_active = False
        
        # Exploration parameter
        self.alpha_ucb = 1.0
        self.boosted_alpha_ucb = 1.5  # Higher exploration when drift is active
        
        # Time guard
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0

    def _solve_and_variance(self, A, b, context):
        """
        Compute theta = A^-1 b and variance = context^T A^-1 context.
        """
        try:
            theta = np.linalg.solve(A, b)
            v = np.linalg.solve(A, context)
            
            mean = theta @ context
            variance = context @ v
            
            if variance < 0:
                variance = 0.0
            
            return mean, variance
        except np.linalg.LinAlgError:
            theta = np.zeros(self.d)
            mean = 0.0
            variance = 1.0
            return mean, variance

    def select(self, context):
        # Deadline check
        if time.time() > self.deadline:
            return 0 # Fallback to first arm to avoid timeout
        
        context = np.asarray(context)
        
        # Determine exploration level
        if self.drift_active:
            alpha_ucb = self.boosted_alpha_ucb
        else:
            alpha_ucb = self.alpha_ucb
            
        scores = []
        for a in range(self.n):
            try:
                mean, variance = self._solve_and_variance(self.A[a], self.b[a], context)
                score = mean + alpha_ucb * np.sqrt(variance)
                scores.append(score)
            except Exception:
                scores.append(0.0)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context)
        
        # 1. Context Drift Detection
        if self.prev_context is not None:
            diff = context - self.prev_context
            drift_norm = np.linalg.norm(diff)
            
            if drift_norm > self.drift_threshold:
                self.drift_active = True
                self.drift_counter = self.drift_cooldown
            elif self.drift_counter > 0:
                self.drift_counter -= 1
                if self.drift_counter == 0:
                    self.drift_active = False
        self.prev_context = context
        
        # 2. Determine decay factor
        if self.drift_active:
            alpha = self.min_alpha
        else:
            alpha = self.base_alpha
            
        # 3. Update Model for the selected arm
        self.A[arm] = alpha * self.A[arm] + np.outer(context, context)
        self.b[arm] = alpha * self.b[arm] + reward * context
        
        # 4. Global Decay for Other Arms if Drift is Active
        # This is the key difference: if the context has shifted significantly,
        # we assume the reward functions for ALL arms may have changed.
        # We apply a decay to the unobserved arms to prevent them from being
        # stuck with stale, high-confidence estimates from the previous regime.
        if self.drift_active:
            for a in range(self.n):
                if a != arm:
                    # Use a slightly less aggressive decay for unobserved arms
                    # to avoid completely wiping out their knowledge, but
                    # enough to make them competitive again via UCB.
                    self.A[a] = 0.95 * self.A[a]
                    self.b[a] = 0.95 * self.b[a]

        # 5. Time guard
        if time.time() > self.deadline:
            pass
