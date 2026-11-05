# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit with:
    - Per-arm discounted least-squares with adaptive forgetting
    - UCB exploration with adaptive coefficient
    - Prediction-error-driven acceleration of forgetting
    - Ridge regularization for numerical stability
    - Deadline guard
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = 115.0  # safety margin before 120s kill

        # Inverse covariance matrices V = A^{-1} for each arm
        # Start with slight ridge for numerical stability
        ridge = 0.1
        self.V = [np.eye(dim) * (1.0 + ridge) for _ in range(n_arms)]
        # Weight vectors theta for each arm
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        
        # Base forgetting factor - more aggressive for non-stationarity
        self.base_gamma = 0.86
        # Exploration coefficient
        self.base_alpha = 1.4
        # Track prediction errors for adaptive forgetting
        self.pred_errors = [0.0] * n_arms
        self.error_decay = 0.95
        # Store b vectors for theta update
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Count steps to adapt alpha
        self.step_count = 0
        self.warmup_steps = 40

    def _check_deadline(self):
        return (time.time() - self.start_time) > self.deadline

    def _adaptive_gamma(self, arm, pred_error):
        """Adapt forgetting rate based on prediction error."""
        # Update error tracker
        self.pred_errors[arm] = self.error_decay * self.pred_errors[arm] + (1 - self.error_decay) * abs(pred_error)
        
        # If prediction error is high, forget faster
        # Normalize error to [0, 1] roughly
        err_norm = min(self.pred_errors[arm] / 3.5, 1.0)
        # gamma between 0.78 (fast forget) and 0.90 (slow forget)
        gamma = 0.90 - 0.12 * err_norm
        return max(gamma, 0.78)

    def _adaptive_alpha(self):
        """Adapt exploration coefficient."""
        # Start with high exploration, decrease over time
        if self.step_count < self.warmup_steps:
            return self.base_alpha * 1.7
        else:
            # Decay alpha slightly as we get more data
            factor = 1.0 / (1.0 + self.step_count / 600.0)
            return self.base_alpha * max(factor, 0.35)

    def select(self, context):
        if self._check_deadline():
            # Return a random arm to finish quickly
            return int(np.random.randint(0, self.n))
        
        self.step_count += 1
        alpha = self._adaptive_alpha()
        
        scores = []
        for a in range(self.n):
            try:
                pred = self.theta[a] @ context
                # UCB bonus
                var = context @ self.V[a] @ context
                bonus = alpha * np.sqrt(var)
                scores.append(pred + bonus)
            except np.linalg.LinAlgError:
                scores.append(np.random.randn())
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        if self._check_deadline():
            return
        
        # Compute prediction error for adaptive forgetting
        pred = self.theta[arm] @ context
        error = reward - pred
        
        # Adaptive forgetting factor
        gamma = self._adaptive_gamma(arm, error)
        
        # Update inverse covariance using Sherman-Morrison-Woodbury with decay
        # V_new = (gamma * A_old + xx^T)^{-1}
        # = (gamma * V_old^{-1} + xx^T)^{-1}
        # Using SMW: (A + uv^T)^{-1} = A^{-1} - A^{-1} u v^T A^{-1} / (1 + v^T A^{-1} u)
        # Here A = gamma * V_old^{-1}, so A^{-1} = (1/gamma) * V_old
        # But we want to update V directly.
        
        # Let's use the standard recursive update for discounted least squares.
        # V_t = (gamma * V_{t-1}^{-1} + x x^T)^{-1}
        # V_t = V_{t-1} / gamma - (V_{t-1} x x^T V_{t-1} / gamma) / (gamma + x^T V_{t-1} x / gamma)
        # Wait, let's derive carefully.
        # A_t = gamma * A_{t-1} + x x^T
        # V_t = A_t^{-1} = (gamma * A_{t-1} + x x^T)^{-1}
        # = (gamma (A_{t-1} + (1/gamma) x x^T))^{-1}
        # = (1/gamma) (A_{t-1} + (1/gamma) x x^T)^{-1}
        # Let B = A_{t-1}, c = 1/gamma.
        # (B + c x x^T)^{-1} = B^{-1} - c B^{-1} x x^T B^{-1} / (1 + c x^T B^{-1} x)
        # = V_{t-1} - (1/gamma) V_{t-1} x x^T V_{t-1} / (1 + (1/gamma) x^T V_{t-1} x)
        # So V_t = (1/gamma) * [ V_{t-1} - (1/gamma) V_{t-1} x x^T V_{t-1} / (1 + (1/gamma) x^T V_{t-1} x) ]
        # V_t = V_{t-1} / gamma - (1/gamma^2) V_{t-1} x x^T V_{t-1} / (1 + (1/gamma) x^T V_{t-1} x)
        
        V = self.V[arm]
        x = context
        gamma = gamma
        
        # Precompute terms
        Vx = V @ x
        xVx = x @ Vx  # scalar
        
        # Denominator
        denom = 1.0 + (1.0 / gamma) * xVx
        
        # Update V
        # V_new = V / gamma - (1/gamma^2) * (Vx x^T Vx) / denom
        # Note: (Vx x^T Vx) is outer(Vx, Vx)
        
        V_new = (1.0 / gamma) * V - (1.0 / (gamma * gamma)) * np.outer(Vx, Vx) / denom
        
        self.V[arm] = V_new
        
        # Update theta
        # theta_t = A_t^{-1} (gamma * b_{t-1} + r x)
        # = V_t (gamma * b_{t-1} + r x)
        # We don't store b, we store theta.
        # b_{t-1} = A_{t-1} theta_{t-1} = V_{t-1}^{-1} theta_{t-1}
        # This is tricky without storing A or b.
        # Alternative: store A and b, and compute theta on the fly?
        # Or update b and then theta?
        # Let's store b as well for easier update.
        # Actually, the standard LinUCB update is:
        # A_t = gamma A_{t-1} + xx^T
        # b_t = gamma b_{t-1} + r x
        # theta_t = A_t^{-1} b_t
        
        # I currently store V (inverse) and theta.
        # To update theta efficiently, I can use:
        # theta_t = V_t (gamma b_{t-1} + r x)
        # But I don't have b_{t-1}.
        # Let's switch to storing A and b, and compute V and theta on the fly?
        # Or store b and update it.
        
        # Update b
        self.b[arm] = gamma * self.b[arm] + reward * context
        
        # Compute new theta
        self.theta[arm] = V_new @ self.b[arm]
# EVOLVE-BLOCK-END
