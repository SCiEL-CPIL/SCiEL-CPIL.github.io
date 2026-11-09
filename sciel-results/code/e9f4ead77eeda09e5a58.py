# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """Discounted Linear Bandit (LinUCB) for non-stationary environments."""
    
    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.dim = dim
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 140.0
        
        # Discount factor: controls how fast old information is forgotten.
        # Closer to 1.0 means longer memory. 0.99 is a common starting point.
        self.discount = 0.99
        
        # Regularization
        self.reg = 1.0
        
        # Initialize A_inv and b for each arm
        # A is the weighted sum of xx^T, b is the weighted sum of r*x
        self.A_inv = [np.eye(self.dim) / self.reg for _ in range(n_arms)]
        self.b = [np.zeros(self.dim) for _ in range(n_arms)]
        
        # Exploration parameter
        self.alpha = 1.5
        
    def _select_arm(self, context):
        """Select arm using UCB with discounted estimates."""
        self.t += 1
        x = np.asarray(context, dtype=np.float64)
        
        scores = []
        for a in range(self.n):
            A_inv = self.A_inv[a]
            b = self.b[a]
            
            # theta = A_inv @ b
            theta = A_inv @ b
            mu = theta @ x
            
            # Compute variance term: x^T A_inv x
            v = x @ A_inv @ x
            
            # UCB score
            score = mu + self.alpha * np.sqrt(max(v, 0.0))
            scores.append(score)
        
        best_arm = int(np.argmax(scores))
        return best_arm
    
    def select(self, context):
        # Deadline guard
        if time.time() > self.deadline:
            return 0
        
        arm = self._select_arm(context)
        return arm
    
    def update(self, context, arm, reward):
        x = np.asarray(context, dtype=np.float64)
        
        # Discount the current A_inv and b
        # A_new = discount * A_old + x x^T
        # This is equivalent to maintaining A_inv_new for A_new
        # Let A_old_inv be the current self.A_inv[arm]
        # We need to update it to reflect A_new = discount * A_old + x x^T
        
        # First, discount the existing inverse?
        # It is numerically unstable to simply discount A_inv.
        # Instead, we can use the Woodbury identity on the updated system.
        # However, discounting A is easier if we store A directly, but A is dense and large.
        # Standard approach for Discounted LinUCB:
        # We maintain A_inv.
        # A_t = discount * A_{t-1} + x_t x_t^T
        # We want A_t^{-1}.
        
        # Let A_old = A_{t-1}. A_old_inv = self.A_inv[arm].
        # A_t = discount * A_old + x x^T
        # This is not a simple rank-1 update on A_old.
        
        # Alternative: Store A directly? For small dim, yes.
        # Let's switch to storing A and inverting periodically or using Cholesky?
        # Or use the property:
        # If we had A_old, A_t = discount * A_old + x x^T.
        # We can compute A_t^{-1} using Woodbury if we can express it as (M + u v^T)^{-1}.
        # Here M = discount * A_old.
        # M^{-1} = (1/discount) * A_old^{-1} = (1/discount) * A_old_inv.
        
        # So, let M = discount * A_old.
        # M_inv = (1.0 / self.discount) * self.A_inv[arm]
        
        # Now apply Woodbury for + x x^T:
        # (M + x x^T)^{-1} = M^{-1} - (M^{-1} x x^T M^{-1}) / (1 + x^T M^{-1} x)
        
        A_old_inv = self.A_inv[arm]
        M_inv = (1.0 / self.discount) * A_old_inv
        
        M_inv_x = M_inv @ x
        denom = 1.0 + x @ M_inv_x
        
        A_inv_new = M_inv - np.outer(M_inv_x, M_inv_x) / denom
        
        # Update b
        # b_t = discount * b_{t-1} + reward * x
        b_old = self.b[arm]
        b_new = self.discount * b_old + reward * x
        
        self.A_inv[arm] = A_inv_new
        self.b[arm] = b_new

# EVOLVE-BLOCK-END
