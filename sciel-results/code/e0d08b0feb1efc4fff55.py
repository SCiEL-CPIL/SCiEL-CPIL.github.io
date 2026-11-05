# EVOLVE-BLOCK-START
import numpy as np


class Policy:
    """Adaptive Exponentially Weighted Linear Bandit."""

    def __init__(self, n_arms, dim, alpha_min=0.85, alpha_max=0.98, ridge=1e-2, beta0=1.5):
        self.n = n_arms
        self.d = dim
        self.D = dim + 1  # Augmented dimension (intercept + context)
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.ridge = ridge
        self.beta0 = beta0
        
        # Per-arm statistics
        self.A = [np.eye(self.D) * ridge for _ in range(n_arms)]
        self.b = [np.zeros(self.D) for _ in range(n_arms)]
        
        # Adaptive alpha state
        self.alpha = 0.95
        self.error_ma = 0.0
        
        self.t = 0

    def _augment(self, context):
        context = np.asarray(context, dtype=float).ravel()
        x_aug = np.zeros(self.D)
        x_aug[0] = 1.0
        x_aug[1:] = context
        return x_aug

    def select(self, context):
        x_aug = self._augment(context)
        scores = np.zeros(self.n)
        
        beta = self.beta0
        
        for a in range(self.n):
            try:
                A_inv = np.linalg.inv(self.A[a])
            except np.linalg.LinAlgError:
                A_inv = np.linalg.pinv(self.A[a])
            
            theta = A_inv @ self.b[a]
            mu = theta @ x_aug
            
            v = x_aug @ A_inv @ x_aug
            bonus = beta * np.sqrt(max(v, 0.0))
            
            scores[a] = mu + bonus
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        x_aug = self._augment(context)
        reward = float(reward)
        
        # Calculate prediction error for the selected arm to adapt alpha
        # We need the inverse of A[arm] to get theta. 
        # To avoid double inversion cost if we cached it, we can just compute it here.
        # Since D is small, this is cheap.
        try:
            A_inv_arm = np.linalg.inv(self.A[arm])
        except np.linalg.LinAlgError:
            A_inv_arm = np.linalg.pinv(self.A[arm])
        
        theta_arm = A_inv_arm @ self.b[arm]
        pred = theta_arm @ x_aug
        error = reward - pred
        abs_error = abs(error)
        
        # Update moving average of error
        self.error_ma = 0.9 * self.error_ma + 0.1 * abs_error
        
        # Adapt alpha:
        # If error is high, we want to forget faster (lower alpha).
        # If error is low, we want to remember more (higher alpha).
        # Map error to [alpha_min, alpha_max]
        # Threshold: if error > threshold, push alpha down.
        threshold = 1.0  # Tune this if needed
        if abs_error > threshold:
            # Decrease alpha
            self.alpha = max(self.alpha_min, self.alpha - 0.01)
        else:
            # Increase alpha slowly
            self.alpha = min(self.alpha_max, self.alpha + 0.005)
            
        # Decay existing statistics with current adaptive alpha
        self.A[arm] *= self.alpha
        self.b[arm] *= self.alpha
        
        # Add new observation
        self.A[arm] += np.outer(x_aug, x_aug)
        self.b[arm] += x_aug * reward
        
        self.t += 1
# EVOLVE-BLOCK-END
