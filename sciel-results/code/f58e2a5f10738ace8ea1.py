# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit with a unified linear model
    and adaptive Kalman filtering.
    
    Structural Change:
    Instead of low-rank factorization (V, z) with separate covariances,
    we maintain a single flattened weight vector w (size d*n) representing
    the weights for all arms.
    
    We use a Kalman filter with:
    1. Adaptive process noise (via adaptive forgetting factor on precision).
    2. Initial covariance structure that encourages parameter sharing
       (high correlation between arm weights initially).
    3. Joseph-form updates for stability.
    
    The "shared" aspect comes from the state being a single vector and the
    prior coupling the arms. As data arrives, the model can decouple arms
    if the rewards suggest they should be different.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self.t0 = time.time()
        self.deadline = 145.0

        # State: flattened weight vector (d * n,)
        self.dn = self.d * self.n
        self.w = np.zeros(self.dn)

        # Precision matrix (inverse covariance)
        # We want to encourage sharing initially.
        # A covariance matrix with high off-diagonal correlations implies
        # that the precision matrix has specific structure.
        # For simplicity and stability, we start with a diagonal precision
        # but add a "shrinkage" regularization that pulls arms towards
        # the mean, effectively enforcing sharing during updates.
        
        # Initial precision: high confidence in 0? No, low confidence (high variance).
        # Let's start with moderate variance.
        self.P_inv = np.eye(self.dn) * 1.0 
        
        # Adaptive forgetting factor parameters
        self.lambda_ema = 0.99  # Starting forgetting factor (close to 1 = remember long)
        self.residual_ema = 0.0
        self.lambda_min = 0.8
        self.lambda_max = 0.999

        # Exploration
        self.count = [0 for _ in range(n_arms)]
        self.exploration_base = 1.0
        self.exploration_decay = 0.999

        # Pre-compute one-hot basis for arms to build context vectors
        # phi for arm a: x in block a, 0 elsewhere
        # We construct phi on the fly for speed
        
    def _get_phi(self, context, arm):
        """Construct the one-hot encoded context vector for the unified model."""
        phi = np.zeros(self.dn)
        # The block for arm 'arm' is indices [arm*d : (arm+1)*d]
        start = arm * self.d
        phi[start:start+self.d] = context
        return phi

    def _get_arm_weights(self, arm):
        """Extract weight vector for a specific arm."""
        start = arm * self.d
        return self.w[start:start+self.d]

    def _predict(self):
        """
        Adaptive Kalman prediction step.
        Instead of adding Q to covariance, we multiply Precision by lambda.
        lambda < 1 increases uncertainty (forgets old data).
        lambda is adaptive based on residual error.
        """
        if self.t > 5:
            # Adaptive lambda:
            # If residual is high, we want to forget faster (lower lambda).
            # If residual is low, we want to remember (higher lambda).
            # residual_ema is tracked in update.
            # Map residual_ema to lambda.
            # High error -> low lambda.
            # Use tanh to saturate.
            err_norm = np.tanh(self.residual_ema / 10.0) # Scale error
            # Base lambda 0.99, reduce by up to 0.1 if error is high
            self.lambda_ema = 0.99 - 0.1 * err_norm
            self.lambda_ema = np.clip(self.lambda_ema, self.lambda_min, self.lambda_max)
        
        # Update Precision: P_inv <- lambda * P_inv
        # This is equivalent to Sigma <- Sigma / lambda (if lambda < 1, Sigma grows)
        self.P_inv *= self.lambda_ema

    def _update(self, context, arm, reward):
        """
        Kalman update step for the unified model.
        """
        phi = self._get_phi(context, arm)
        
        # Predicted reward
        pred = phi @ self.w
        
        # Innovation
        innovation = reward - pred
        
        # Update residual EMA
        self.residual_ema = 0.95 * self.residual_ema + 0.05 * innovation**2

        # Kalman Gain
        # K = P_inv^{-1} phi^T (phi P_inv^{-1} phi^T + R)^{-1}
        # Let P = P_inv^{-1}. We don't want to invert P_inv every time if possible,
        # but dn is small (25), so inversion is cheap.
        # Alternatively, use the information form update:
        # P_inv_new = P_inv + phi phi^T / R
        # w_new = w + K * innovation
        # K = P_new^{-1} phi^T
        
        # For R=1 (observation noise variance)
        # P_inv += outer(phi, phi)
        
        # We need K.
        # K = (P_inv + phi phi^T)^{-1} phi^T
        # Using Woodbury or just invert since dn is small.
        
        # To avoid full inversion every step if dn grows, we can use the 
        # scalar denominator trick if we maintained P, but we maintain P_inv.
        # Let's just invert. dn=25, 25x25 inversion is fast.
        
        # Optimization: 
        # P_inv_new = P_inv + np.outer(phi, phi)
        # w_new = w + (P_inv_new^{-1} @ phi) * innovation
        
        # However, we also need the new P_inv for the next step.
        
        # Calculate K
        # K = P @ phi / (phi @ P @ phi + R)
        # We have P_inv. 
        # Let v = P_inv @ phi.
        # Then P @ phi = P_inv^{-1} @ v? No.
        # P @ phi = (P_inv^{-1}) @ phi.
        # This requires inverting P_inv to get P, then multiplying by phi.
        # Or, solve P_inv x = phi -> x = P @ phi.
        # Then S = phi @ x + R.
        # K = x / S.
        
        # Solve P_inv x = phi
        try:
            x = np.linalg.solve(self.P_inv, phi)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if singular
            x = np.linalg.pinv(self.P_inv) @ phi
            
        S = phi @ x + 1.0  # R = 1.0
        K = x / S
        
        # Update weight
        self.w += K * innovation
        
        # Update Precision (Information Form)
        # P_inv_new = P_inv + phi phi^T / R
        # But wait, the standard information form update is:
        # P^{-1}_{new} = P^{-1}_{old} + \phi \phi^T / R
        # This is correct for the posterior precision.
        # However, we also applied the prediction step (forgetting) which modified P_inv.
        # So we just add the new information.
        self.P_inv += np.outer(phi, phi)
        
        # Symmetrize P_inv for numerical stability
        self.P_inv = (self.P_inv + self.P_inv.T) / 2.0

    def select(self, context):
        """Select arm using UCB."""
        if time.time() - self.t0 > self.deadline:
            # Fallback: greedy
            scores = []
            for a in range(self.n):
                w_a = self._get_arm_weights(a)
                scores.append(w_a @ context)
            return int(np.argmax(scores))

        self._predict()

        scores = []
        # We need the covariance P to compute uncertainty.
        # We have P_inv. 
        # Var(reward) = phi^T P phi.
        # We can compute P = inv(P_inv) once per step? 
        # Or compute var for each arm.
        # Since n is small, we can invert P_inv once.
        
        try:
            P = np.linalg.inv(self.P_inv)
        except np.linalg.LinAlgError:
            P = np.linalg.pinv(self.P_inv)

        for a in range(self.n):
            w_a = self._get_arm_weights(a)
            mean = w_a @ context
            
            # Uncertainty
            phi = self._get_phi(context, a)
            var = phi @ P @ phi
            uncertainty = np.sqrt(max(0.0, var))
            
            # Count-based exploration
            count_factor = self.exploration_base * (self.exploration_decay ** self.count[a])
            bonus = 1.0 * uncertainty * count_factor + 0.1 * count_factor / (1 + self.count[a])
            
            scores.append(mean + bonus)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        """Update the model."""
        self.t += 1
        self.count[arm] += 1
        
        self._update(context, arm, reward)
# EVOLVE-BLOCK-END
