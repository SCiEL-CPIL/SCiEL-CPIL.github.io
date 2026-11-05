"""Non-stationary linear contextual bandit with RLS (Forgetting Factor) and Thompson Sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Non-Stationary Linear Contextual Bandit Policy using RLS with Forgetting Factor 
    and Thompson Sampling.
    
    Strategy:
    - Per-arm Recursive Least Squares (RLS) estimator with a forgetting factor (lambda).
      This effectively maintains a sliding window of recent observations, allowing
      the model to adapt to changing arm weights without explicit process noise.
    - Thompson Sampling (TS) for exploration: samples weights from the posterior
      distribution (assumed Gaussian) to select arms. This is often more adaptive
      than UCB in non-stationary settings.
    - Deadline guard to prevent timeouts.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Initialize weight vectors (means) to zero
        self.means = [np.zeros(dim) for _ in range(n_arms)]
        
        # Initialize inverse covariance matrices (P_inv) to small identity.
        # P = (lambda * P_inv)^-1. 
        # Starting with small P_inv means large initial uncertainty (P is large).
        # alpha controls initial uncertainty.
        initial_alpha = 1e-3
        self.p_invs = [np.eye(dim) * initial_alpha for _ in range(n_arms)]
        
        # Hyperparameters
        # Forgetting factor: closer to 1 means longer memory.
        # 0.95 is a common choice for non-stationary bandits.
        self.lambda_ = 0.95
        self.measurement_noise = 0.1
        
        # Deadline guard
        self.start_time = time.time()
        self.deadline = self.start_time + 140.0  # 10s safety margin

    def _check_deadline(self):
        """Check if we are approaching the time limit."""
        return time.time() > self.deadline

    def select(self, context):
        if self._check_deadline():
            return int(np.random.randint(self.n))
        
        context = np.asarray(context, dtype=np.float64)
        
        # Thompson Sampling: Sample weights from N(mean, P) for each arm
        # P = (lambda * P_inv)^-1
        # We need to compute the covariance matrix P for sampling.
        # Since d is small (likely <= 10-20), inverting P_inv is feasible.
        # However, we can sample more efficiently using Cholesky or eigen decomposition.
        # For simplicity and robustness, we compute P directly if d is small.
        
        scores = np.zeros(self.n)
        for a in range(self.n):
            # Compute Covariance P = (lambda * P_inv)^-1
            # Note: P_inv is updated as P_inv_new = lambda * P_inv_old + x x^T / R
            # So the current effective precision matrix is P_inv.
            # The covariance is P = P_inv^-1.
            
            # To avoid full inversion every step for every arm (which is O(d^3)),
            # we can use the fact that we only need x^T P x for UCB, but for TS we need
            # to sample w ~ N(mean, P).
            # If d is small, P_inv^-1 is fast.
            try:
                P = np.linalg.inv(self.p_invs[a])
            except np.linalg.LinAlgError:
                # If singular, add small regularization
                P = np.linalg.inv(self.p_invs[a] + 1e-6 * np.eye(self.d))
            
            # Sample weights
            # w_sample = mean + L @ z, where z ~ N(0, I) and P = L L^T
            # Use Cholesky for stability
            try:
                L = np.linalg.cholesky(P)
            except np.linalg.LinAlgError:
                # Fallback to eigen if Cholesky fails (numerical issues)
                vals, vecs = np.linalg.eigh(P)
                vals = np.clip(vals, 0, None)
                L = vecs @ np.diag(np.sqrt(vals))
            
            z = np.random.randn(self.d)
            w_sample = self.means[a] + L @ z
            
            # Score is the predicted reward for this sampled weight
            scores[a] = w_sample @ context
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        if self._check_deadline():
            return
            
        context = np.asarray(context, dtype=np.float64)
        
        # RLS Update with Forgetting Factor
        # P_inv_new = lambda * P_inv_old + (x x^T) / R
        # w_new = w_old + P_new x * (reward - x^T w_old) / (R + x^T P_new x)
        # Note: Standard RLS update formulas:
        # K = P_old x / (R + x^T P_old x)  <-- This is gain using OLD P
        # w_new = w_old + K * innovation
        # P_new = (I - K x^T) P_old / lambda  <-- This maintains the forgetting
        
        # Alternatively, using the P_inv formulation which is often more stable:
        # 1. Update P_inv:
        #    P_inv_new = lambda * P_inv_old + (x x^T) / R
        # 2. Compute Gain K_new:
        #    K_new = P_new x / (R + x^T P_new x)
        #    But P_new = P_inv_new^-1. This requires inversion.
        
        # Let's stick to the standard RLS gain update which avoids inverting P_inv
        # at every step if we maintain P directly, OR we maintain P_inv and compute
        # gain via Woodbury.
        
        # Efficient RLS Update:
        # Let P be the current covariance (before update).
        # We maintain P_inv.
        # Step 1: Update P_inv
        self.p_invs[arm] = self.lambda_ * self.p_invs[arm] + np.outer(context, context) / self.measurement_noise
        
        # Step 2: Compute Gain K
        # K = P_new x / (R + x^T P_new x)
        # Note: P_new = P_inv_new^-1
        # We can compute K without full inversion using the identity:
        # K = (P_inv_new^-1 x) / (R + x^T P_inv_new^-1 x)
        # This is still O(d^3) if we invert.
        # However, d is typically small (e.g., 5-10). Inversion is acceptable.
        
        # To be safe and fast, we compute the gain using the Sherman-Morrison-Woodbury
        # or just invert if d < 50.
        try:
            P_new = np.linalg.inv(self.p_invs[arm])
        except np.linalg.LinAlgError:
            P_new = np.linalg.pinv(self.p_invs[arm])
            
        # Prediction error
        pred_mean = self.means[arm] @ context
        innovation = reward - pred_mean
        
        # Denominator: R + x^T P_new x
        denom = self.measurement_noise + context @ P_new @ context
        
        # Gain vector
        k_gain = (P_new @ context) / denom
        
        # Update Mean
        self.means[arm] += k_gain * innovation
        
        # Update Covariance P
        # P_new_updated = P_new - K x^T P_new
        # This maintains symmetry and PSD.
        px = P_new @ context
        self.p_invs[arm] = np.linalg.inv(P_new - np.outer(k_gain, px))
        
        # Alternatively, update P_inv directly in the next step?
        # No, we already updated P_inv in Step 1.
        # The standard RLS update for P is:
        # P_new = (I - K x^T) P_old / lambda
        # But we are using the P_inv form which accumulates information.
        # The formula P_inv_new = lambda P_inv_old + x x^T / R is correct for the
        # "sliding window" interpretation.
        # The gain K derived from P_new = P_inv_new^-1 is consistent with this update.
        
        # Symmetry check
        # We don't store P directly, we store P_inv.
        # The inversion in the next select() or update() handles the covariance.
        # However, to avoid numerical drift in P_inv, we can symmetrize it.
        self.p_invs[arm] = (self.p_invs[arm] + self.p_invs[arm].T) / 2.0

# EVOLVE-BLOCK-END
