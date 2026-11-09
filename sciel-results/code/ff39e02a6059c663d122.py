import numpy as np
import time

class Policy:
    def __init__(self, n_arms, dim):
        self.n_arms = n_arms
        self.dim = dim
        
        # Factorized Model Parameters
        # Global latent state z: dim vector
        # Local deviations delta_a: dim vector for each arm
        # Arm-specific loading W_a: dim x dim matrix (fixed, initially identity)
        
        # Initialize global state estimator
        # We use RLS formulation: P_z^{-1} and b_z
        # P_z is the covariance matrix of z estimate
        self.P_z = np.eye(dim) * 10.0  # Large initial uncertainty
        self.b_z = np.zeros(dim)
        
        # Initialize local deviation estimators for each arm
        self.P_delta = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        self.b_delta = [np.zeros(dim) for _ in range(n_arms)]
        
        # Fixed loading matrices W_a
        # To allow for some flexibility, we can start with identity 
        # or a random orthogonal matrix. Identity is safest for initialization.
        self.W = [np.eye(dim) for _ in range(n_arms)]
        
        # Forgetting factors
        # Global state should adapt faster to regime changes
        self.lambda_z = 0.95
        # Local deviations should adapt slower to capture stable arm properties
        self.lambda_delta = 0.99
        
        # Exploration parameters
        self.alpha_start = 3.0
        self.alpha_min = 0.5
        self.decay_rate = 0.0005
        
        # Time tracking
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 145.0 # 5s margin
        
        # Reset threshold
        self.reset_threshold = 3.0
        
        # Track residual errors for adaptive lambda if needed
        self.err_ema_z = 0.0
        self.err_ema_delta = [0.0 for _ in range(n_arms)]
        self.err_alpha = 0.1

    def _solve(self, P_inv, b):
        """Solve P_inv @ x = b. P_inv is the inverse covariance."""
        try:
            return np.linalg.solve(P_inv, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(P_inv, b, rcond=None)[0]

    def _get_inverse_solve(self, P_inv, vec):
        """Solve P_inv x = vec."""
        try:
            return np.linalg.solve(P_inv, vec)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(P_inv, vec, rcond=None)[0]

    def _predict(self, context, arm):
        """
        Predict mean reward and variance for a given arm and context.
        Model: r = (W_a @ z + delta_a) @ x + noise
        Let theta_a = W_a @ z + delta_a.
        Mean = theta_a @ x = (W_a @ z) @ x + delta_a @ x
        Variance = x^T (W_a^T Cov(z) W_a + Cov(delta_a)) x
        """
        context = np.asarray(context, dtype=np.float64)
        
        # Estimate z and delta_a
        z_hat = self._solve(np.linalg.inv(self.P_z), self.b_z)
        delta_hat = self._solve(np.linalg.inv(self.P_delta[arm]), self.b_delta[arm])
        
        # Mean prediction
        # theta_a = W_a @ z_hat + delta_hat
        theta_hat = self.W[arm] @ z_hat + delta_hat
        mu = theta_hat @ context
        
        # Variance calculation
        # Cov(theta_a) = W_a Cov(z) W_a^T + Cov(delta_a)
        # Note: P_z is Cov(z), P_delta is Cov(delta)
        Cov_z = self.P_z
        Cov_delta = self.P_delta[arm]
        
        # We need x^T Cov(theta_a) x
        # = x^T (W_a Cov_z W_a^T + Cov_delta) x
        # = (W_a^T x)^T Cov_z (W_a^T x) + x^T Cov_delta x
        
        W_T_x = self.W[arm].T @ context
        var_z_part = W_T_x @ Cov_z @ W_T_x
        var_delta_part = context @ Cov_delta @ context
        
        total_var = var_z_part + var_delta_part
        
        return mu, max(total_var, 0.0)

    def select(self, context):
        # Deadline check
        if time.time() > self.deadline:
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n_arms):
                mu, var = self._predict(context, a)
                # Use a fixed alpha for safety in deadline mode
                score = mu + 1.0 * np.sqrt(var)
                if score > best_score:
                    best_score = score
                    best_arm = a
            return best_arm

        self.t += 1
        context = np.asarray(context, dtype=np.float64)
        
        # Adaptive Alpha
        alpha = max(self.alpha_min, self.alpha_start * np.exp(-self.decay_rate * self.t))
        
        best_arm = 0
        best_score = -np.inf
        
        for a in range(self.n_arms):
            mu, var = self._predict(context, a)
            score = mu + alpha * np.sqrt(var)
            
            if score > best_score:
                best_score = score
                best_arm = a
                
        return best_arm

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        
        # 1. Update Global State z
        # The observation for z is derived from the residual after removing local effects?
        # Actually, in a factorized model, we can update z and delta jointly or sequentially.
        # A common approach is to treat the total theta_a as the target.
        # However, to share information, we want to update z using ALL data.
        
        # Let's use a sequential update strategy for stability:
        # First, update the local deviation delta_a assuming z is fixed.
        # Then, update the global state z using the residual error of the global model.
        
        # Current estimates
        z_hat = self._solve(np.linalg.inv(self.P_z), self.b_z)
        delta_hat = self._solve(np.linalg.inv(self.P_delta[arm]), self.b_delta[arm])
        
        # Predicted reward using current estimates
        mu_pred = (self.W[arm] @ z_hat + delta_hat) @ context
        
        # Residual
        residual = reward - mu_pred
        
        # Update Error EMAs
        self.err_ema_z = (1 - self.err_alpha) * self.err_ema_z + self.err_alpha * (residual ** 2)
        self.err_ema_delta[arm] = (1 - self.err_alpha) * self.err_ema_delta[arm] + self.err_alpha * (residual ** 2)
        
        # Shock Detection: If residual is large, reset local deviation
        # We need a scale for residual. Use total variance.
        _, var_total = self._predict(context, arm)
        std_pred = np.sqrt(var_total) + 1.0
        
        if abs(residual) > self.reset_threshold * std_pred:
            # Reset local deviation for this arm
            self.P_delta[arm] = np.eye(self.dim) * 10.0
            self.b_delta[arm] = np.zeros(self.dim)
            # Do NOT reset global z, as the shock might be local or global.
            # If it's global, z will adapt via the update below.
            # If it's local, resetting delta helps.
            pass

        # --- Update Local Deviation delta_a ---
        # Model: r = (W_a z + delta_a) x + noise
        # We can view this as: r = delta_a x + (W_a z) x + noise
        # Let c_a = W_a x. Then r = delta_a x + c_a z + noise.
        # This is a linear model in parameters [delta_a, z].
        # But we want to update them separately to maintain factorization.
        
        # Strategy: 
        # 1. Update delta_a using the residual from the global model?
        #    No, that's circular.
        
        # Better Strategy: Joint Update via Block Matrix?
        # Too expensive.
        
        # Alternative Strategy: 
        # Update z first, then delta_a?
        # Or update delta_a first, then z?
        
        # Let's try:
        # 1. Update delta_a using the assumption that z is known (z_hat).
        #    Residual for delta_a: r - (W_a z_hat) x
        #    This is a standard RLS update for delta_a with target (r - (W_a z_hat) x).
        
        target_delta = reward - (self.W[arm] @ z_hat) @ context
        
        # RLS Update for delta_a
        # P_delta^{-1} <- lambda_delta * P_delta^{-1} + x x^T
        # b_delta <- lambda_delta * b_delta + x * target_delta
        
        P_inv_delta = np.linalg.inv(self.P_delta[arm])
        P_inv_delta = self.lambda_delta * P_inv_delta + np.outer(context, context)
        self.b_delta[arm] = self.lambda_delta * self.b_delta[arm] + context * target_delta
        self.P_delta[arm] = np.linalg.inv(P_inv_delta)
        
        # 2. Update z using the residual from the local model?
        #    Residual for z: r - (delta_a_new) x
        #    This is a standard RLS update for z with target (r - delta_a_new x).
        #    But we have multiple arms. We can update z using the data from this arm.
        
        delta_hat_new = self._solve(np.linalg.inv(self.P_delta[arm]), self.b_delta[arm])
        target_z = reward - delta_hat_new @ context
        
        # RLS Update for z
        # P_z^{-1} <- lambda_z * P_z^{-1} + (W_a x) (W_a x)^T
        # b_z <- lambda_z * b_z + (W_a x) * target_z
        
        W_x = self.W[arm] @ context
        P_inv_z = np.linalg.inv(self.P_z)
        P_inv_z = self.lambda_z * P_inv_z + np.outer(W_x, W_x)
        self.b_z = self.lambda_z * self.b_z + W_x * target_z
        self.P_z = np.linalg.inv(P_inv_z)
        
        # Note: This sequential update is an approximation of the joint MLE.
        # It allows information to flow:
        # - Arm selection provides info about delta_a.
        # - Arm selection provides info about z (via residual after delta_a).
        # - z update improves predictions for ALL arms.
        
        # Optimization: Inverting matrices is O(d^3). 
        # For small d (e.g., 5-10), this is fine.
        # If d is large, we should use matrix inversion lemma.
        # Given the context of "five-arm" and typical bandit dims, d is likely small.
