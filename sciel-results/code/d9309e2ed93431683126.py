import numpy as np
import time

class Policy:
    def __init__(self, n_arms, dim):
        self.n_arms = n_arms
        self.dim = dim
        
        # Factorized model:
        # Global latent state z (shared across arms)
        # Arm-specific loading matrices W_a
        # Reward model: r = (W_a @ z) @ x + noise
        
        # Initialize global latent state estimator
        # z is estimated via a global RLS-like update
        self.z = np.zeros(dim)  # Current estimate of latent state
        self.P_z = np.eye(dim) * 1.0  # Covariance of z estimate
        
        # Initialize arm-specific loadings
        # W_a: dim x dim matrix
        # We'll use a simple initialization
        self.W = [np.eye(dim) for _ in range(n_arms)]
        
        # Forgetting factors
        self.gamma_global = 0.95  # Forgetting for global state
        self.gamma_arm = 0.98     # Forgetting for arm loadings
        
        # Exploration
        self.alpha = 1.5
        self.alpha_decay = 0.0005
        self.alpha_min = 0.3
        
        # Time tracking
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 145.0
        
        # Regularization
        self.reg = 1.0

    def _solve(self, A, b):
        """Solve linear system with fallback."""
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, b, rcond=None)[0]

    def _get_inverse_solve(self, A, vec):
        """Solve A x = vec."""
        try:
            return np.linalg.solve(A, vec)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, vec, rcond=None)[0]

    def select(self, context):
        # Deadline check
        if time.time() > self.deadline:
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n_arms):
                theta = self.W[a] @ self.z
                mu = theta @ context
                if mu > best_score:
                    best_score = mu
                    best_arm = a
            return best_arm

        self.t += 1
        context = np.asarray(context, dtype=np.float64)
        
        # Adaptive Alpha
        alpha = max(self.alpha_min, self.alpha * np.exp(-self.alpha_decay * self.t))
        
        best_arm = 0
        best_score = -np.inf
        
        for a in range(self.n_arms):
            # Effective weight vector for arm a
            theta = self.W[a] @ self.z
            mu = theta @ context
            
            # Uncertainty estimation
            # Approximate variance as x^T (W_a^T P_z W_a) x
            # This is computationally expensive, so we'll use a simpler approximation
            # based on the norm of the effective weight and the trace of P_z
            P_trace = np.trace(self.P_z)
            theta_norm = np.linalg.norm(theta)
            v = P_trace * (theta @ context) ** 2 / (theta_norm ** 2 + 1e-6)
            
            # UCB Score
            score = mu + alpha * np.sqrt(max(v, 0.0))
            
            if score > best_score:
                best_score = score
                best_arm = a
                
        return best_arm

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        
        # 1. Compute prediction
        theta = self.W[arm] @ self.z
        mu_pred = theta @ context
        residual = reward - mu_pred
        
        # 2. Update global latent state z
        # The observation is: reward = (W_a @ z) @ x + noise
        # This can be rewritten as: reward = z @ (W_a^T @ x) + noise
        # So we can treat (W_a^T @ x) as the "context" for z
        
        # Compute effective context for z
        x_eff = self.W[arm].T @ context
        
        # RLS update for z with forgetting
        # P = (1/gamma) * (P - P @ x_eff @ (gamma + x_eff^T @ P @ x_eff)^-1 @ x_eff^T @ P)
        # z = z + P @ x_eff * (reward - x_eff^T @ z) / (gamma + x_eff^T @ P @ x_eff)
        
        gamma = self.gamma_global
        Px = self.P_z @ x_eff
        denom = gamma + x_eff @ Px
        
        # Update covariance
        self.P_z = (self.P_z - np.outer(Px, x_eff) / denom) / gamma
        self.P_z += 1e-8 * np.eye(self.dim)  # Regularization
        
        # Update state estimate
        self.z = self.z + Px * residual / denom
        
        # 3. Update arm-specific loading W_a
        # This is trickier. We need to learn W_a such that (W_a @ z) @ x ≈ reward
        # Let's use a simple gradient-based update
        # Loss: (reward - (W_a @ z) @ x)^2
        # Gradient wrt W_a: -2 * residual * outer(z, x)
        
        # Simple SGD-like update
        lr = 0.01
        grad = -2.0 * residual * np.outer(self.z, context)
        self.W[arm] = self.W[arm] + lr * grad
        
        # Apply forgetting to W_a
        self.W[arm] = self.gamma_arm * self.W[arm] + (1 - self.gamma_arm) * np.eye(self.dim) * 0.1
