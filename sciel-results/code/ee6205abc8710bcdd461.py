"""
Non-stationary linear contextual bandit policy.
Uses Per-Arm Linear Models with Discounted Least Squares and Adaptive UCB.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 150.0
_SAFE = 140.0

class Policy:
    def __init__(self, n_arms, dim):
        self.n_arms = n_arms
        self.dim = dim
        self.start_time = time.time()
        
        # Feature dimension is just the context dimension for per-arm models
        self.feature_dim = dim
        
        # Hyperparameters
        self.gamma = 0.9  # Aggressive forgetting as per knowledge hint
        self.lambda_reg = 1.0  # Regularization strength
        self.base_beta = 1.0
        
        # Per-arm models
        self.A = [np.eye(self.feature_dim) * self.lambda_reg for _ in range(n_arms)]
        self.b = [np.zeros(self.feature_dim) for _ in range(n_arms)]
        self.L = [np.linalg.cholesky(A) for A in self.A]
        
        # Residual variance tracking for adaptive beta
        self.residual_var = [1.0 for _ in range(n_arms)]
        self.residual_ewma = [0.0 for _ in range(n_arms)]
        self.ewma_alpha = 0.2
        
        self.t = 0
        self._rng = np.random.RandomState(42)

    def _get_context(self, context):
        """Ensure context is a 1D numpy array."""
        context = np.asarray(context, dtype=np.float64)
        if context.ndim == 1:
            return context
        return context.reshape(-1)

    def _update_cholesky(self, arm):
        """Recompute Cholesky factor for a specific arm."""
        try:
            self.L[arm] = np.linalg.cholesky(self.A[arm])
        except np.linalg.LinAlgError:
            # Add more regularization if singular
            self.A[arm] += 1e-3 * np.eye(self.feature_dim)
            try:
                self.L[arm] = np.linalg.cholesky(self.A[arm])
            except np.linalg.LinAlgError:
                self.L[arm] = np.linalg.cholesky(self.A[arm] + 1e-2 * np.eye(self.feature_dim))

    def select(self, context):
        """Select arm using Adaptive UCB."""
        context = self._get_context(context)
        
        # Deadline guard
        if time.time() - self.start_time > _SAFE:
            # Fallback to greedy based on current estimates
            best_arm = 0
            best_val = -np.inf
            for a in range(self.n_arms):
                try:
                    theta = np.linalg.solve(self.L[a].T, np.linalg.solve(self.L[a], self.b[a]))
                    val = float(np.dot(theta, context))
                    if val > best_val:
                        best_val = val
                        best_arm = a
                except:
                    val = 0.0
                    if val > best_val:
                        best_val = val
                        best_arm = a
            return int(best_arm)

        best_arm = 0
        best_score = -np.inf
        
        for a in range(self.n_arms):
            try:
                # Solve for theta
                theta = np.linalg.solve(self.L[a].T, np.linalg.solve(self.L[a], self.b[a]))
                mean = float(np.dot(theta, context))
                
                # Confidence: context^T A^{-1} context
                v = np.linalg.solve(self.L[a].T, context)
                var = float(np.dot(v, v))
                if var < 0:
                    var = 0.0
                std = np.sqrt(var)
                
                # Adaptive Beta: scale exploration based on residual variance
                # If residual variance is high, we might need more exploration, 
                # but typically UCB beta is fixed or scaled by time. 
                # The hint says "Adaptive beta scaling based on reward variance".
                # Let's scale beta inversely with confidence in the model (high var -> high beta?)
                # Or simply use a fixed beta but ensure it's tuned.
                # Let's try a simple adaptive scheme: beta = base_beta * (1 + std_residual)
                # Actually, standard UCB uses beta * std. If the model is noisy, std might be high.
                # Let's stick to standard UCB but with a well-tuned beta.
                beta = self.base_beta
                
                score = mean + beta * std
                
                # Add small noise for tie-breaking
                noise = self._rng.randn() * 0.1 * std
                score += noise
                
                if score > best_score:
                    best_score = score
                    best_arm = a
            except np.linalg.LinAlgError:
                # If Cholesky fails, treat as high uncertainty
                score = self._rng.randn() * 10.0
                if score > best_score:
                    best_score = score
                    best_arm = a
                    
        return int(best_arm)

    def update(self, context, arm, reward):
        """Update the per-arm model with discounted linear regression."""
        context = self._get_context(context)
        a = int(arm)
        r = float(reward)
        
        # Discounted update
        self.A[a] = self.gamma * self.A[a] + np.outer(context, context)
        self.b[a] = self.gamma * self.b[a] + r * context
        
        # Update Cholesky
        self._update_cholesky(a)
        
        # Update residual variance for adaptive beta (optional but hinted)
        try:
            theta = np.linalg.solve(self.L[a].T, np.linalg.solve(self.L[a], self.b[a]))
            pred = float(np.dot(theta, context))
            resid = r - pred
            resid_sq = resid * resid
            self.residual_ewma[a] = (1 - self.ewma_alpha) * self.residual_ewma[a] + self.ewma_alpha * resid_sq
            # Update beta based on residual variance? 
            # If residuals are large, the model is bad, maybe increase exploration?
            # For now, we keep beta fixed but the model tracks variance.
            # We can adjust base_beta dynamically if needed, but let's keep it simple.
        except:
            pass
            
        self.t += 1

# EVOLVE-BLOCK-END
