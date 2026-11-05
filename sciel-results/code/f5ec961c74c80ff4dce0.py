# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Non-stationary linear contextual bandit using Bayesian Thompson Sampling
    with adaptive forgetting based on Mahalanobis distance of residuals.
    
    Key features:
    1. Thompson Sampling: Sample theta from posterior N(mu, Sigma) for each arm.
    2. Adaptive Forgetting: Decay factor beta is determined by the Mahalanobis
       distance of the latest residual. Large residuals (regime shift) trigger
       faster forgetting (lower beta).
    3. Matrix Inversion Lemma: Efficiently update posterior precision and mean.
    4. Time Guard: Ensures completion within wall-clock limits.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.lam = 1.0  # Ridge regularization / Prior precision scaling

        # Per-arm Bayesian parameters
        # Prior: theta ~ N(0, (lam * I)^-1) => precision = lam * I, mean = 0
        # We store precision (A) and b = A @ mu
        self.A = [np.eye(dim) * self.lam for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]

        # Adaptive forgetting parameters
        # We track the residual to determine beta
        # beta = 1 / (1 + k * residual_mahalanobis)
        # Or simpler: beta = exp(-k * residual_sq)
        # Let's use a robust mapping.
        # Mahalanobis distance d = sqrt(r^2 / (sigma^2 + x^T Sigma x))
        # If d is large, the observation is surprising -> regime change likely.
        
        self.forgetting_k = 0.5  # Sensitivity of forgetting to surprise
        self.beta_min = 0.70     # Max forgetting rate (fastest)
        self.beta_max = 0.99     # Min forgetting rate (slowest, almost no forgetting)
        
        # Exploration is inherent in Thompson Sampling
        # No explicit alpha needed, but we can scale the sample if needed
        
        # Step counter
        self.step = 0
        
        # Time management
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0  # Safety margin for 120s limit

    def _safe(self):
        return time.time() >= self.deadline

    def _get_beta(self, residual, context, arm):
        """
        Calculate adaptive discount factor beta based on residual surprise.
        """
        # Compute Mahalanobis distance of the residual
        # Sigma = A^-1
        # Mahalanobis^2 = r^2 / (sigma_noise^2 + x^T Sigma x)
        # We assume unit noise variance for the surprise calculation, or just use
        # the prediction variance.
        
        try:
            # Solve for Sigma @ context
            # A @ v = context => v = A^-1 context = Sigma context
            # Then x^T Sigma x = context @ v
            v = np.linalg.solve(self.A[arm], context)
            var_pred = float(context @ v)
            
            # The prediction variance is var_pred. The residual variance is 1 + var_pred.
            # Standardized residual z = r / sqrt(1 + var_pred)
            z = residual / np.sqrt(1.0 + var_pred)
            
            # Use z to determine beta
            # If z is small, beta is high (close to 1).
            # If z is large, beta is low.
            # beta = 1 / (1 + k * z^2)
            # Clamp beta
            beta = 1.0 / (1.0 + self.forgetting_k * z * z)
            beta = np.clip(beta, self.beta_min, self.beta_max)
            
        except Exception:
            # Fallback to a conservative beta
            beta = 0.95
            
        return beta

    def _sample_theta(self, a, context):
        """
        Sample theta from the posterior N(mu, Sigma) for arm a.
        We only need the score theta @ context, so we can sample the score directly
        or sample theta. Sampling theta is O(d^3) for Cholesky + solve + random.
        Sampling score: score ~ N(mu @ context, context @ Sigma @ context)
        This is O(d^2) for solve and dot product. Much faster.
        """
        try:
            # Solve A @ v = context to get Sigma @ context
            v = np.linalg.solve(self.A[a], context)
            
            # Mean prediction
            mu = np.linalg.solve(self.A[a], self.b[a])
            mean_score = float(mu @ context)
            
            # Variance of the score
            var_score = float(context @ v)
            
            # Sample from N(mean_score, var_score)
            # Use standard normal
            z = np.random.randn()
            score = mean_score + np.sqrt(max(var_score, 0.0)) * z
            
            return score
        except Exception:
            return 0.0

    def select(self, context):
        if self._safe():
            # Emergency: pick arm with highest mean prediction
            scores = []
            for a in range(self.n):
                try:
                    mu = np.linalg.solve(self.A[a], self.b[a])
                    scores.append(float(mu @ context))
                except Exception:
                    scores.append(0.0)
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            score = self._sample_theta(a, context)
            scores.append(score)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.step += 1
        
        # 1. Compute residual BEFORE updating the model
        # We need the current posterior to compute the surprise.
        try:
            mu = np.linalg.solve(self.A[arm], self.b[arm])
            pred = float(mu @ context)
            residual = reward - pred
        except Exception:
            residual = 0.0

        # 2. Calculate adaptive beta
        beta = self._get_beta(residual, context, arm)

        # 3. Update the model using Bayesian update with decay
        # Old: A_old, b_old
        # New: A_new = beta * A_old + x x^T
        #       b_new = beta * b_old + r x
        
        # Note: This is equivalent to updating the precision.
        # The mean update is implicit in b_new = A_new @ mu_new
        
        self.A[arm] = beta * self.A[arm] + np.outer(context, context)
        self.b[arm] = beta * self.b[arm] + reward * context

# EVOLVE-BLOCK-END
