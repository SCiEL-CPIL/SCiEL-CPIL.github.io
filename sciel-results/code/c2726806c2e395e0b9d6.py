# EVOLVE-BLOCK-START
import numpy as np


class Policy:
    """Discounted Linear Thompson Sampling for non-stationary bandits."""

    def __init__(self, n_arms, dim, gamma=0.95, sigma=1.0, ridge=1.0):
        self.n = n_arms
        self.d = dim
        self.D = dim + 1  # Augmented dimension
        self.gamma = gamma
        self.sigma = sigma
        self.ridge = ridge
        
        # Per-arm parameters for the posterior N(mu, V)
        # We store P = V^{-1} (precision) and m = P @ mu (precision-weighted mean)
        # This allows efficient updates:
        # P_new = gamma * P_old + (1/sigma^2) * x x^T
        # m_new = gamma * m_old + (1/sigma^2) * x r
        
        self.P = [np.eye(self.D) * ridge for _ in range(n_arms)]
        self.m = [np.zeros(self.D) for _ in range(n_arms)]
        
        self.t = 0

    def _augment(self, context):
        context = np.asarray(context, dtype=float).ravel()
        x_aug = np.zeros(self.D)
        x_aug[0] = 1.0
        x_aug[1:] = context
        return x_aug

    def select(self, context):
        x_aug = self._augment(context)
        
        # Sample theta from posterior for each arm
        thetas = []
        for a in range(self.n):
            # Compute V = P^{-1} and mu = V @ m
            try:
                V = np.linalg.inv(self.P[a])
            except np.linalg.LinAlgError:
                V = np.linalg.pinv(self.P[a])
            
            mu = V @ self.m[a]
            
            # Sample from N(mu, V)
            # Ensure V is symmetric positive semi-definite for sampling
            # multivariate_normal can be unstable if V is ill-conditioned
            # We can add a small jitter if needed, but P should be well-conditioned due to ridge
            try:
                theta_sample = np.random.multivariate_normal(mu, V)
            except np.linalg.LinAlgError:
                # Fallback to mean if sampling fails
                theta_sample = mu
                
            thetas.append(theta_sample)
        
        scores = np.array([t @ x_aug for t in thetas])
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        x_aug = self._augment(context)
        reward = float(reward)
        
        # Update precision and precision-weighted mean
        # P_new = gamma * P_old + (1/sigma^2) * x x^T
        # m_new = gamma * m_old + (1/sigma^2) * x r
        
        inv_sigma2 = 1.0 / (self.sigma * self.sigma)
        
        self.P[arm] = self.gamma * self.P[arm] + inv_sigma2 * np.outer(x_aug, x_aug)
        self.m[arm] = self.gamma * self.m[arm] + inv_sigma2 * x_aug * reward
        
        self.t += 1
# EVOLVE-BLOCK-END
