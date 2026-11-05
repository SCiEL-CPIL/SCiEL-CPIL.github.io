import numpy as np
import time

class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Global Linear Bandit with Thompson Sampling
        # We assume a single global weight vector w such that reward ~ w^T x + noise
        # This is a structural change from per-arm models to a global model.
        # In many contextual bandits, the context encodes the arm (e.g., one-hot) or the arm effect is small.
        # If the context does not encode the arm, this model assumes all arms have similar underlying dynamics.
        # To be safe, we can augment the context with a one-hot encoding of the arm if we suspect arm-specific offsets,
        # but the task says "linear contextual bandit with changing arm weights".
        # Usually, this means r = x^T w_a + noise. 
        # However, if we use a global model r = x^T w + noise, we lose arm-specific bias.
        # Let's stick to the global model but use Thompson Sampling which is robust.
        # If the context already contains arm info (e.g., one-hot), this works perfectly.
        # If not, we might underfit. 
        # Alternative: Use a per-arm model but with Thompson Sampling instead of Epsilon-Greedy/Sliding Window LS.
        
        # Let's reconsider: The current program uses per-arm sliding window LS.
        # The inspirations suggest EMA is lighter. 
        # A structural change could be: **Per-arm Thompson Sampling with Exponential Forgetting**.
        # Instead of just tracking mean (b) and inverse covariance (A_inv) for point estimates (Greedy),
        # we maintain a posterior distribution N(mu, Sigma) for each arm's weights.
        # Update: mu_t, Sigma_t based on new (x, r).
        # Select: Sample theta_a ~ N(mu_a, Sigma_a) for each arm, pick argmax theta_a^T x.
        
        # This is different from the current Epsilon-Greedy + Point Estimate.
        # It is also different from simple EMA (which is point estimate).
        
        self.gamma = 0.95 # Forgetting factor
        
        # Per-arm posterior parameters:
        # mu[a]: mean vector (dim,)
        # Sigma[a]: covariance matrix (dim, dim)
        
        # Initialize with prior: mu=0, Sigma=I (or large variance)
        # Using large variance initially encourages exploration.
        self.mu = [np.zeros(dim) for _ in range(n_arms)]
        self.Sigma = [np.eye(dim) * 10.0 for _ in range(n_arms)] # High initial variance
        
        # Deadline guard
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0
        self.t = 0

    def select(self, context):
        if time.time() > self.deadline:
            # Fallback: Greedy using current means
            context_vec = np.array(context)
            scores = [self.mu[a] @ context_vec for a in range(self.n)]
            return int(np.argmax(scores))

        context_vec = np.array(context)
        
        # Thompson Sampling: Sample from posterior for each arm
        sampled_scores = []
        for a in range(self.n):
            # Sample theta ~ N(mu[a], Sigma[a])
            # Cholesky decomposition for sampling
            try:
                L = np.linalg.cholesky(self.Sigma[a])
            except np.linalg.LinAlgError:
                # If covariance is not positive definite, add small jitter
                L = np.linalg.cholesky(self.Sigma[a] + 1e-6 * np.eye(self.d))
            
            z = np.random.normal(0, 1, self.d)
            theta_sample = self.mu[a] + L @ z
            sampled_scores.append(theta_sample @ context_vec)
            
        return int(np.argmax(sampled_scores))

    def update(self, context, arm, reward):
        self.t += 1
        context_vec = np.array(context)
        
        # Update posterior for the selected arm
        # Bayesian Linear Regression Update with Forgetting Factor
        # Prior: theta ~ N(mu_0, Sigma_0)
        # Likelihood: r | theta, x ~ N(x^T theta, sigma_r^2)
        
        # We use a forgetting factor gamma to discount old information.
        # Effective prior for update:
        # mu_prev = self.mu[arm]
        # Sigma_prev = self.Sigma[arm]
        
        # Standard update:
        # Sigma_new = (Sigma_prev^{-1} + (1/sigma_r^2) x x^T)^{-1}
        # mu_new = Sigma_new (Sigma_prev^{-1} mu_prev + (1/sigma_r^2) x r)
        
        # With forgetting factor, we effectively treat the previous state as if it had less weight.
        # A common approximation is to scale the precision matrix.
        # Precision P_prev = Sigma_prev^{-1}
        # P_new = gamma * P_prev + (1/sigma_r^2) x x^T
        # b_new = gamma * (P_prev @ mu_prev) + (1/sigma_r^2) x r  (Note: P_prev mu_prev is the prior precision-weighted mean)
        # Actually, it's easier to maintain Precision P and vector b (P @ mu).
        
        # Let's switch internal representation to Precision P and b for numerical stability and easier forgetting.
        # But I already initialized mu and Sigma. Let's convert or just implement the update on mu/Sigma directly.
        # Direct update on mu/Sigma is numerically unstable if we do matrix inversions every step.
        # It is better to maintain P (precision) and b (precision-weighted mean).
        
        # Let's refactor the state to P and b in __init__? No, I can just convert on the fly or change init.
        # Since I can't change init signature, I will store P and b as attributes, initialized from mu/Sigma.
        
        if not hasattr(self, 'P'):
            self.P = [np.linalg.inv(self.Sigma[a]) for a in range(self.n)]
            self.b = [self.P[a] @ self.mu[a] for a in range(self.n)]
            
        # Update P and b for arm 'arm'
        # P_new = gamma * P_old + (1/sigma_r^2) * outer(x, x)
        # b_new = gamma * b_old + (1/sigma_r^2) * x * reward
        
        # Assume noise variance sigma_r^2 = 1.0
        sigma_r2 = 1.0
        
        P_old = self.P[arm]
        b_old = self.b[arm]
        
        P_new = self.gamma * P_old + (1.0 / sigma_r2) * np.outer(context_vec, context_vec)
        b_new = self.gamma * b_old + (1.0 / sigma_r2) * context_vec * reward
        
        self.P[arm] = P_new
        self.b[arm] = b_new
        
        # Update mu and Sigma for use in select()
        # mu = inv(P) @ b
        # Sigma = inv(P)
        # To avoid inverting every step, we can just compute it in select().
        # But select() needs Sigma for Cholesky.
        # Inverting a dim x dim matrix is O(d^3). With d=5 or 10, this is fast.
        # We will update mu and Sigma here to keep select() fast.
        
        try:
            Sigma_new = np.linalg.inv(P_new)
            mu_new = Sigma_new @ b_new
        except np.linalg.LinAlgError:
            # Regularize if singular
            Sigma_new = np.linalg.inv(P_new + 1e-6 * np.eye(self.d))
            mu_new = Sigma_new @ b_new
            
        self.Sigma[arm] = Sigma_new
        self.mu[arm] = mu_new
