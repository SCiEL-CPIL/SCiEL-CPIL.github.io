import numpy as np
import time

class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0
        
        # Discount factor for non-stationarity
        # 0.99 means we effectively keep ~100 recent samples
        self.gamma = 0.99
        
        # Regularization to prevent singular A
        self.reg = 1.0
        
        # Initialize A (precision) and b (cross product) for each arm
        # A is d x d, b is d x 1
        self.A = [np.eye(self.d) * self.reg for _ in range(self.n)]
        self.b = [np.zeros(self.d) for _ in range(self.n)]
        
        # Cache for inverse and mean to avoid recomputation if select is called
        # multiple times without update? No, update is always followed by select.
        # But select is called once, then update. So we compute in select.
        
        # For sampling, we need the Cholesky of Sigma or just sample from N(mu, Sigma)
        # np.random.multivariate_normal is slow. 
        # Better: Sample z ~ N(0, I), then theta = mu + L z, where L is Cholesky of Sigma.
        # Or, since A is SPD, we can sample from N(mu, A^{-1}) by:
        # 1. Compute L_A = Cholesky(A)
        # 2. Sample z ~ N(0, I)
        # 3. Solve L_A w = z  => w = L_A^{-1} z
        # 4. theta = mu + A^{-1} w ? No.
        # If X ~ N(0, A^{-1}), then A^{1/2} X ~ N(0, I).
        # Let X = A^{-1/2} Z.
        # Using Cholesky: A = L L^T. A^{-1/2} = L^{-T} (L^{-1})^T? 
        # Simpler: Solve A X = Z. Then X ~ N(0, A^{-1}).
        # So:
        # 1. z = np.random.randn(d)
        # 2. delta = np.linalg.solve(A, z)
        # 3. theta_sample = mu + delta
        
        self.t = 0

    def select(self, context):
        if time.time() > self.deadline:
            # Fallback: Greedy using current means
            context_vec = np.array(context)
            scores = []
            for a in range(self.n):
                mu = np.linalg.solve(self.A[a], self.b[a])
                scores.append(np.dot(mu, context_vec))
            return int(np.argmax(scores))

        context_vec = np.array(context)
        best_arm = 0
        best_score = -np.inf
        
        for a in range(self.n):
            # Compute mean mu = A^{-1} b
            # Use solve for stability
            mu = np.linalg.solve(self.A[a], self.b[a])
            
            # Sample theta from N(mu, A^{-1})
            # z ~ N(0, I)
            z = np.random.randn(self.d)
            # delta = A^{-1} z
            delta = np.linalg.solve(self.A[a], z)
            theta_sample = mu + delta
            
            score = np.dot(theta_sample, context_vec)
            
            if score > best_score:
                best_score = score
                best_arm = a
                
        return best_arm

    def update(self, context, arm, reward):
        if time.time() > self.deadline:
            return

        context_vec = np.array(context)
        
        # Update A and b for the chosen arm
        # A = gamma * A + x x^T
        # b = gamma * b + r x
        
        self.A[arm] = self.gamma * self.A[arm] + np.outer(context_vec, context_vec)
        self.b[arm] = self.gamma * self.b[arm] + reward * context_vec
        
        self.t += 1
