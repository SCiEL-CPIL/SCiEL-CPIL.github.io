# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    CMA-ES Optimizer for BBOB-style f13 (Sharp Ridge).
    
    Strategy:
    1. Use CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
    2. CMA-ES is particularly effective for ill-conditioned functions like sharp ridges
       because it adapts the covariance matrix of the sampling distribution to match
       the local curvature.
    3. This allows the algorithm to take large steps along the flat directions of the
       ridge and small steps in the sharp directions, without explicitly computing
       gradients or performing 1D line searches.
    4. We use a standard CMA-ES implementation with adaptive population size and
       restarts if stagnation is detected, though for a 500-call budget in 5D,
       a single run with careful parameter tuning is likely sufficient.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls_made = 0

        def timed_call(x):
            nonlocal best_x, best_f, calls_made
            if calls_made >= self.budget:
                return None
            if time.time() > deadline - 2:
                return None
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initialization
        # Start with a few random points to get a reasonable initial mean and sigma.
        # For f13, the minimum is in [-4, 4]^5.
        # We initialize the CMA-ES mean at the origin or a random point.
        # Let's use a small random initialization to avoid symmetry issues if the shift is small.
        
        # Standard CMA-ES parameters
        # Population size lambda
        lam = 4 + int(3 * np.log(self.dim))  # Standard formula: 4 + 3*log(5) ~ 8
        # For 5D, lam = 8.
        
        # Initial sigma
        # The box is [-5, 5]. The minimum is in [-4, 4].
        # A good initial sigma is roughly 1.0 or 2.0.
        sigma = 1.0
        
        # Mean
        mean = np.zeros(self.dim)
        
        # Covariance matrix
        C = np.eye(self.dim)
        
        # Eigenvalues and eigenvectors of C
        eigvals = np.ones(self.dim)
        eigvecs = np.eye(self.dim)
        
        # Learning rates
        # Standard CMA-ES learning rates
        mu = lam // 2  # Number of parents
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Step size adaptation
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + c_sigma
        
        # Mean adaptation
        c_mean = (mu_eff + 2) / (self.dim + mu_eff + 5)
        
        # Covariance adaptation
        c_c = (4 - mu_eff + c_sigma) / (self.dim + 4)
        c_cov = (2 * (mu_eff - 1 + 1/mu_eff)) / (self.dim**2 + 10)
        c_cov = min(c_cov, 2 * c_c)
        # Correction term for covariance update
        d_cov = c_cov * (3 * self.dim - 1) + 2 * (c_cov - c_c)
        
        # Path lengths
        p_sigma = np.zeros(self.dim)
        p_c = np.zeros(self.dim)
        
        # Counters
        counter = 0
        no_improvement_count = 0
        
        # Initial evaluation of mean
        f_mean = timed_call(mean)
        if f_mean is None:
            return mean, np.inf
        best_f = f_mean
        best_x = mean.copy()
        
        # Main loop
        while calls_made < self.budget:
            if time.time() > deadline - 2:
                break
            
            # Generate offspring
            offspring = []
            offspring_f = []
            
            for i in range(lam):
                if calls_made >= self.budget:
                    break
                if time.time() > deadline - 2:
                    break
                    
                # Sample from multivariate normal
                # x = mean + sigma * eigvecs @ (sqrt(eigvals) * rng.normal(size=dim))
                z = rng.normal(size=self.dim)
                x = mean + sigma * (eigvecs @ (np.sqrt(eigvals) * z))
                
                # Clip to box [-5, 5]
                x = np.clip(x, -5, 5)
                
                fx = timed_call(x)
                if fx is None:
                    break
                    
                offspring.append(x)
                offspring_f.append(fx)
            
            if len(offspring) < 2:
                break
            
            offspring_f = np.array(offspring_f)
            offspring = np.array(offspring)
            
            # Sort offspring by fitness
            sorted_indices = np.argsort(offspring_f)
            sorted_offspring = offspring[sorted_indices]
            sorted_offspring_f = offspring_f[sorted_indices]
            
            # Check for improvement
            if sorted_offspring_f[0] < best_f:
                best_f = sorted_offspring_f[0]
                best_x = sorted_offspring[0].copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                
            # Restart if stagnation is severe
            if no_improvement_count > 10 * lam:
                # Try a random restart from the best point found so far
                # Or just break and rely on the best point
                # For a 500 budget, restarting might waste too much.
                # Let's just shrink sigma significantly and continue
                sigma *= 0.5
                if sigma < 1e-6:
                    break
                no_improvement_count = 0
            
            # Update mean
            # Weighted average of the best mu offspring
            new_mean = np.zeros(self.dim)
            for i in range(mu):
                new_mean += weights[i] * sorted_offspring[i]
            
            # Update path lengths
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C^(-1/2) * (x_new - x_old) / sigma
            # C^(-1/2) is approximated by eigvecs @ diag(1/sqrt(eigvals)) @ eigvecs.T
            # But for efficiency, we can use the inverse square root of C directly if we maintain it.
            # Here, we use the eigen decomposition.
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            # Calculate direction
            direction = (new_mean - mean) / sigma
            
            # Project direction onto the principal axes
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ direction)
            
            # Check if p_sigma is too long
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(counter + 1)) > (1.4 + 2 / (self.dim + 1)) * np.sqrt(mu_eff):
                sigma /= np.exp(0.5 * c_sigma * (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(counter + 1)) - np.sqrt(mu_eff)))
                
            # Update p_c
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv_sqrt @ direction)
            
            # Update covariance matrix
            # C = (1 - c_c - c_cov) * C + c_c * (p_c * p_c.T) + c_cov * sum(weights[i] * (y_i * y_i.T))
            # where y_i = (x_i - x_old) / sigma
            
            rank_one_update = np.zeros((self.dim, self.dim))
            for i in range(mu):
                y_i = (sorted_offspring[i] - mean) / sigma
                rank_one_update += weights[i] * np.outer(y_i, y_i)
                
            C = (1 - c_c - c_cov) * C + c_c * np.outer(p_c, p_c) + c_cov * rank_one_update
            
            # Ensure C is positive semi-definite by regularizing if needed
            # Simple regularization: add small identity if eigenvalues become negative
            eigvals_new, eigvecs_new = np.linalg.eigh(C)
            if np.any(eigvals_new < 1e-12):
                C += 1e-12 * np.eye(self.dim)
                eigvals_new, eigvecs_new = np.linalg.eigh(C)
                
            # Update eigen decomposition
            eigvals = np.clip(eigvals_new, 1e-12, None)
            eigvecs = eigvecs_new
            
            # Update mean
            mean = new_mean
            counter += 1
            
            # Optional: If convergence is very slow, we might want to do a local polish
            # But CMA-ES should handle it.
            
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
            calls_made += 1
            
        return best_x, best_f

# EVOLVE-BLOCK-END
