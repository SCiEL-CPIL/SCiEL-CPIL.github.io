"""CMA-ES Optimizer for BBOB f9 (Rosenbrock rotated)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) Optimizer.
    
    Strategy:
    - Use a population-based evolutionary strategy that adapts the search distribution.
    - The covariance matrix adapts to the local curvature and orientation of the function,
      which is crucial for the rotated Rosenbrock function.
    - Handles the box constraints [-5, 5]^5 by clipping and re-evaluation or penalty.
    - Manages the 500 evaluation budget and 330s time limit strictly.
    """
    
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 325  # Safety margin for 330s limit
        n = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Initial mean: Center of the box
        x_mean = np.zeros(n)
        
        # Initial step size: Large enough to explore the box, but not too large
        # Box is [-5, 5], so width is 10. A good start is around 2-3.
        sigma = 2.0
        
        # Population size
        # Default lambda = 4 + floor(3 * log(n))
        lam = 4 + int(3 * np.log(n))
        
        # Weights for recombination
        mu = lam // 2  # Number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        
        # Effective population size
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # Damping for covariance update
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        
        # Initial covariance matrix
        C = np.eye(n)
        
        # State variables
        pc = np.zeros(n)  # Evolution path for covariance
        ps = np.zeros(n)  # Evolution path for step size
        
        # Cholesky factor of C
        C_sqrt = np.linalg.cholesky(C)
        
        # Best point found so far
        best_x = x_mean.copy()
        best_f = float('inf')
        
        calls_made = 0
        
        # Helper to check time
        def time_left():
            return time.time() < deadline
        
        # Helper to evaluate a point
        def evaluate(x):
            nonlocal calls_made, best_f, best_x
            if not time_left():
                return None
            if calls_made >= budget:
                return None
            
            # Clip to box
            x_clipped = np.clip(x, -5, 5)
            
            f = func(x_clipped)
            calls_made += 1
            
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            
            return f, x_clipped
        
        # Initial evaluation of the mean
        res = evaluate(x_mean)
        if res is None:
            return best_x, best_f
        f_mean, x_mean_eval = res
        
        # Main Loop
        while calls_made < budget and time_left():
            
            # 1. Sample offspring
            offspring = []
            offspring_vals = []
            
            for i in range(lam):
                # Generate random vector
                z = np.random.randn(n)
                
                # Transform using C_sqrt
                # x = x_mean + sigma * C_sqrt @ z
                x_new = x_mean + sigma * (C_sqrt @ z)
                
                # Evaluate
                res = evaluate(x_new)
                if res is None:
                    break
                
                f_val, x_eval = res
                offspring.append(x_eval)
                offspring_vals.append(f_val)
            
            if not offspring:
                break
            
            # Sort offspring by function value
            indices = np.argsort(offspring_vals)
            offspring_sorted = [offspring[i] for i in indices]
            vals_sorted = [offspring_vals[i] for i in indices]
            
            # 2. Update mean (Recombination)
            # Use the best mu offspring
            old_x_mean = x_mean.copy()
            x_mean = np.sum([weights[i] * offspring_sorted[i] for i in range(mu)], axis=0)
            
            # 3. Update step size (sigma)
            # Calculate normalized step
            # C_sqrt_inv_old is needed, but we can approximate or use the current C_sqrt
            # Ideally, we use the C_sqrt from the start of the iteration
            # For simplicity and stability, we use the current C_sqrt (which is from start of iteration)
            # Note: Strict CMA-ES uses C_sqrt^{-1} (x_mean - x_mean_old) / sigma
            try:
                C_sqrt_inv = np.linalg.inv(C_sqrt)
            except:
                C_sqrt_inv = np.eye(n)
            
            diff = (x_mean - old_x_mean) / sigma
            norm_diff = np.linalg.norm(C_sqrt_inv @ diff)
            
            # Expected norm of N(0, I)
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            
            # Update ps
            h_sigma = (norm_diff / np.sqrt(1 - (1 - c_c)**(1 + calls_made/2))) < (1.4 + 2/(n+1)) * chi_n
            h_sigma = 1.0 if h_sigma else 0.0
            
            ps = (1 - c_sigma) * ps + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt_inv @ diff)
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Clamp sigma to reasonable bounds
            sigma = np.clip(sigma, 1e-8, 50.0)
            
            # 4. Update covariance matrix (C)
            # Update pc
            pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * diff
            
            # Rank-one and rank-mu updates
            v = x_mean - old_x_mean
            
            # Rank-one
            C = (1 + c_1 * h_sigma * (2 - h_sigma) - c_1) * C + c_1 * np.outer(pc, pc)
            
            # Rank-mu
            # y_i = (x_i - x_mean_old) / sigma
            # y = [y_1, ..., y_mu]
            y = np.array([(offspring_sorted[i] - old_x_mean) / sigma for i in range(mu)])
            
            # Weighted outer product
            # sum(weights[i] * y[i] * y[i].T)
            # This is equivalent to Y @ (weights[:, None] * Y.T)
            weighted_outer = y.T @ (weights[:mu] * y)
            
            C = (1 + c_mu * h_sigma * (2 - h_sigma) - c_1 - c_mu) * C + c_mu * weighted_outer
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Ensure positive definiteness
            try:
                C_sqrt = np.linalg.cholesky(C)
            except:
                # If Cholesky fails, add a small diagonal
                C += 1e-6 * np.eye(n)
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except:
                    # Reset C to identity if completely broken
                    C = np.eye(n)
                    C_sqrt = np.eye(n)
                    pc = np.zeros(n)
                    ps = np.zeros(n)
                    sigma = 1.0
            
            # Check for stagnation or convergence
            if sigma < 1e-6:
                break
                
            # If we have used most of the budget, we might want to stop early if no improvement
            # But CMA-ES is robust, so let it run until budget or time is out.
            
        return best_x, best_f
# EVOLVE-BLOCK-END
