import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0
        
        # Global best
        x_global = None
        f_global = np.inf
        
        # We will run multiple CMA-ES restarts
        # Budget is 300. CMA-ES typically needs ~300-500 evals for dim=5 to converge well.
        # Let's do 2 restarts of ~150 evals each, or 1 large one?
        # Multimodal functions (Rastrigin, Gallagher) benefit from restarts.
        # Unimodal (Sphere, Rosenbrock) might benefit from one long run, but CMA-ES is robust.
        # Let's try 3 restarts of 100 evals each.
        
        num_restarts = 3
        budget_per_restart = self.budget // num_restarts
        
        rng = np.random.default_rng(42)
        
        for r in range(num_restarts):
            # Check time
            if time.time() > deadline:
                break
            
            # Initialize CMA-ES state
            # Start point: random in [-5, 5]
            x0 = rng.uniform(-5, 5, self.dim)
            
            # CMA-ES parameters
            # lambda: population size. For dim 5, lambda = 4 + floor(3 * log(5)) = 4 + 5 = 9? 
            # Standard formula: 4 + floor(3 * log(n))
            lam = 4 + int(3 * np.log(self.dim))
            
            # Initial sigma
            # Heuristic: 1.0 is often good for box [-5,5]
            sigma = 1.0
            
            # Mean
            mean = x0.copy()
            
            # Covariance matrix (identity initially)
            C = np.eye(self.dim)
            
            # Evolution paths
            ps = np.zeros(self.dim)
            pc = np.zeros(self.dim)
            
            # Adaptation parameters
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            
            c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (self.dim + mu_eff - 3)) + c_sigma
            c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
            c_1 = 2 / ((self.dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1. / mu_eff) / ((self.dim + 2) ** 2 + mu_eff))
            
            # Cholesky factor of C
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                B = np.eye(self.dim)
            
            # Current best for this restart
            x_best_local = x0.copy()
            f_best_local = func(x0)
            calls_made = 1
            
            if f_best_local < f_global:
                f_global = f_best_local
                x_global = x_best_local.copy()
            
            # Main loop for this restart
            while calls_made < budget_per_restart:
                if time.time() > deadline:
                    break
                
                # Generate population
                offspring = []
                fitnesses = []
                
                # We have budget_per_restart - calls_made left.
                # We can generate up to lam offspring.
                # But we must not exceed budget_per_restart.
                current_batch_size = min(lam, budget_per_restart - calls_made)
                
                for i in range(current_batch_size):
                    # Sample from N(mean, sigma^2 * C)
                    z = rng.standard_normal(self.dim)
                    x_new = mean + sigma * (B @ z)
                    
                    # Clip to box
                    x_new = np.clip(x_new, -5, 5)
                    
                    # Evaluate
                    f_new = func(x_new)
                    calls_made += 1
                    
                    offspring.append(x_new)
                    fitnesses.append(f_new)
                    
                    # Update local best
                    if f_new < f_best_local:
                        f_best_local = f_new
                        x_best_local = x_new.copy()
                        
                        # Update global best
                        if f_new < f_global:
                            f_global = f_new
                            x_global = x_new.copy()
                
                # Sort by fitness
                sorted_idx = np.argsort(fitnesses)
                offspring = [offspring[i] for i in sorted_idx]
                fitnesses = [fitnesses[i] for i in sorted_idx]
                
                # Update mean
                mean_new = np.zeros(self.dim)
                for i in range(mu):
                    mean_new += weights[i] * offspring[i]
                
                # Update evolution paths
                # Check if mean changed significantly
                if np.linalg.norm(mean_new - mean) > 1e-16:
                    # Inverse Cholesky
                    try:
                        B_inv = np.linalg.inv(B)
                    except np.linalg.LinAlgError:
                        B_inv = np.eye(self.dim)
                    
                    # ps update
                    ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B_inv @ ((mean_new - mean) / sigma))
                    
                    # pc update
                    if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made + 1) / lam)) < (1.4 + 2 / (self.dim + 1)) * np.sqrt((self.dim + 1) / (self.dim - 1)):
                        pc = (1 - c_c) * pc + c_c * (B_inv @ ((mean_new - mean) / sigma))
                
                mean = mean_new
                
                # Update Covariance
                # Rank-mu update
                v = np.zeros((self.dim, mu))
                for i in range(mu):
                    v[:, i] = (offspring[i] - mean) / sigma
                    
                C = (1 + c_1 + c_mu - c_1) * C + c_1 * (np.outer(ps, ps) - C) + c_mu * (v @ np.diag(weights) @ v.T - C)
                
                # Rank-one update
                # Actually, the standard update is:
                # C = (1 - c_1 - c_mu) * C + c_1 * (ps @ ps.T + (1 - h_ps) * C) + c_mu * sum(w_i * (x_i - mean) @ (x_i - mean).T)
                # Let's use the simpler stable form if possible, or just recompute Cholesky and fix if it fails.
                
                # Simpler robust update for this context:
                # Just use the standard CMA-ES update rules.
                # Recalculate B
                try:
                    # Ensure symmetry
                    C = (C + C.T) / 2
                    # Eigenvalue check for positive definiteness
                    eigvals = np.linalg.eigvalsh(C)
                    if np.min(eigvals) < 1e-16:
                        # Add small epsilon
                        C += 1e-16 * np.eye(self.dim)
                    B = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # If Cholesky fails, reset C to identity or previous valid one
                    # For simplicity and robustness in this short budget, reset to identity
                    C = np.eye(self.dim)
                    B = np.eye(self.dim)
                    # Also reset paths?
                    ps = np.zeros(self.dim)
                    pc = np.zeros(self.dim)
                
                # Adapt sigma
                h_sigma = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made + 1) / lam)) < (1.4 + 2 / (self.dim + 1)) * np.sqrt((self.dim + 1) / (self.dim - 1)) else 0
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(2 * self.dim * (1 - (1 - c_sigma) ** (calls_made / lam))) - 1))
                
                # Ensure sigma doesn't explode or vanish
                sigma = np.clip(sigma, 1e-8, 10.0)

        return x_global, f_global
