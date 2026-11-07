import numpy as np
import time
import sys

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 250.0  # Safety margin for 300s limit
        
        dim = self.dim
        # Initial parameters
        x0 = np.zeros(dim)
        f0 = func(x0)
        
        # Initial standard deviation. For [-5, 5]^5, a starting sigma of 2-3 is reasonable.
        # We start a bit wide to explore.
        sigma = 2.0
        
        # Population size
        # Standard formula: lambda = 4 + floor(3 * ln(dim))
        lam = 4 + int(3 * np.log(dim)) if dim > 1 else 1
        # Ensure lambda is at least 4
        lam = max(4, lam)
        
        # CMA-ES parameters
        mu = lam // 2  # Number of recombination points
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma_ps = (mu_eff + 2) / (dim + mu_eff + 5)
        c_sigma_1 = (2 + 2.0 / (mu_eff + 1)) / (dim + 2 + 2.0 / (mu_eff + 1))
        c_cov = (
            (4 + mu_eff / dim) * c_sigma_1
            - c_sigma_1
            + 2 * (min(1, c_sigma_1 - 1) * (mu_eff + 1) / dim - 1)
        ) / (dim + 3)
        d_cov = 5 + 3 * np.log(dim)
        
        # State variables
        C = np.eye(dim)
        C_sqrt_inv = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_N = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (20 * dim**2))
        
        x_best = x0.copy()
        f_best = f0
        count = 1
        
        # Initial population
        # Generate first population
        # We need to generate mu + (lam - mu) points
        
        while count < self.budget:
            # Check time limit
            if time.time() - start_time > time_limit:
                break
            
            # Recombination
            # Sample population
            z = np.random.randn(lam, dim)
            # Transform to search space
            # x_i = x0 + sigma * z_i @ C_sqrt_inv
            # Efficient multiplication: (C_sqrt_inv @ z.T).T
            pop = x0 + sigma * (z @ C_sqrt_inv.T)
            
            # Evaluate population
            fs = np.zeros(lam)
            for i in range(lam):
                if time.time() - start_time > time_limit:
                    # Fill remaining with worst value if time runs out mid-eval
                    fs[i:] = f_best + 1e10 
                    break
                fs[i] = func(pop[i])
                count += 1
                if fs[i] < f_best:
                    f_best = fs[i]
                    x_best = pop[i].copy()
            
            # Sort by fitness
            order = np.argsort(fs)
            # Take best mu
            z_best = z[order[:mu]]
            x_best_pop = pop[order[:mu]]
            f_best_pop = fs[order[:mu]]
            
            # Weighted mean of best mu
            x_m = np.sum(weights[:, None] * x_best_pop, axis=0)
            
            # Update step sizes
            # Path cumulation for sigma control
            ps = (1 - c_sigma_ps) * ps + np.sqrt(c_sigma_ps * (2 - c_sigma_ps) * mu_eff) / sigma * (C_sqrt_inv @ (x_m - x0))
            h_sig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma_ps) ** (count % 10 + 1))) < (1.4 + 2 / (dim + 1)) * chi_N
            
            if h_sig:
                pc = (1 - c_sigma_1) * pc + np.sqrt(c_sigma_1 * (2 - c_sigma_1) * mu_eff) * (x_m - x0) / sigma
            else:
                pc = (1 - c_sigma_1) * pc
            
            # Update covariance matrix
            if h_sig:
                v = (x_m - x0) / sigma
                C = (1 + c_cov * (c_sigma_1 * (2 - c_sigma_1) - c_cov) * mu_eff + c_cov * (np.dot(pc, pc) - 1) - c_sigma_1 * c_cov) * C + c_cov * (np.outer(v, v) - C)
            else:
                C = (1 + c_cov * (c_sigma_1 * (2 - c_sigma_1) - c_cov) * mu_eff - c_sigma_1 * c_cov) * C + c_cov * (np.outer(v, v) - C)
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma_ps / (5 * dim) * (np.linalg.norm(ps) / chi_N - 1))
            
            # Ensure sigma doesn't explode or vanish
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            # Update x0
            x0 = x_m
            
            # Diagonalize C to update C_sqrt_inv
            # Use eigendecomposition for stability
            eigval, eigvec = np.linalg.eigh(C)
            # Clip small negative eigenvalues due to numerical errors
            eigval = np.clip(eigval, 1e-12, None)
            
            # Check condition number to detect singularity
            if eigval[0] < 1e-12 or eigval[-1] / eigval[0] > 1e10:
                # Reset C to identity if it becomes ill-conditioned
                C = np.eye(dim)
                eigval = np.ones(dim)
                eigvec = np.eye(dim)
                
            # Compute C_sqrt_inv
            # C = V diag(eigval) V^T
            # C^(-1/2) = V diag(eigval^(-1/2)) V^T
            C_sqrt_inv = eigvec @ np.diag(1.0 / np.sqrt(eigval)) @ eigvec.T
            
            # Optional: Local restart if sigma is very small and no progress
            if sigma < 1e-6:
                # Perturb around best found
                x0 = x_best + np.random.randn(dim) * 1.0
                sigma = 1.0
                C = np.eye(dim)
                C_sqrt_inv = np.eye(dim)
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                # Re-evaluate new center to ensure validity (optional, but safe)
                # We just let the next loop handle it.

        return x_best, f_best

if __name__ == "__main__":
    # Dummy test to ensure syntax is correct
    def dummy_func(x):
        return np.sum(x**2)
    
    opt = Optimizer(100, 5)
    res = opt(dummy_func)
    print(f"Best: {res[0]}, Fitness: {res[1]}")
