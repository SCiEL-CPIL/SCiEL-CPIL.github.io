"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for BBOB-style Sphere f1."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        t0 = time.time()
        deadline = t0 + 280  # Safety margin before 300s limit

        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= total_budget:
                return np.inf
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # 1. Global Exploration Phase
        # Use a small budget for random sampling to find a good basin.
        n_global = min(15, total_budget // 5)
        
        for _ in range(n_global):
            if time.time() > deadline or calls_made >= total_budget:
                break
            x = np.random.uniform(-4.0, 4.0, dim)
            safe_call(x)
            
        if best_x is None:
            x0 = np.zeros(dim)
            safe_call(x0)
            if best_x is None:
                return None, np.inf

        # 2. CMA-ES Phase with multi-start strategy
        # Progressive step sizes: start wide, then narrow
        step_sizes = [4.0, 1.5, 0.5, 0.1]
        
        for step in step_sizes:
            if calls_made >= total_budget or time.time() > deadline:
                break
            
            # Initialize CMA-ES state
            x_mean = best_x.copy()
            sigma = step
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            
            # CMA-ES parameters
            lambda_ = 4 + int(np.floor(3 * np.log(dim)))  # population size
            cmu = 1.0 / (dim + 2)
            c_sigma = (mu0 + 1.5) / (dim + 3)
            c_c = (4 + mu0 / c_sigma) / (dim + 4 + 2 * mu0 / c_sigma)
            chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim**2))
            mu0 = int(np.floor(lambda_ / 2))
            
            # Run CMA-ES for a limited number of generations
            max_generations = 50
            stagnation_counter = 0
            best_f_in_cma = best_f
            
            for gen in range(max_generations):
                if calls_made >= total_budget or time.time() > deadline:
                    break
                
                # Check for stagnation
                if best_f_in_cma - best_f < 1e-10:
                    stagnation_counter += 1
                else:
                    stagnation_counter = 0
                    best_f_in_cma = best_f
                
                if stagnation_counter > 10:
                    break
                
                # Generate lambda_ samples
                samples = np.zeros((lambda_, dim))
                for i in range(lambda_):
                    z = np.random.randn(dim)
                    samples[i] = x_mean + sigma * (C @ z)
                
                # Evaluate samples
                fvals = np.zeros(lambda_)
                for i in range(lambda_):
                    if calls_made >= total_budget or time.time() > deadline:
                        fvals[i] = np.inf
                        break
                    fvals[i] = safe_call(samples[i])
                
                if calls_made >= total_budget or time.time() > deadline:
                    break
                
                # Sort by fitness
                idx = np.argsort(fvals)
                samples = samples[idx]
                fvals = fvals[idx]
                
                # Update mean
                weights = np.ones(mu0) / mu0
                x_mean_new = np.sum(weights[:, None] * samples[:mu0], axis=0)
                
                # Update covariance
                y = (x_mean_new - x_mean) / sigma
                pc = (1 - c_c) * pc + c_c * y
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu0) / np.linalg.norm(pc) * pc
                rank_mu = np.outer(y, y)
                rank_one = np.outer(ps, ps)
                C = (1 + c_c * c_sigma * (2 - c_sigma) - c_sigma) * C + c_sigma * rank_one + c_c * c_sigma * (2 - c_sigma) * rank_mu
                C = (C + C.T) / 2  # Symmetrize
                
                # Update sigma
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                
                x_mean = x_mean_new
                
                # Check if we've converged
                if np.linalg.norm(y) < 1e-8:
                    break

        return best_x, best_f

# EVOLVE-BLOCK-END
