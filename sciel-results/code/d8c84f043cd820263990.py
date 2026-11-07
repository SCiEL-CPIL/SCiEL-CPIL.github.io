"""New program: CMA-ES variant with adaptive step size for BBOB f9."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES style optimizer with population-based search."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        
        # Initial population size
        lambda_pop = max(4, int(4 + 3 * np.log(dim)))
        
        # Start at origin (common good starting point for Rosenbrock)
        mean = np.zeros(dim)
        sigma = 2.0
        
        # Track best solution found
        best_x = mean.copy()
        best_f = func(best_x)
        calls_made = 1
        
        # CMA-ES parameters
        c_sigma = (lambda_pop + 2) / (dim + lambda_pop + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((lambda_pop - 1) / (dim + 1)) - 1) + c_sigma
        
        # History for step size adaptation
        sigma_hist = []
        
        while calls_made < self.budget:
            # Generate population
            pop = []
            for _ in range(lambda_pop):
                step = rng.standard_normal(dim)
                candidate = np.clip(mean + sigma * step, -5, 5)
                pop.append(candidate)
            
            # Evaluate population
            fvals = []
            for x in pop:
                if calls_made >= self.budget:
                    break
                f = func(x)
                fvals.append(f)
                calls_made += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            
            if len(fvals) == 0:
                break
            
            # Sort by fitness
            sorted_idx = np.argsort(fvals)
            mu = lambda_pop // 2  # Number of best individuals used for update
            
            # Weighted recombination
            weights = np.ones(mu)
            weights /= weights.sum()
            
            new_mean = np.zeros(dim)
            for i in range(mu):
                new_mean += weights[i] * pop[sorted_idx[i]]
            
            # Update sigma using success rule
            dist = np.linalg.norm(new_mean - mean)
            if dist > 1e-10:
                sigma *= np.exp(c_sigma * (dist / sigma - 1) / d_sigma)
            
            # Update mean
            mean = new_mean
            mean = np.clip(mean, -5, 5)
            
            # Adaptive sigma based on recent improvements
            if dist < 0.01 * sigma:
                sigma *= 0.9  # Shrink if no progress
            elif dist > 1.0 * sigma:
                sigma *= 1.1  # Expand if large steps
            
            sigma = max(1e-4, min(sigma, 10.0))
        
        return best_x, best_f
# EVOLVE-BLOCK-END
