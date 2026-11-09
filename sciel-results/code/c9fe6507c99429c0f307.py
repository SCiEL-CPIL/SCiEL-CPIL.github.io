import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        
        # Multiple random starts to find a good initial point
        num_starts = 10
        x_best = None
        fx_best = np.inf
        for _ in range(num_starts):
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            if fx < fx_best:
                fx_best = fx
                x_best = x.copy()
        
        # (1+1) ES with adaptive step size
        sigma = 1.0
        success_rate = 0.5
        for _ in range(self.budget - num_starts):
            y = np.clip(x_best + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            if fy < fx_best:
                x_best, fx_best = y, fy
                success_rate = 0.6 * success_rate + 0.4 * 1.0
            else:
                success_rate = 0.6 * success_rate + 0.4 * 0.0
            
            # Adapt step size based on success rate
            if success_rate > 0.5:
                sigma *= 1.1
            elif success_rate < 0.5:
                sigma *= 0.9
            
            # Ensure sigma is within reasonable bounds
            sigma = np.clip(sigma, 1e-3, 10.0)
        
        # Local search to refine the solution
        sigma_local = 0.1
        for _ in range(50):
            y = np.clip(x_best + sigma_local * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            if fy < fx_best:
                x_best, fx_best = y, fy
                sigma_local *= 1.1
            else:
                sigma_local *= 0.9
            sigma_local = np.clip(sigma_local, 1e-3, 1.0)
        
        return x_best, fx_best
