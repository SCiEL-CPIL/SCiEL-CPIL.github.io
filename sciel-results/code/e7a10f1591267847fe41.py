"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Deterministic multi-start Nelder-Mead optimizer for WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        
        # Count evaluations
        eval_count = 0
        
        def eval_func(x):
            nonlocal eval_count
            if eval_count >= self.budget:
                return np.array([np.inf, np.inf, np.inf])
            eval_count += 1
            try:
                return np.array(problem(x))
            except Exception:
                return np.array([np.inf, np.inf, np.inf])

        # Generate diverse starting points based on WFG7 Pareto front structure
        # WFG7 Pareto front: f1 = x1, f2 = 1 - (f1/3)^0.5, f3 = 1 - (f1/5)^0.5
        # We sample t in [0.01, 0.99] to avoid boundaries
        n_starts = min(12, self.budget // 15)  # Use ~15 evals per start
        
        starts = []
        # Uniform sampling in t-space
        t_vals = np.linspace(0.05, 0.95, n_starts)
        
        for t in t_vals:
            # Map t to x1 in [0.1, 2.9] (within bounds typically)
            x1 = 0.1 + t * 2.8
            # Set other variables to midpoint of bounds as a reasonable default
            x0 = np.zeros(dim)
            x0[0] = x1
            for i in range(1, dim):
                x0[i] = (lo[i] + hi[i]) / 2.0
            
            # Clip to bounds
            x0 = np.clip(x0, lo, hi)
            starts.append(x0)
        
        # Add a few random starts for robustness
        rng = np.random.RandomState(42)
        n_random = max(1, self.budget // 50)
        for _ in range(n_random):
            x_rand = lo + (hi - lo) * rng.random(dim)
            starts.append(x_rand)
        
        # Run Nelder-Mead from each start
        all_points = []
        
        for x0 in starts:
            if eval_count >= self.budget:
                break
            if time.time() > deadline - 10:
                break
                
            try:
                res = minimize(
                    eval_func,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 30, 'fatol': 1e-4, 'xatol': 1e-4}
                )
                if res.fun[0] < np.inf:
                    all_points.append(res.x)
                    all_points.append(res.fun)
            except Exception:
                pass
        
        # If we have remaining budget, do a few more random evaluations
        while eval_count < self.budget and time.time() < deadline - 5:
            x_rand = lo + (hi - lo) * rng.random(dim)
            f_rand = eval_func(x_rand)
            if f_rand[0] < np.inf:
                all_points.append(x_rand)
                all_points.append(f_rand)
        
        return all_points

# EVOLVE-BLOCK-END
