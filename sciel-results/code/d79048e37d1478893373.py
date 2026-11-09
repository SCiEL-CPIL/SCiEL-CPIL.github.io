"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc


class Optimizer:
    """
    Optimizer for BBOB-style Schwefel f20 on [-5, 5]^5.
    
    Strategy:
      - Diverse initial exploration using Sobol quasi-random points for even coverage.
      - (1+1)-ES with adaptive sigma (1/5 success rule inspired) for fast convergence.
      - Periodic restarts using remaining Sobol points if stagnation is detected.
      - Final polish pass with scipy L-BFGS-B to refine the best solution.
      - Deadline guard to ensure we return within 330s wall-clock.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # safety margin
        
        rng = np.random.default_rng(42)
        
        # Generate Sobol sequence for global exploration
        try:
            sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
            sobol_points = sampler.random(200)
            sobol_points = sobol_points * 10.0 - 5.0
        except Exception:
            sobol_points = rng.uniform(-5, 5, (200, dim))
        
        sobol_idx = 0
        best_x = None
        best_f = np.inf
        
        # Phase 1: Global exploration with Sobol points
        # Evaluate first 10 Sobol points to find a decent basin quickly
        for i in range(10):
            if time.time() > deadline or budget <= 0:
                break
            x0 = sobol_points[sobol_idx]
            sobol_idx += 1
            x0 = np.clip(x0, -5, 5)
            fx = func(x0)
            budget -= 1
            if fx < best_f:
                best_f = fx
                best_x = x0.copy()
        
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            budget -= 1
        
        # Phase 2: (1+1)-ES with adaptive sigma and restarts
        sigma = 1.5
        sigma_min = 0.01
        sigma_max = 5.0
        stagnation_limit = 30
        stagnation_counter = 0
        
        while budget > 0 and time.time() < deadline:
            # Generate offspring
            y = np.clip(best_x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            budget -= 1
            
            if fy < best_f:
                best_x = y
                best_f = fy
                sigma *= 1.3  # Increase step size on success
                stagnation_counter = 0
            else:
                sigma *= 0.87  # Decrease step size on failure
                stagnation_counter += 1
            
            sigma = np.clip(sigma, sigma_min, sigma_max)
            
            # Restart if stagnating
            if stagnation_counter >= stagnation_limit:
                # Pick a new point from Sobol sequence or random
                if sobol_idx < len(sobol_points):
                    new_x = sobol_points[sobol_idx]
                    sobol_idx += 1
                else:
                    new_x = rng.uniform(-5, 5, dim)
                
                new_x = np.clip(new_x, -5, 5)
                if budget > 0:
                    fn = func(new_x)
                    budget -= 1
                    if fn < best_f:
                        best_f = fn
                        best_x = new_x.copy()
                        sigma = 1.5
                    else:
                        # Keep current best but reset stagnation
                        sigma = min(sigma * 2.0, sigma_max)
                stagnation_counter = 0
        
        # Final polish pass with scipy L-BFGS-B
        # Only if we have enough budget and time left
        if budget > 20 and time.time() < deadline - 10:
            try:
                calls_used_in_scipy = [0]
                
                def wrapped_func(x):
                    nonlocal budget
                    if budget <= 0 or time.time() > deadline:
                        return best_f  # Return current best if out of budget
                    x = np.clip(x, -5, 5)
                    fx = func(x)
                    budget -= 1
                    calls_used_in_scipy[0] += 1
                    return fx

                # Limit iterations to avoid getting stuck, but allow enough for convergence
                maxit = min(50, max(1, budget // 2))
                
                result = minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': maxit, 'ftol': 1e-12, 'gtol': 1e-8}
                )

                if result.fun < best_f:
                    best_x = np.clip(result.x, -5, 5)
                    best_f = result.fun

            except Exception:
                pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
