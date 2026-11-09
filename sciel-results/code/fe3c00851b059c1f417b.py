"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import qmc

class Optimizer:
    """Hybrid Sobol global search and (1+1)-ES local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Phase 1: Global Search using Sobol sequence
        # Use a portion of the budget for global exploration
        n_global = min(100, budget // 2)
        n_local = budget - n_global
        
        sampler = qmc.Sobol(d=dim, scramble=True, seed=0)
        # We need at least 2 points for Sobol to work well, but 100 is fine
        # Sobol requires power of 2 for optimal uniformity, but it works for any N
        # Let's stick to power of 2 for better distribution if possible, but 100 is okay.
        # Actually, let's use 128 (2^7) if budget allows, else 64, else 32.
        # For budget 500, n_global=100 is fine.
        
        # Generate Sobol points in [0, 1]^dim
        # Note: Sobol sequence generation can be slow for large N, but 100 is tiny.
        # We need to ensure we don't generate more than needed.
        # qmc.Sobol.random(N) returns N points.
        # We will generate n_global points.
        
        # To handle the power of 2 requirement for best quality, we can round up to next power of 2
        # but just use the first N points.
        points = sampler.random(n_global)
        
        # Scale to [-5, 5]^dim
        points = points * 10.0 - 5.0
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        for x in points:
            if calls_made >= budget:
                break
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        # Phase 2: Local Refinement using (1+1)-ES
        if best_x is None:
            # Fallback if something went wrong
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made += 1

        x = best_x
        fx = best_f
        sigma = 0.5 # Start with a small step size for polishing
        
        # Run local ES for remaining budget
        while calls_made < budget:
            y = np.clip(x + sigma * np.random.default_rng(0).standard_normal(dim), -5, 5)
            fy = func(y)
            calls_made += 1
            
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.2 # Increase step if improving
            else:
                sigma *= 0.8 # Decrease step if not improving
                # Optional: If sigma gets too small, we might stop early to save time?
                # But we have budget to spare, so let it run.
                
        return x, fx
# EVOLVE-BLOCK-END
