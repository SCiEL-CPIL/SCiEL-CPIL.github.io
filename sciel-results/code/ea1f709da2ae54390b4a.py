"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Portfolio of (1+1)-ES runs with simple 1/5 success rule."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(0)
        
        # Portfolio structure: multiple independent (1+1)-ES runs
        # Each run gets an equal share of the budget
        n_runs = 5
        calls_per_run = self.budget // n_runs  # 100 calls per run
        
        best_x = None
        best_f = np.inf
        total_calls = 0
        
        for run_idx in range(n_runs):
            if time.time() >= deadline:
                break
            if total_calls >= self.budget:
                break
            
            # Each run starts from a different random point
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            total_calls += 1
            sigma = 2.0
            
            # (1+1)-ES with 1/5 success rule for this run
            for _ in range(calls_per_run - 1):
                if time.time() >= deadline:
                    break
                if total_calls >= self.budget:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                total_calls += 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.5
                else:
                    sigma *= 0.87
            
            # Update global best
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        if best_x is None:
            # Fallback: should not happen, but be safe
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
