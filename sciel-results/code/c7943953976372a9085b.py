"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """(1+1)-ES with adaptive sigma for BBOB-style Sphere f1."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        t0 = time.time()
        deadline = t0 + 285  # Safety margin before 300s limit

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

        # Phase 1: Global exploration with random sampling
        # Use a small budget for random sampling to find a good basin.
        # Since the domain is [-5, 5]^5 and minimizer is in [-4, 4]^5,
        # random sampling in [-4, 4]^5 is effective.
        n_global = min(20, total_budget // 5)
        
        for _ in range(n_global):
            if time.time() > deadline or calls_made >= total_budget:
                break
            x = np.random.uniform(-4.0, 4.0, dim)
            safe_call(x)
            
        if best_x is None:
            # Fallback if no calls were made (should not happen)
            x0 = np.zeros(dim)
            safe_call(x0)
            if best_x is None:
                return None, np.inf

        # Phase 2: Local refinement with (1+1)-ES
        # (1+1)-ES is simple, fast, and effective for smooth unimodal functions like Sphere
        x = best_x.copy()
        fx = best_f
        sigma = 1.0  # Initial step size
        
        # (1+1)-ES loop
        while calls_made < total_budget and time.time() < deadline:
            # Generate offspring
            y = np.clip(x + sigma * np.random.randn(dim), -5.0, 5.0)
            fy = safe_call(y)
            
            if calls_made >= total_budget:
                break
                
            # Selection and adaptation
            if fy < fx:
                # Success: move to offspring and increase step size
                x = y
                fx = fy
                sigma *= 1.2  # Increase step size on success
            else:
                # Failure: decrease step size
                sigma *= 0.8  # Decrease step size on failure
            
            # Convergence check: if sigma becomes very small, we've converged
            if sigma < 1e-8:
                break
            
            # Additional convergence check: if no improvement in last 100 calls
            # (This is implicitly handled by the sigma adaptation)

        return best_x, best_f

# EVOLVE-BLOCK-END
