"""Multi-start derivative-free optimizer: random exploration + Nelder-Mead refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-start Nelder-Mead optimizer with time guard.
    
    Strategy:
    1. Sample diverse starting points (random + quasi-random)
    2. Run Nelder-Mead from the best few starting points
    3. Track best solution found, respecting budget and time limits
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin
        n = self.dim
        rng = np.random.default_rng(123)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_func(x):
            nonlocal calls_made
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            calls_made += 1
            return func(x)

        # Phase 1: Global exploration with diverse starting points
        # Use a combination of random and quasi-random (Halton-like) points
        n_starts = min(20, max(5, self.budget // 50))
        
        starting_points = []
        
        # Random points
        for _ in range(n_starts // 2 + 1):
            x = rng.uniform(-4, 4, n)
            starting_points.append(x)
        
        # Quasi-random points (simple LHS-like)
        for i in range(n_starts // 2):
            x = np.zeros(n)
            for d in range(n):
                x[d] = -4 + 8 * ((i * (d + 1) + 1) / (n_starts // 2 + 1) % 1.0)
            x = np.clip(x, -4, 4)
            starting_points.append(x)
        
        # Center point
        starting_points.append(np.zeros(n))
        
        # Evaluate all starting points
        for x0 in starting_points:
            if time.time() > deadline or calls_made >= self.budget - 10:
                break
            f0 = safe_func(x0)
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
        
        # Phase 2: Local refinement with Nelder-Mead from best starting points
        # Sort starting points by fitness and refine the best few
        # We'll try Nelder-Mead from the best 3-5 points
        
        # Re-evaluate to get sorted list (we already have best_x, but let's try others)
        # Actually, let's just run Nelder-Mead from best_x and a few perturbations
        
        nm_trials = 5
        for trial in range(nm_trials):
            if time.time() > deadline or calls_made >= self.budget - 5:
                break
            
            if trial == 0:
                x0 = best_x.copy()
            else:
                # Perturb the best point
                x0 = best_x + rng.uniform(-1, 1, n)
                x0 = np.clip(x0, -5, 5)
            
            try:
                res = minimize(
                    safe_func,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': min(50, max(10, (self.budget - calls_made) // nm_trials)),
                        'xatol': 1e-8,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass
        
        # Phase 3: If we have budget left, try a few more random restarts
        remaining = self.budget - calls_made
        if remaining > 30 and time.time() < deadline - 30:
            n_more = min(5, remaining // 30)
            for _ in range(n_more):
                if time.time() > deadline or calls_made >= self.budget - 5:
                    break
                x0 = rng.uniform(-4, 4, n)
                try:
                    res = minimize(
                        safe_func,
                        x0,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 20,
                            'xatol': 1e-6,
                            'fatol': 1e-10,
                            'adaptive': True
                        }
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5, 5)
                except Exception:
                    pass
        
        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
