"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-start local search with structure-aware initialization for Büche-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 285  # Safety margin for 300s limit
        
        # Call counter
        call_count = 0
        max_calls = self.budget
        
        best_x = None
        best_f = float('inf')
        
        # Wrapper to count calls and enforce budget
        def counted_func(x):
            nonlocal call_count
            if call_count >= max_calls:
                raise ValueError("Budget exceeded")
            call_count += 1
            x_clipped = np.clip(x, -5, 5)
            return func(x_clipped)

        # Strategy: Büche-Rastrigin has minima near integer coordinates in the rotated frame.
        # Since we don't know the rotation, we use a combination of:
        # 1. Points near integers (where Rastrigin minima tend to be)
        # 2. Center point
        # 3. Random points for exploration
        
        # Generate candidate starting points
        candidates = []
        
        # Center
        candidates.append(np.zeros(self.dim))
        
        # Points near integers: sample from {-2, -1, 0, 1, 2}^d with small perturbations
        # This exploits the periodic structure of Rastrigin
        int_vals = np.array([-2, -1, 0, 1, 2])
        for _ in range(8):
            base = rng.choice(int_vals, size=self.dim)
            pert = rng.uniform(-0.3, 0.3, size=self.dim)
            candidates.append(np.clip(base + pert, -5, 5))
        
        # Some random points for exploration
        for _ in range(4):
            candidates.append(rng.uniform(-5, 5, size=self.dim))
        
        # Shuffle to mix exploration and exploitation
        rng.shuffle(candidates)
        
        # Phase 1: Quick exploration with Nelder-Mead (faster than Powell for low-dim)
        # Use a smaller budget per run to allow more restarts
        nm_budget_per_run = 30
        nm_runs = min(len(candidates), max(1, (max_calls - 50) // nm_budget_per_run))
        
        for i in range(nm_runs):
            if call_count >= max_calls - 30:  # Reserve 30 calls for polish
                break
            if time.time() > deadline - 10:
                break
                
            x0 = candidates[i]
            
            # Run Nelder-Mead with limited iterations
            try:
                res = minimize(counted_func, x0, method='Nelder-Mead',
                               options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4,
                                        'initial_simplex': 0.5})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 2: Polish the best point with tighter tolerance
        if best_x is not None and call_count < max_calls - 10 and time.time() < deadline - 5:
            try:
                res = minimize(counted_func, best_x, method='Powell',
                               options={'maxiter': 200, 'xtol': 1e-8, 'ftol': 1e-10})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 3: If we have budget left, try a few more random starts with quick NM
        if call_count < max_calls - 20 and time.time() < deadline - 5:
            for _ in range(3):
                if call_count >= max_calls - 10:
                    break
                if time.time() > deadline - 3:
                    break
                x0 = rng.uniform(-5, 5, size=self.dim)
                try:
                    res = minimize(counted_func, x0, method='Nelder-Mead',
                                   options={'maxiter': 30, 'xatol': 1e-4, 'fatol': 1e-4})
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

        # If no successful optimization, return a default point
        if best_x is None:
            best_x = np.zeros(self.dim)
            if call_count < max_calls:
                best_f = func(best_x)
            else:
                best_f = float('inf')
            
        return best_x, best_f
# EVOLVE-BLOCK-END
