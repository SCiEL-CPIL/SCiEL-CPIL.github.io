import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf
        
        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Latin Hypercube Sampling for global basin identification
        n_lhs = 150
        # Generate LHS points
        # For each dimension, divide [-5, 5] into n_lhs intervals
        # Shuffle the order for each dimension
        lhs_points = np.zeros((n_lhs, dim))
        for d in range(dim):
            # Create uniform points in [0, 1)
            u = np.random.rand(n_lhs)
            # Stratify: sort and assign to intervals
            # Actually, standard LHS: for each dimension, create a permutation of [0..n-1]
            # Then point i in dim d is (perm[i] + U) / n
            perm = np.random.permutation(n_lhs)
            u_stratified = (perm + np.random.rand(n_lhs)) / n_lhs
            # Map to [-5, 5]
            lhs_points[:, d] = -5 + 10 * u_stratified
        
        # Evaluate LHS points
        for i in range(n_lhs):
            if calls >= budget - 50 or time.time() > deadline:
                break
            x = lhs_points[i]
            f = func(x)
            calls += 1
            update_best(x, f)
        
        # Phase 2: Local refinement with Nelder-Mead on top candidates
        # Select top 5 candidates from LHS
        # We need to track which points were evaluated
        # Since we didn't store them, let's just use best_x and maybe a few random perturbations?
        # Better: Store the LHS points and their values
        
        # Let's redo Phase 1 to store points
        # Actually, let's just do a second pass of smaller LHS or use the best found.
        # The directive says: "dense Latin Hypercube sampling ... followed by a fixed-step Nelder-Mead simplex method centered on the best sample"
        
        # So we just need to run Nelder-Mead on best_x.
        
        # But maybe we should do a few local searches from the best few points?
        # Let's keep it simple as per directive: "centered on the best sample"
        
        if calls < budget and time.time() < deadline:
            # Use Nelder-Mead with tight tolerances
            try:
                res = minimize(func, best_x, method='Nelder-Mead', 
                               options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass
        
        # If we have time, do a final polish with Powell or Nelder-Mead again with even tighter settings?
        # Or maybe do a small CMA-ES run to ensure robustness?
        # The directive says "avoid covariance adaptation overhead", so let's stick to LHS + NM.
        
        # However, Nelder-Mead can get stuck. Maybe do a few restarts if time permits?
        # Let's check time remaining.
        
        remaining_calls = budget - calls
        remaining_time = deadline - time.time()
        
        if remaining_calls > 50 and remaining_time > 10:
            # Try a few more local optimizations from slightly perturbed best_x
            for _ in range(3):
                if calls >= budget - 10 or time.time() > deadline:
                    break
                x0 = best_x + np.random.randn(dim) * 0.1
                x0 = np.clip(x0, -5, 5)
                try:
                    res = minimize(func, x0, method='Nelder-Mead', 
                                   options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x
                except Exception:
                    pass
        
        return best_x, best_f
