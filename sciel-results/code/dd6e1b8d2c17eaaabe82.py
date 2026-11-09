"""Rewritten optimizer for BBOB f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product


class Optimizer:
    """Deterministic multi-scale grid descent for f6."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            if time.time() >= deadline:
                return False
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return calls_made < budget

        # Phase 1: Coarse grid over [-5, 5]^5 with spacing 2.0
        # Points: -5, -3, -1, 1, 3, 5 -> 6^5 = 7776 points, too many
        # Use spacing 2.5: -5, -2.5, 0, 2.5, 5 -> 5^5 = 3125, still too many
        # Use a structured subset: start at center, expand outward
        # Actually, let's use a smarter approach: evaluate at strategic points
        
        # Phase 1: Evaluate at key strategic points (call 1-50)
        strategic_points = [
            np.zeros(dim),
            np.ones(dim) * 2.0,
            np.ones(dim) * -2.0,
            np.ones(dim) * 3.0,
            np.ones(dim) * -3.0,
            np.ones(dim) * 1.0,
            np.ones(dim) * -1.0,
            np.ones(dim) * 4.0,
            np.ones(dim) * -4.0,
            np.array([0, 2, 0, -2, 1]),
            np.array([2, 0, -2, 1, 0]),
            np.array([-2, 1, 0, 2, -1]),
            np.array([1, -2, 2, 0, -1]),
            np.array([0, 0, 2, 2, 0]),
            np.array([0, 0, -2, -2, 0]),
            np.array([2, 2, 0, 0, 0]),
            np.array([-2, -2, 0, 0, 0]),
            np.array([0, 2, 2, 0, 0]),
            np.array([0, -2, -2, 0, 0]),
            np.array([2, 0, 0, 2, 0]),
            np.array([-2, 0, 0, -2, 0]),
        ]
        
        for pt in strategic_points:
            if not evaluate(pt):
                break

        # Phase 2: Coarse-to-fine grid refinement around best point
        if best_x is not None and calls_made < budget:
            # Multi-scale refinement: step sizes 2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01
            step_sizes = [2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
            
            for step in step_sizes:
                if calls_made >= budget or time.time() >= deadline:
                    break
                
                # Generate grid points around best_x within bounds
                # For each dimension, try: best_x[i] - step, best_x[i], best_x[i] + step
                # But that's 3^dim points. For dim=5, that's 243 per step size.
                # We have ~450 calls left, so we can do about 1-2 full grids.
                
                # Use a more efficient approach: coordinate-wise search
                # For each step size, do a coordinate-wise line search
                for _ in range(3):  # 3 passes per step size
                    if calls_made >= budget or time.time() >= deadline:
                        break
                    
                    for d in range(dim):
                        if calls_made >= budget or time.time() >= deadline:
                            break
                        
                        # Try moving in + and - direction along dimension d
                        for direction in [1, -1]:
                            if calls_made >= budget or time.time() >= deadline:
                                break
                            
                            trial = best_x.copy()
                            trial[d] += direction * step
                            trial = np.clip(trial, -5, 5)
                            
                            if not evaluate(trial):
                                break
                            
                            # If improved, continue in same direction (momentum)
                            # Already updated best_x if improved
        
        # Phase 3: Fine local polish with Nelder-Mead if time permits
        if best_x is not None and calls_made < budget and time.time() < deadline - 10:
            try:
                from scipy.optimize import minimize
                x0 = best_x.copy()
                result = minimize(
                    lambda x: func(np.clip(x, -5, 5)),
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5, 5)
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
