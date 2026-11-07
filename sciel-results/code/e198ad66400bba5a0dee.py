"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with derivative-free fallback for Discus (f11)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Count evaluations
        calls = [0]
        
        def wrapped_func(x):
            calls[0] += 1
            x = np.clip(x, -5.0, 5.0)
            return float(func(x))
        
        # Generate strategic starting points focused on the likely minimum region
        # For Discus, the minimum is typically near the origin after transformation
        n_starts = 8
        evals_per_run = max(20, (budget - n_starts) // n_starts)
        
        # Deterministic starting points: center + axis-aligned points at small radii
        start_points = []
        start_points.append(np.zeros(dim))
        
        # Points at small distances along axes (exploit local structure)
        for i in range(dim):
            for sign in [-1, 1]:
                p = np.zeros(dim)
                p[i] = sign * 1.0
                start_points.append(p)
        
        # Add a few points at moderate distance to catch broader features
        start_points.append(0.5 * np.ones(dim))
        start_points.append(-0.5 * np.ones(dim))
        
        # Select n_starts points, prioritizing center and axis points
        if len(start_points) > n_starts:
            # Keep center first, then take next n_starts-1
            start_points = [start_points[0]] + start_points[1:n_starts]
        else:
            while len(start_points) < n_starts:
                start_points.append(rng.uniform(-3, 3, dim))
        
        best_x = None
        best_f = np.inf
        
        # Run L-BFGS-B from each starting point with aggressive settings
        for idx, x0 in enumerate(start_points):
            if calls[0] >= budget - 5:
                break
            
            x0 = np.clip(x0, -5.0, 5.0)
            f0 = wrapped_func(x0)
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
            
            # Run L-BFGS-B with tight tolerances to exploit smoothness
            remaining_for_this = min(evals_per_run, budget - calls[0] - 5)
            if remaining_for_this < 5:
                break
                
            try:
                result = minimize(
                    wrapped_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={
                        'maxiter': 200,
                        'maxfun': remaining_for_this,
                        'gtol': 1e-10,
                        'ftol': 1e-14,
                        'maxls': 50
                    }
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5.0, 5.0)
            except Exception:
                pass
            
            # If we've found a very good solution, spend remaining budget on refinement
            if best_f < 1.0 and calls[0] < budget - 10:
                break
        
        # Refinement phase: use remaining budget for fine-tuning
        remaining = budget - calls[0]
        if remaining > 10 and best_x is not None:
            x = best_x.copy()
            fx = best_f
            
            # Use L-BFGS-B again from the best point with very tight tolerances
            if remaining > 20:
                try:
                    result = minimize(
                        wrapped_func,
                        x,
                        method='L-BFGS-B',
                        bounds=[(-5.0, 5.0)] * dim,
                        options={
                            'maxiter': 100,
                            'maxfun': remaining - 5,
                            'gtol': 1e-12,
                            'ftol': 1e-16
                        }
                    )
                    if result.fun < fx:
                        fx = result.fun
                        x = np.clip(result.x, -5.0, 5.0)
                except Exception:
                    pass
            
            # Final (1+1)-ES polish with very small steps
            remaining = budget - calls[0]
            if remaining > 5:
                sigma = 0.01
                for _ in range(remaining - 1):
                    if calls[0] >= budget:
                        break
                    y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                    fy = wrapped_func(y)
                    if fy < fx:
                        x, fx = y, fy
                        sigma *= 1.1
                    else:
                        sigma *= 0.7
                    
                    if sigma < 1e-8:
                        break
                
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
        
        # Ensure we made at least one call
        if calls[0] == 0:
            x0 = np.zeros(dim)
            best_f = wrapped_func(x0)
            best_x = x0.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
