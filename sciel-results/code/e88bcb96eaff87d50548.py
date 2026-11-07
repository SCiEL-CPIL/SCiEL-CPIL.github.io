"""Optimizer for BBOB-style f11 (Discus) using multi-start Powell and L-BFGS-B polishing."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    """Multi-start derivative-free (Powell) and L-BFGS-B optimizer for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _sobol_sample(self, n, dim, seed=0):
        """Generate quasi-random points in [-5, 5]^dim using a simple Halton-like sequence."""
        rng = np.random.default_rng(seed)
        points = np.zeros((n, dim))
        # Use stratified sampling with jitter for better coverage
        for d in range(dim):
            # Base points for stratification
            base = np.linspace(0, 1, n + 1)
            # Add jitter
            jitter = rng.random(n) * (1.0 / n)
            pts = base[:n] + jitter
            # Scale to [-5, 5]
            points[:, d] = -5.0 + 10.0 * pts
        return points

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        bounds = [(-5.0, 5.0)] * dim
        bounds_np = (np.full(dim, -5.0), np.full(dim, 5.0))
        
        # Track function calls
        call_tracker = [0]
        
        def wrapped_func(x):
            call_tracker[0] += 1
            return func(x)
        
        best_x = None
        best_f = np.inf
        
        # Phase 1: Initial global exploration
        # Use Sobol-like sampling for better coverage
        n_initial = min(12, budget // 50)
        
        # Generate diverse starting points
        rng = np.random.default_rng(42)
        starts = self._sobol_sample(n_initial, dim, seed=42)
        
        # Add a few random points and center
        random_starts = rng.uniform(-4, 4, size=(3, dim))
        center = np.zeros(dim)
        
        all_starts = np.vstack([starts, random_starts, [center]])
        all_starts = np.clip(all_starts, bounds_np[0], bounds_np[1])
        
        # Evaluate all starting points
        start_values = []
        for x0 in all_starts:
            if call_tracker[0] >= budget - 20:
                break
            if time.time() > deadline:
                break
            
            fx0 = wrapped_func(x0)
            start_values.append((fx0, x0.copy()))
            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()
        
        # Sort by function value
        start_values.sort(key=lambda x: x[0])
        
        # Phase 2: Derivative-free local search (Powell) from top starting points
        # Powell's method is robust to non-smoothness and doesn't require gradients
        n_local_starts = min(3, len(start_values))
        
        for i in range(n_local_starts):
            if call_tracker[0] >= budget - 50:
                break
            if time.time() > deadline:
                break
            
            fx0, x0 = start_values[i]
            
            try:
                # Run Powell's method (derivative-free)
                # maxiter is limited to ensure we don't consume all budget
                # Powell can be slow, so we set a reasonable maxiter
                result = minimize(
                    wrapped_func,
                    x0,
                    method='Powell',
                    bounds=bounds,
                    options={'maxiter': 50, 'xtol': 1e-4, 'ftol': 1e-4}
                )
                
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
                    
            except Exception:
                continue
        
        # Phase 3: L-BFGS-B polishing from the best point
        # L-BFGS-B is very efficient for smooth functions near the optimum
        if best_x is not None and call_tracker[0] < budget - 10:
            if time.time() < deadline:
                try:
                    result = minimize(
                        wrapped_func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 100, 'maxls': 20}
                    )
                    
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                        
                except Exception:
                    pass
        
        # Phase 4: Final high-precision polish if budget remains
        if best_x is not None and call_tracker[0] < budget - 5:
            if time.time() < deadline:
                try:
                    result = minimize(
                        wrapped_func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 50, 'maxls': 10, 'gtol': 1e-8}
                    )
                    
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                        
                except Exception:
                    pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
