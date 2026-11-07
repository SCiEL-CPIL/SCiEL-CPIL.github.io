# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

class Optimizer:
    """Multi-start L-BFGS-B optimizer for Bent Cigar.
    L-BFGS-B's quasi-Newton Hessian approximation efficiently scales steps
    along the narrow anisotropic valley. Multiple diverse restarts ensure
    global coverage while each local run converges rapidly."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 285.0
        dim = self.dim
        budget = self.budget
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)

        def wrapper(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            calls += 1
            try:
                x_clipped = np.clip(x, -5, 5)
                f_val = float(func(x_clipped))
                if f_val < best_f:
                    best_f = f_val
                    best_x = x_clipped.copy()
                return f_val
            except Exception:
                return float('inf')

        rng = np.random.default_rng(42)
        
        # Generate diverse starting points:
        # - Origin
        # - Axis-aligned points (helps if shift is along an axis)
        # - Random points in the box
        # - Points near the boundary (minimizer is in [-4,4]^5)
        
        start_points = [np.zeros(dim)]
        
        # Axis-aligned starts
        for axis in range(dim):
            for sign in [-1, 1]:
                pt = np.zeros(dim)
                pt[axis] = sign * 3.0
                start_points.append(pt)
        
        # Random points
        for _ in range(15):
            start_points.append(rng.uniform(-4, 4, dim))
        
        # Near-boundary points
        for _ in range(10):
            start_points.append(rng.uniform(-4.5, 4.5, dim))
        
        # Shuffle to mix strategies
        rng.shuffle(start_points)
        
        # Allocate budget per restart with diminishing returns
        # First few restarts get more budget, later ones less
        n_restarts = len(start_points)
        
        for i, x_start in enumerate(start_points):
            if calls >= budget - 15:
                break
            if time.time() > deadline:
                break
            
            # Allocate budget: first restart gets most, decreasing
            # Total budget is 500, so average ~33 per restart if using all
            # But we want early restarts to have more budget for convergence
            remaining = budget - calls
            if i == 0:
                alloc = min(60, remaining - 10)
            elif i < 5:
                alloc = min(45, remaining - 10)
            elif i < 15:
                alloc = min(30, remaining - 10)
            else:
                alloc = min(20, remaining - 10)
            
            if alloc < 10:
                break
            
            try:
                res = scipy.optimize.minimize(
                    wrapper,
                    x_start,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={
                        'maxiter': 200,
                        'maxfev': alloc,
                        'ftol': 1e-12,
                        'gtol': 1e-8,
                        'eps': 1e-6
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
