"""Optimizer for BBOB-style f13 (Sharp ridge) - direct approach with structured search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc


class Optimizer:
    """Structured optimizer for BBOB f13: grid + adaptive local search."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0
        n = self.dim
        budget = self.budget

        def elapsed():
            return time.time() - t0

        def remaining():
            return deadline - time.time()

        best_x = np.zeros(n)
        best_f = np.inf
        calls = 0

        def try_func(x):
            nonlocal best_f, best_x, calls
            if calls >= budget or elapsed() > deadline:
                return best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f = float(func(x))
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Structured grid search with increasing resolution
        # For f13 (Sharp ridge), the function is:
        # f(x) = 4*(x1 - x2)^2 + 0.1*(x1 + x2)^2 + sum(4*(x_i - x_{i+1})^2 + 0.1*(x_i + x_{i+1})^2 for i=2..n-1)
        # The minimum is along the line x1 = x2 = ... = xn = c for some c
        # But with hidden shift, rotation, and offset, we need to search.
        
        # Start with a coarse grid to find a good region
        # Use Latin hypercube / Sobol for initial sampling
        try:
            sobol = qmc.Sobol(d=n, scramble=True, seed=12345)
            sobol.random(1)  # skip origin
            n_init = min(100, budget // 5)
            pts = sobol.random(n_init) * 10.0 - 5.0
            for i in range(n_init):
                if calls >= budget or elapsed() > deadline:
                    break
                try_func(pts[i])
        except Exception:
            pass

        # Phase 2: Adaptive local search from best point with tighter tolerances
        if best_f < np.inf:
            def obj(x):
                return try_func(x)
            
            try:
                res = minimize(obj, best_x, method='L-BFGS-B', 
                              bounds=[(-5, 5)] * n,
                              options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-12})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0)
            except Exception:
                pass

        # Phase 3: Multi-start refinement with more diverse starting points
        if calls < budget and elapsed() < deadline:
            rng = np.random.default_rng(789)
            n_starts = min(30, (budget - calls) // 10)
            
            for s in range(n_starts):
                if calls >= budget or elapsed() > deadline:
                    break
                
                # Different scales for exploration - wider range
                if s < 10:
                    scale = 0.5 + s * 0.3
                elif s < 20:
                    scale = 3.5 + (s - 10) * 0.2
                else:
                    scale = 0.1 + (s - 20) * 0.05
                
                x0 = best_x + rng.standard_normal(n) * scale
                x0 = np.clip(x0, -5.0, 5.0)
                
                f0 = try_func(x0)
                if f0 >= best_f:
                    continue
                
                # Local refinement from this point with tighter convergence
                if calls < budget - 15:
                    def obj2(x):
                        return try_func(x)
                    
                    try:
                        res = minimize(obj2, x0, method='L-BFGS-B',
                                      bounds=[(-5, 5)] * n,
                                      options={'maxiter': 100, 'gtol': 1e-7, 'ftol': 1e-11})
                        if res.fun < best_f:
                            best_f = res.fun
                            best_x = np.clip(res.x, -5.0, 5.0)
                    except Exception:
                        pass

        # Phase 4: Fine-tuning with Nelder-Mead if budget remains
        if calls < budget - 30 and elapsed() < deadline:
            def obj3(x):
                return try_func(x)
            
            try:
                res = minimize(obj3, best_x, method='Nelder-Mead',
                              options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-12, 'initial_simplex_size': 0.3})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0)
            except Exception:
                pass

        # Phase 5: Final coordinate-wise polishing with adaptive step sizes
        if calls < budget - 20 and elapsed() < deadline:
            for scale in [0.1, 0.01, 0.001, 0.0001]:
                if calls >= budget or elapsed() > deadline:
                    break
                
                for dim_idx in range(n):
                    if calls >= budget or elapsed() > deadline:
                        break
                    
                    improved = True
                    while improved and calls < budget and elapsed() < deadline:
                        improved = False
                        for delta in [scale, -scale]:
                            if calls >= budget or elapsed() > deadline:
                                break
                            
                            x_new = best_x.copy()
                            x_new[dim_idx] += delta
                            x_new = np.clip(x_new, -5.0, 5.0)
                            
                            f_new = try_func(x_new)
                            if f_new < best_f:
                                improved = True
                                best_x = x_new.copy()
                                best_f = f_new
                                break

        # Phase 6: Final high-precision polish with very tight BFGS
        if calls < budget - 10 and elapsed() < deadline:
            def obj4(x):
                return try_func(x)
            
            try:
                res = minimize(obj4, best_x, method='L-BFGS-B',
                              bounds=[(-5, 5)] * n,
                              options={'maxiter': 300, 'gtol': 1e-10, 'ftol': 1e-14, 'maxls': 50})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0)
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
