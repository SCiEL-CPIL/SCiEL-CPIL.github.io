"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-start CMA-ES / L-BFGS hybrid with random restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # We'll track the best solution found so far
        best_x = None
        best_f = np.inf

        # Strategy: Use multiple short CMA-ES runs or L-BFGS from different starts
        # Since CMA-ES is expensive, and we have 500 calls, let's use a mix:
        # 1. Initial random sampling to get a decent start
        # 2. Then use L-BFGS-B from that start (which uses few function evaluations if gradient is estimated)
        # But L-BFGS-B with finite differences uses ~ (dim+1) calls per gradient step.
        # For dim=5, that's 6 calls per step. With 500 budget, we can do ~80 steps.
        # However, L-BFGS-B is local. So we need good starts.
        
        # Alternative: Use CMA-ES from scipy if available? No, scipy.optimize doesn't have CMA-ES.
        # Let's implement a simple CMA-ES or use Nelder-Mead?
        # Nelder-Mead is derivative-free and robust.
        # Or we can use a multi-start strategy with L-BFGS-B.
        
        # Let's try: 
        # Phase 1: 50 random evaluations to find a good starting point
        # Phase 2: Use L-BFGS-B from the best point found, with maxiter controlled by remaining budget
        
        # Actually, L-BFGS-B with finite differences: each gradient evaluation costs dim+1 = 6 function calls.
        # If we allow up to 100 gradient steps, that's 600 calls. Too many.
        # Let's limit gradient steps to (budget - 50) // 6.
        
        # But L-BFGS-B might converge faster. Let's try it.
        
        # First, do random search for 30 evaluations
        n_random = 30
        xs = rng.uniform(-5, 5, (n_random, dim))
        fs = np.array([func(x) for x in xs])
        idx = np.argmin(fs)
        best_x = xs[idx].copy()
        best_f = fs[idx]
        calls_used = n_random
        
        # Now use L-BFGS-B from best_x
        remaining = budget - calls_used
        # L-BFGS-B with finite differences: cost per gradient = dim + 1 = 6
        # We can afford at most remaining // 6 gradient evaluations
        # But L-BFGS-B may use fewer if it converges.
        
        def objective(x):
            nonlocal calls_used
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_used += 1
            return f

        bounds = [(-5, 5)] * dim
        
        # Use L-BFGS-B
        try:
            res = minimize(
                objective,
                best_x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'maxfun': remaining, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
        
        # If we still have budget, do more random restarts + L-BFGS
        while calls_used < budget:
            # Random restart
            x0 = rng.uniform(-5, 5, dim)
            f0 = func(x0)
            calls_used += 1
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
            
            remaining = budget - calls_used
            if remaining < 10:
                break
                
            def objective2(x):
                nonlocal calls_used
                x = np.clip(x, -5, 5)
                f = func(x)
                calls_used += 1
                return f
                
            try:
                res = minimize(
                    objective2,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 50, 'maxfun': remaining, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
