import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        d = self.dim
        budget = self.budget
        calls = [0]
        
        def wrapped(x):
            calls[0] += 1
            return float(func(x))
            
        def alive():
            return calls[0] < budget and time.time() < deadline

        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        
        # Phase 1: Multi-start Nelder-Mead to find good basins
        # BBOB f11 (Discus) is anisotropic. Nelder-Mead adapts shape well.
        # We use multiple starts to cover the domain.
        
        n_starts = 10
        # Spread starts across the domain [-4, 4]^d
        for i in range(n_starts):
            if not alive():
                break
            
            # Deterministic-ish spread for coverage, plus some randomness
            # Use a simple grid-like or random distribution
            # For d=5, we can use Latin Hypercube or just uniform random
            x0 = rng.uniform(-4, 4, d)
            
            # Initial simplex size
            # Nelder-Mead default is 0.05 * x0 or 0.00025 if x0 is 0
            # For a shifted function, a larger initial spread might be good, 
            # but NM can struggle if the basin is narrow.
            # Let's try a moderate initial step.
            step = 1.0
            
            options = {'maxiter': 50, 'maxfev': 50, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            
            try:
                res = minimize(wrapped, x0, method='Nelder-Mead', options=options, bounds=[(-5, 5)]*d)
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 2: Refine the best point with L-BFGS-B
        # L-BFGS-B is very efficient for smooth functions near the minimum.
        if alive() and best_x is not None:
            try:
                res = minimize(wrapped, best_x, method='L-BFGS-B', options={'maxiter': 50, 'maxfun': 100, 'ftol': 1e-12, 'gtol': 1e-12}, bounds=[(-5, 5)]*d)
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 3: If budget remains, run a final CMA-ES or more NM starts?
        # Given the time limit and budget (500 calls), and that NM/LBFGS are fast,
        # we might have used very few calls. 
        # The scoring is AOCC, which rewards early progress.
        # NM and LBFGS converge fast, so we should be well into the valley.
        
        # If we have plenty of budget left, we can do more multi-starts or refine further.
        # But usually, 10 NM starts * 50 calls = 500 calls. So we might be close to budget.
        # If we have budget left, let's do a few more refined starts around the best point.
        
        if alive() and best_x is not None:
            # Try a few local restarts around the best point to ensure we haven't missed a better local min
            for _ in range(5):
                if not alive():
                    break
                x0 = best_x + rng.normal(0, 0.5, d)
                x0 = np.clip(x0, -5, 5)
                try:
                    res = minimize(wrapped, x0, method='Nelder-Mead', options={'maxiter': 20, 'maxfev': 20, 'adaptive': True}, bounds=[(-5, 5)]*d)
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass
        
        return best_x, best_f
