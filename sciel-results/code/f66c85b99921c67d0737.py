"""Multi-start Nelder-Mead with Coordinate Descent polish for BBOB f22."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

class Optimizer:
    """Multi-start Nelder-Mead optimizer with coordinate-descent polish for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0

        dim = self.dim
        budget = self.budget
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)

        def eval_and_track(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return best_f
            calls += 1
            try:
                val = func(np.asarray(x, dtype=float))
                if not np.isfinite(val):
                    val = best_f
                if val < best_f:
                    best_f = val
                    best_x = np.asarray(x, dtype=float).copy()
                return val
            except Exception:
                return best_f

        rng = np.random.default_rng(12345)

        def coordinate_descent_polish(x, n_sweeps=3, search_radius=0.5):
            """Polish using 1D golden section search on each coordinate."""
            nonlocal calls, best_f, best_x
            for sweep in range(n_sweeps):
                if calls >= budget or time.time() > deadline:
                    break
                order = rng.permutation(dim)
                for axis in order:
                    if calls >= budget or time.time() > deadline:
                        break
                    # Define search interval
                    lo = max(-5.0, x[axis] - search_radius)
                    hi = min(5.0, x[axis] + search_radius)
                    
                    # Golden section search
                    gr = (1.0 + math.sqrt(5.0)) / 2.0
                    inv_gr = 1.0 / gr
                    inv_gr2 = 1.0 / (gr * gr)
                    
                    x1 = x.copy()
                    x1[axis] = lo + inv_gr2 * (hi - lo)
                    x2 = x.copy()
                    x2[axis] = lo + inv_gr * (hi - lo)
                    
                    f1 = eval_and_track(x1)
                    f2 = eval_and_track(x2)
                    
                    # Iterate
                    for _ in range(8):
                        if calls >= budget or time.time() > deadline:
                            break
                        if f1 < f2:
                            hi = x2[axis]
                            x2 = x1.copy()
                            f2 = f1
                            x1 = x.copy()
                            x1[axis] = lo + inv_gr2 * (hi - lo)
                            f1 = eval_and_track(x1)
                        else:
                            lo = x1[axis]
                            x1 = x2.copy()
                            f1 = f2
                            x2 = x.copy()
                            x2[axis] = lo + inv_gr * (hi - lo)
                            f2 = eval_and_track(x2)
                    
                    # Update x to best point found in this coordinate
                    if f1 < f2:
                        x = x1
                    else:
                        x = x2
            return x

        # Phase 1: Initialization
        # Use a mix of random and structured points
        n_starts = 8
        candidates = []
        
        # Structured points
        pts = [
            np.zeros(dim),
            rng.uniform(-5, 5, size=dim),
            rng.uniform(-5, 5, size=dim),
            rng.uniform(-5, 5, size=dim),
            rng.uniform(-5, 5, size=dim),
            rng.uniform(-5, 5, size=dim),
            rng.uniform(-5, 5, size=dim),
            rng.uniform(-5, 5, size=dim),
        ]
        
        for p in pts:
            if calls >= budget: break
            x = np.clip(p, -5, 5)
            f_val = eval_and_track(x)
            candidates.append((f_val, x.copy()))

        candidates.sort(key=lambda t: t[0])

        # Phase 2: Nelder-Mead on top candidates
        from scipy.optimize import minimize
        
        for i in range(min(5, len(candidates))):
            if calls >= budget or time.time() > deadline:
                break
            
            x0 = candidates[i][1].copy()
            remaining = budget - calls
            if remaining < 20:
                break
            
            # Allocate budget
            nm_budget = min(remaining, 60)
            
            try:
                res = minimize(
                    eval_and_track,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': nm_budget,
                        'maxfev': nm_budget,
                        'xatol': 1e-7,
                        'fatol': 1e-9,
                        'adaptive': True,
                    },
                    bounds=[(-5, 5)] * dim,
                )
            except Exception:
                pass

        # Phase 3: Polish
        if calls < budget and time.time() < deadline:
            x = best_x.copy()
            # Moderate radius polish
            x = coordinate_descent_polish(x, n_sweeps=2, search_radius=0.5)
            eval_and_track(x)
            
            # Tight radius polish
            if calls < budget and time.time() < deadline:
                x = best_x.copy()
                x = coordinate_descent_polish(x, n_sweeps=2, search_radius=0.1)
                eval_and_track(x)

        return best_x, best_f
# EVOLVE-BLOCK-END
