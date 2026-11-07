"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from scipy.optimize import minimize
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid Halton initialization with Powell local search."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0
        
        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def func_wrapper(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                return 1e10
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def callback(xk):
            nonlocal calls
            if calls >= budget or time.time() > deadline:
                return True
            return False

        # Phase 1: Global Sampling
        n_init = 20
        try:
            from scipy.stats import qmc
            sampler = qmc.Halton(dim, seed=42)
            points = sampler.random(n_init) * 8 - 4
        except ImportError:
            rng = np.random.default_rng(42)
            points = rng.uniform(-4, 4, (n_init, dim))

        init_results = []
        for p in points:
            if calls >= budget or time.time() > deadline:
                break
            fx = func_wrapper(p)
            if fx < 1e9:
                init_results.append((fx, p.copy()))

        if not init_results:
            return best_x, best_f

        init_results.sort(key=lambda x: x[0])
        
        # Phase 2: Local Refinement
        top_k = 3
        top_candidates = [x for f, x in init_results[:top_k]]
        
        for x0 in top_candidates:
            if calls >= budget or time.time() > deadline:
                break
            
            try:
                res = minimize(func_wrapper, x0, method='Powell', 
                               callback=callback,
                               options={'maxiter': 100})
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
