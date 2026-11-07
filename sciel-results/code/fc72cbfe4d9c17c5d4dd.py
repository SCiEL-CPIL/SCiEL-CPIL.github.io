import numpy as np
import scipy.optimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0 # Safety margin
        
        rng = np.random.default_rng(42)
        
        # Phase 1: Coarse Random Search
        n_random = 50
        candidates = []
        best_f = float('inf')
        best_x = None
        
        for i in range(n_random):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x
            candidates.append((fx, x))
            
        # Keep top 5 for local refinement
        candidates.sort(key=lambda t: t[0])
        top_candidates = [c[1] for c in candidates[:5]]
        
        # Phase 2: Local Refinement
        # We have budget - n_random evaluations left
        remaining_budget = self.budget - n_random
        # Distribute remaining budget among top candidates
        # Let's say 5 candidates, ~50 evals each
        evals_per_candidate = max(10, remaining_budget // len(top_candidates))
        
        call_counter = [n_random]
        
        def wrapped_func(x):
            if call_counter[0] >= self.budget:
                return 1e10
            call_counter[0] += 1
            # Clip to bounds just in case, though minimize should handle it
            x_clipped = np.clip(x, -5, 5)
            return func(x_clipped)
            
        for start_x in top_candidates:
            if time.time() > deadline:
                break
            if call_counter[0] >= self.budget:
                break
                
            try:
                res = scipy.optimize.minimize(
                    wrapped_func, 
                    start_x, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 10, 'maxfun': evals_per_candidate}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_f
