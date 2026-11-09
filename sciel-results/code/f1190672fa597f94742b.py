"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # State variables
        best_x = np.zeros(dim)
        best_f = float('inf')
        call_count = 0
        
        # Wrapper to count calls and enforce budget
        def counted_func(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget:
                return best_f
            call_count += 1
            try:
                f_val = func(x)
                if f_val < best_f:
                    best_f = f_val
                    best_x = np.array(x)
                return f_val
            except:
                return 1e100

        bounds = [(-5, 5)] * dim
        
        # Strategy:
        # 1. Global Search: Use Latin Hypercube Sampling (LHS) to efficiently cover the [-5, 5]^5 space.
        #    This ensures we find the valley of the rotated Rosenbrock function early, which is
        #    critical for AOCC (Area Over the Curve) since early progress is weighted heavily.
        #    We allocate ~40% of the budget to this phase.
        # 2. Local Polish: Use L-BFGS-B from the top few candidates found in the global phase.
        #    L-BFGS-B is extremely effective for Rosenbrock once in the valley, allowing us to
        #    reach high precision with the remaining budget.
        
        # Phase 1: Global Search
        # Allocate 40% of budget to global search, minimum 50 calls to ensure coverage.
        n_global = max(50, int(budget * 0.4))
        
        # Use Latin Hypercube Sampling for better space-filling properties than uniform random.
        sampler = qmc.LatinHypercube(d=dim, seed=123)
        samples = sampler.random(n=n_global)
        
        # Scale from [0, 1] to [-5, 5]
        samples = samples * 10 - 5
        
        # Evaluate global samples
        candidates = []
        for x in samples:
            if call_count >= budget:
                break
            f_val = counted_func(x)
            candidates.append((f_val, x.copy()))
            
        # Sort candidates by function value
        candidates.sort(key=lambda item: item[0])
        
        # Phase 2: Local Polish
        # Select top K candidates for polishing.
        # K=5 provides a good balance between diversity and convergence.
        K = 5
        top_candidates = [x for f, x in candidates[:K]]
        
        # Allocate remaining budget to polishing.
        # We'll try to run L-BFGS-B from each top candidate.
        # To manage budget, we limit iterations and check time/budget frequently.
        
        for i, x0 in enumerate(top_candidates):
            if call_count >= budget - 10:  # Leave a small buffer
                break
            if time.time() > deadline:
                break
                
            try:
                # Use a tighter tolerance for later iterations if budget allows?
                # For now, standard L-BFGS-B settings.
                res = minimize(
                    counted_func, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=bounds, 
                    options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-10}
                )
                # Update best if improved (counted_func already updates best_x/best_f)
            except Exception:
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
