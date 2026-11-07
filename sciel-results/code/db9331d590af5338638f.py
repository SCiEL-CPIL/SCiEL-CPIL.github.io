import numpy as np
import time
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        eval_count = 0
        
        def counting_func(x):
            nonlocal eval_count, best_f, best_x
            if eval_count >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x_clipped = np.clip(np.asarray(x, dtype=float), -5, 5)
            val = float(func(x_clipped))
            eval_count += 1
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val
        
        # Phase 1: Latin Hypercube Sampling for space-filling initialization
        n_initial = 40
        rng = np.random.default_rng(42)
        
        # Generate LHS points in [0,1]^dim
        lhs_points = np.zeros((n_initial, dim))
        for d in range(dim):
            # Divide [0,1] into n_initial strata, pick one random point per stratum
            strata = rng.permutation(n_initial)
            for i in range(n_initial):
                lhs_points[i, d] = (strata[i] + rng.random()) / n_initial
        
        # Map to [-5, 5]^dim
        lhs_points = lhs_points * 10.0 - 5.0
        
        # Evaluate initial points
        for pt in lhs_points:
            if eval_count >= budget or time.time() > deadline:
                break
            counting_func(pt)
        
        if best_x is None:
            return np.zeros(dim), 0.0
        
        # Phase 2: Multi-start CMA-ES from top-k initial points
        # Collect all evaluated points and their values
        # Re-evaluate to get sorted list (we only stored best, so do a quick re-scan)
        # Instead, let's just use the best point and a few perturbed versions
        
        remaining = budget - eval_count
        if remaining > 50:
            # Use CMA-ES with multiple restarts from promising regions
            n_restarts = 3
            per_restart_budget = remaining // (n_restarts + 1)  # +1 for final polish
            
            for r in range(n_restarts):
                if eval_count >= budget or time.time() > deadline:
                    break
                
                # Perturb the best point for diversity
                if r == 0:
                    start_pt = best_x.copy()
                    sigma0 = 1.5
                else:
                    # Add random perturbation scaled by distance from center
                    noise = rng.normal(0, 1.0, size=dim)
                    start_pt = best_x + noise * 0.5
                    start_pt = np.clip(start_pt, -5, 5)
                    sigma0 = 1.0
                
                # Evaluate starting point
                counting_func(start_pt)
                
                if eval_count >= budget or time.time() > deadline:
                    break
                
                # Run CMA-ES from this point
                try:
                    popsize = max(6, min(12, dim + 4))
                    max_iter = max(3, (per_restart_budget - eval_count) // popsize - 2)
                    
                    res = scipy.optimize.cma(
                        counting_func,
                        start_pt,
                        sigma0=sigma0,
                        options={
                            'maxiter': max_iter,
                            'popsize': popsize,
                            'tol': 1e-10,
                            'verbose': 0,
                            'bounds': [(-5.0, 5.0)] * dim,
                            'CMA_adaptation_rate': 0.5
                        }
                    )
                except Exception:
                    pass
        
        # Phase 3: Final local polish with Nelder-Mead
        remaining = budget - eval_count
        if remaining > 10 and best_x is not None:
            try:
                res_nm = scipy.optimize.minimize(
                    counting_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': min(remaining, 200),
                        'xatol': 1e-10,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
            except Exception:
                pass
        
        # Ensure we have a valid result
        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = float(func(best_x))
                eval_count += 1
            except Exception:
                best_f = 0.0
        
        return best_x, best_f
