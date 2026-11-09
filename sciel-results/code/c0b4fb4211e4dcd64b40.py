"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Empirical Pareto front discovery via multi-start local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Strategy: 
        # 1. Use multiple random starting points.
        # 2. For each start, minimize the sum of objectives using Nelder-Mead.
        # 3. This empirically finds points on the Pareto front without assuming 
        #    the specific location of g's minimum or the structure of the front.
        # 4. Collect the resulting points.
        
        # We have budget=200 evaluations.
        # Nelder-Mead is expensive in function calls.
        # Let's use a smaller number of starts, but allow enough iterations.
        # If we use 50 starts, and each takes ~3-5 evaluations on average to converge 
        # (or hit maxiter), we stay well within budget.
        # However, Nelder-Mead can take more. 
        # Let's cap total function calls strictly.
        
        total_evals = 0
        max_total_evals = self.budget
        results = []
        
        # Number of local searches to perform
        # We want to cover the front well.
        n_starts = 80
        
        # Define the objective function for local search
        def scalar_obj(x):
            nonlocal total_evals
            if total_evals >= max_total_evals:
                return 0.0 # Dummy return if budget exhausted
            if time.time() > deadline:
                return 0.0
            total_evals += 1
            try:
                f = problem(x)
                return np.sum(f)
            except:
                return 1e10

        # Run local searches
        for i in range(n_starts):
            if time.time() > deadline - 5:
                break
            if total_evals >= max_total_evals:
                break
            
            # Random start point
            x0 = lo + (hi - lo) * np.random.rand(self.dim)
            
            # Nelder-Mead settings
            # maxiter limits the number of function evaluations per call
            # We give it a reasonable limit, but the global counter enforces the hard cap
            options = {
                'maxiter': 20,
                'xatol': 1e-3,
                'fatol': 1e-3
            }
            
            try:
                res = minimize(scalar_obj, x0, method='Nelder-Mead', options=options)
                # res.x is the best point found
                # We should evaluate res.x explicitly to record it, 
                # but scalar_obj already counted it during the optimization.
                # However, the exact point res.x might not have been the last one evaluated 
                # in a way that is clearly "recorded" as a distinct point if we just 
                # rely on the internal calls. 
                # The task says "Every returned objective vector is recorded".
                # The problem function is called inside scalar_obj. 
                # So every call to problem(x) inside minimize is recorded by the evaluator.
                # We don't need to call problem(res.x) again, it's already in the log.
                
                # We just need to ensure we don't exceed budget.
                # The loop continues.
            except Exception:
                pass

        # If we have time and budget left, we could add some random points to ensure coverage
        # but local search should have covered the front reasonably well.
        
        # The return value is ignored, but we must return something.
        # The evaluator records all calls to problem().
        return None

# EVOLVE-BLOCK-END
