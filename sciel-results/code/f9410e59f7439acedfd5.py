import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0
        n = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return best_f
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def run_lbfgsb(x0, max_iter, gtol=1e-8):
            nonlocal calls
            if calls >= budget or time.time() >= deadline:
                return
            try:
                def objective(x):
                    return evaluate(x)
                
                minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * n,
                    options={
                        'maxiter': max_iter,
                        'gtol': gtol
                    }
                )
            except Exception:
                pass

        def run_nm(x0, max_iter, xatol=1e-6, fatol=1e-8):
            nonlocal calls
            if calls >= budget or time.time() >= deadline:
                return
            try:
                def objective(x):
                    return evaluate(x)
                
                minimize(
                    objective,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_iter,
                        'xatol': xatol,
                        'fatol': fatol,
                        'adaptive': True
                    }
                )
            except Exception:
                pass

        # Phase 1: Generate diverse starting points using Sobol sequence
        # 32 points is sufficient for diversity in 5D while leaving budget for optimization
        sobol_sample_count = 32
        sobol = qmc.Sobol(d=n, scramble=True, seed=42)
        sobol_points = sobol.random(sobol_sample_count)
        sobol_points = sobol_points * 10.0 - 5.0
        
        # Evaluate starting points to establish a baseline and sort them
        # We don't need to evaluate all if we run optimizers from all, but 
        # sorting by initial value helps prioritize better starts if budget is tight.
        # However, for multi-start, we typically run optimizer from each start.
        # Let's run L-BFGS-B from each start point.
        
        starts = []
        for point in sobol_points:
            if calls >= budget or time.time() >= deadline:
                break
            # Evaluate the start point itself
            fx = evaluate(point)
            starts.append((fx, point.copy()))
            
        # Sort starts by function value to prioritize promising regions
        starts.sort(key=lambda x: x[0])
        
        # Phase 2: Multi-restart L-BFGS-B and Nelder-Mead refinement
        # Run L-BFGS-B from each start point, then NM for polish
        # We limit the number of restarts to ensure we have budget for polishing
        # If budget allows, we run from all 32. If not, we stop early.
        
        num_restarts = len(starts)
        
        for i in range(num_restarts):
            if calls >= budget or time.time() >= deadline:
                break
            
            fx_start, x_start = starts[i]
            
            # 1. L-BFGS-B for fast convergence on smooth landscape
            # Allocate ~30 iterations or until budget/time runs out
            run_lbfgsb(x_start, max_iter=50)
            
            # 2. Nelder-Mead for local polishing
            # Use the best point found so far (global or local)
            if best_x is not None and calls < budget and time.time() < deadline:
                run_nm(best_x, max_iter=30)
                
        # Phase 3: Final polish on the global best
        if best_x is not None and calls < budget and time.time() < deadline:
            remaining = budget - calls
            if remaining > 10:
                run_lbfgsb(best_x, max_iter=remaining, gtol=1e-10)
            
            if best_x is not None and calls < budget and time.time() < deadline:
                remaining = budget - calls
                if remaining > 5:
                    run_nm(best_x, max_iter=remaining, xatol=1e-8, fatol=1e-10)

        return best_x, best_f
