"""Optimizer for BBOB f19 (Griewank-Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """LHS global sampling + BFGS local polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # safety margin before 300s

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Phase 1: Stratified Latin Hypercube sampling for global coverage
        # Use a good fraction of the budget for global exploration
        n_lhs = min(200, budget // 3)
        
        # Generate LHS points
        lhs_points = np.zeros((n_lhs, dim))
        for i in range(n_lhs):
            for j in range(dim):
                lhs_points[i, j] = -5.0 + (i + rng.random()) / n_lhs * 10.0
        
        # Evaluate all LHS points
        best_x = None
        best_f = np.inf
        eval_count = 0
        
        for i in range(n_lhs):
            if time.time() > deadline - 60:
                break
            x = lhs_points[i]
            fx = func(x)
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 2: Local BFGS polish from the best point and some other good points
        # Sort by function value and pick top candidates for local search
        candidates = []
        for i in range(n_lhs):
            if time.time() > deadline - 60:
                break
            x = lhs_points[i]
            fx = func(x)
            eval_count += 1
            candidates.append((fx, x))
        
        candidates.sort(key=lambda t: t[0])
        
        # Polish the best few candidates with BFGS
        n_polish = min(5, len(candidates))
        for i in range(n_polish):
            if time.time() > deadline - 30:
                break
            if eval_count >= budget:
                break
            
            fx0, x0 = candidates[i]
            
            # Use a wrapper that tracks evaluations
            evals = [0]
            
            def wrapped_f(x):
                evals[0] += 1
                if evals[0] > budget - eval_count:
                    return func(x)  # still evaluate but we'll stop
                return func(x)
            
            # BFGS with reasonable options
            try:
                res = minimize(wrapped_f, x0, method='L-BFGS-B', 
                              bounds=[(-5.0, 5.0)] * dim,
                              options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
                if res.success and res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                eval_count += evals[0]
            except Exception:
                pass
            
            if time.time() > deadline - 30:
                break
        
        # Phase 3: Final refinement with very tight tolerance from best point
        if best_x is not None and time.time() < deadline - 10 and eval_count < budget:
            try:
                evals = [0]
                def wrapped_f2(x):
                    evals[0] += 1
                    return func(x)
                
                res = minimize(wrapped_f2, best_x, method='L-BFGS-B',
                              bounds=[(-5.0, 5.0)] * dim,
                              options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-10})
                if res.success and res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
