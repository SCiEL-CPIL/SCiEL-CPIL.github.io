"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B optimizer with time guard for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        d = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)
        calls_made = 0
        best_x = None
        best_f = np.inf

        # Phase 1: Random sampling to find good starting points
        num_samples = 20
        for i in range(num_samples):
            if not time_ok() or calls_made >= budget:
                break
            x = rng.uniform(-5, 5, d)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(d)
            best_f = func(best_x)
            calls_made += 1

        # Phase 2: Multi-start L-BFGS-B from multiple initial points
        # Use the best few points from random sampling as starting points
        # Collect top starting points
        candidates = []
        for i in range(min(10, calls_made)):
            if not time_ok() or calls_made >= budget:
                break
            # Use points from our random sampling
            x = rng.uniform(-5, 5, d)
            fx = func(x)
            calls_made += 1
            candidates.append((x.copy(), fx))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Sort candidates by function value
        candidates.sort(key=lambda t: t[1])

        # Run L-BFGS-B from top candidates
        num_starts = min(5, len(candidates))
        for i in range(num_starts):
            if not time_ok() or calls_made >= budget:
                break
            
            x0 = candidates[i][0]
            remaining = budget - calls_made
            if remaining < 10:
                break
            
            # Limit max iterations based on remaining budget
            max_iter = min(remaining - 5, 200)
            if max_iter < 10:
                break
            
            try:
                result = minimize(
                    func,
                    x0,
                    method='L-BFGS-B',
                    options={
                        'maxiter': max_iter,
                        'maxfun': max_iter,
                        'gtol': 1e-8,
                        'ftol': 1e-10
                    }
                )
                calls_made += result.get('nfev', max_iter)
                
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
                    
                # If we found a good solution, stop early
                if best_f < -50:
                    break
                    
            except Exception:
                pass

        # Phase 3: Final polish with Nelder-Mead if budget remains
        if calls_made < budget and time_ok() and best_x is not None:
            remaining = budget - calls_made
            if remaining > 10:
                try:
                    nm_budget = min(remaining - 5, 50)
                    result = minimize(
                        func,
                        best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': nm_budget,
                            'xatol': 1e-8,
                            'fatol': 1e-10
                        }
                    )
                    calls_made += result.get('nfev', nm_budget)
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
