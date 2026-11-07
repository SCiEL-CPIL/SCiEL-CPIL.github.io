"""Optimizer using Differential Evolution for global search and SLSQP for local polishing."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time


class Optimizer:
    """Hybrid DE + SLSQP optimizer for box-constrained minimization on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        call_count = [0]

        def wrapped_func(x):
            if time.time() > deadline:
                raise TimeoutError()
            call_count[0] += 1
            val = float(func(x))
            return val

        def time_ok():
            return time.time() < deadline

        def run_slsqp(x0, maxiter=100):
            """Run SLSQP from x0, return (success, result)."""
            try:
                res = minimize(
                    wrapped_func,
                    x0,
                    method='SLSQP',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': maxiter, 'ftol': 1e-12}
                )
                return True, res
            except Exception:
                return False, None

        # Phase 1: Global Search with Differential Evolution
        # DE is robust for Rosenbrock's non-convex landscape.
        # We allocate ~60% of the budget to DE.
        de_budget = int(budget * 0.6)
        
        # differential_evolution uses a population. 
        # popsize=15 is default. 
        # maxiter controls the number of generations.
        # Total calls approx = popsize * dim * maxiter (roughly, depends on strategy)
        # Actually, DE call count is tricky to predict exactly, so we rely on the wrapper and time check.
        
        try:
            # tol and maxiter help stop it early if converged
            de_res = differential_evolution(
                wrapped_func,
                bounds=[(-5, 5)] * dim,
                maxiter=50, # Limit iterations to prevent timeout
                popsize=15,
                tol=1e-8,
                seed=42
            )
            if de_res.fun < best_f:
                best_f = de_res.fun
                best_x = de_res.x.copy()
        except Exception:
            # If DE fails or times out, we might have partial results in best_f if we tracked it, 
            # but DE doesn't return partial bests easily via standard API unless we use callback.
            # For simplicity, if it fails, we fall back to random starts below.
            pass

        # Check if we have a good candidate from DE
        if best_x is None or not time_ok() or call_count[0] >= budget - 20:
             # Fallback: if DE didn't find anything or used all budget, do a quick random search
             if best_x is None:
                 rng = np.random.default_rng(42)
                 for _ in range(10):
                     if not time_ok() or call_count[0] >= budget - 5:
                         break
                     x0 = rng.uniform(-4, 4, dim)
                     f0 = wrapped_func(x0)
                     if f0 < best_f:
                         best_f = f0
                         best_x = x0.copy()
        
        # Phase 2: Local Polishing with SLSQP
        # SLSQP has quadratic convergence, ideal for the final steps.
        if best_x is not None and time_ok() and call_count[0] < budget - 10:
            success, res = run_slsqp(best_x, maxiter=100)
            if success and res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()

        # Phase 3: If we have budget left, try a few random restarts with SLSQP
        # This helps if DE got stuck in a local minimum.
        rng = np.random.default_rng(123)
        n_extra_starts = 5
        for i in range(n_extra_starts):
            if not time_ok() or call_count[0] >= budget - 5:
                break
            
            # Start near the best solution or random
            if i % 2 == 0 and best_x is not None:
                x0 = best_x + rng.normal(0, 1.0, dim)
            else:
                x0 = rng.uniform(-4, 4, dim)
            
            x0 = np.clip(x0, -5, 5)
            
            success, res = run_slsqp(x0, maxiter=50)
            if success and res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()

        # Fallback if nothing was found
        if best_x is None:
            best_x = np.zeros(dim)
            if call_count[0] < budget:
                best_f = wrapped_func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
