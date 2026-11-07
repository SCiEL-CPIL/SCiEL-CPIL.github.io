"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution


class Optimizer:
    """Multi-start CMA-ES with L-BFGS-B polish for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start_time = time.time()
        time_limit = 285.0  # Safety margin for 300s limit
        calls_made = 0
        best_x = None
        best_f = np.inf

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                raise StopIteration
            calls_made += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def check_time():
            return time.time() - start_time < time_limit

        # Strategy: Use a few well-chosen starting points with CMA-ES-like
        # evolution, then polish with L-BFGS-B.
        # For Rosenbrock, the basin of attraction is narrow, so we need good
        # starting points. We'll use a combination of random starts and a
        # coarse grid search.

        dim = self.dim
        budget = self.budget

        # Phase 1: Coarse exploration with multiple short runs
        # Use 5 starting points with ~60 calls each for ES
        num_starts = 5
        es_budget_per_start = min(80, (budget - 100) // num_starts)
        
        # Starting points: mix of random and structured
        starts = []
        # One near center
        starts.append(np.zeros(dim))
        # Random points
        for i in range(num_starts - 1):
            starts.append(rng.uniform(-4, 4, dim))

        for s_idx, x0 in enumerate(starts):
            if not check_time() or calls_made >= budget - 50:
                break

            x = x0.copy()
            try:
                fx = safe_call(x)
            except StopIteration:
                break

            # Simple (1+1)-ES with adaptive sigma
            sigma = 1.0
            for it in range(es_budget_per_start):
                if not check_time() or calls_made >= budget - 50:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                try:
                    fy = safe_call(y)
                except StopIteration:
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.2, 5.0)
                else:
                    sigma = max(sigma * 0.9, 0.01)

        # Phase 2: Local polish with L-BFGS-B from the best point found
        if check_time() and calls_made < budget - 10 and best_x is not None:
            def lbfun(x_in):
                nonlocal calls_made
                if calls_made >= self.budget:
                    raise StopIteration
                calls_made += 1
                return func(x_in)

            try:
                res = minimize(lbfun, best_x, method='L-BFGS-B',
                               bounds=[(-5.0, 5.0) for _ in range(dim)],
                               options={'maxiter': 100, 'maxfun': budget - calls_made})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except StopIteration:
                pass
            except Exception:
                pass

        # Phase 3: If we have budget left, try another polish from a different
        # point if the first polish didn't converge well
        if check_time() and calls_made < budget - 10 and best_f > 1.0:
            # Try a second polish from a slightly perturbed best point
            x2 = np.clip(best_x + 0.5 * rng.standard_normal(dim), -5, 5)
            try:
                f2 = safe_call(x2)
                if f2 < best_f:
                    best_f = f2
                    best_x = x2.copy()

                def lbfun2(x_in):
                    nonlocal calls_made
                    if calls_made >= self.budget:
                        raise StopIteration
                    calls_made += 1
                    return func(x_in)

                res2 = minimize(lbfun2, best_x, method='L-BFGS-B',
                                bounds=[(-5.0, 5.0) for _ in range(dim)],
                                options={'maxiter': 50, 'maxfun': budget - calls_made})
                if res2.fun < best_f:
                    best_f = res2.fun
                    best_x = res2.x.copy()
            except StopIteration:
                pass
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
