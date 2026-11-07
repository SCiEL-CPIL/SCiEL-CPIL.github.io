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

        # Strategy: For Rosenbrock, the valley is narrow and roughly along
        # the diagonal. Use structured + random starts, then (1+1)-ES with
        # adaptive sigma, then multi-point L-BFGS-B polish.

        dim = self.dim
        budget = self.budget

        # Phase 1: Coarse exploration with multiple short runs
        num_starts = 6
        es_budget_per_start = min(70, (budget - 120) // num_starts)
        
        # Starting points: structured points along valley + random
        starts = []
        # Along the Rosenbrock valley direction (1,1,1,1,1) at various offsets
        valley_dir = np.ones(dim) / np.sqrt(dim)
        for offset in [0.0, 1.0, -1.0, 2.0]:
            starts.append(offset * valley_dir + rng.uniform(-0.5, 0.5, dim))
        # Two random points
        for i in range(2):
            starts.append(rng.uniform(-4, 4, dim))

        # Track top candidates for later polishing
        candidates = []  # list of (f, x)

        for s_idx, x0 in enumerate(starts):
            if not check_time() or calls_made >= budget - 60:
                break

            x = x0.copy()
            try:
                fx = safe_call(x)
            except StopIteration:
                break

            # (1+1)-ES with adaptive sigma using 1/5 rule approximation
            sigma = 1.5
            successes = 0
            window = 10
            for it in range(es_budget_per_start):
                if not check_time() or calls_made >= budget - 60:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                try:
                    fy = safe_call(y)
                except StopIteration:
                    break
                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                else:
                    successes = 0  # reset on failure for simplicity

                # Adapt sigma based on recent success rate
                if (it + 1) % window == 0:
                    recent_successes = min(successes, window)
                    if recent_successes > window * 0.6:
                        sigma = min(sigma * 1.15, 5.0)
                    elif recent_successes < window * 0.2:
                        sigma = max(sigma * 0.85, 0.01)
                    successes = 0

            candidates.append((fx, x.copy()))

        # Phase 2: Local polish with L-BFGS-B from top candidates
        # Sort candidates by fitness and polish the best 2-3
        candidates.sort(key=lambda t: t[0])
        top_n = min(3, len(candidates))
        
        for cand_idx in range(top_n):
            if not check_time() or calls_made >= budget - 15:
                break
            if time.time() - start_time > time_limit - 30:
                break

            cand_f, cand_x = candidates[cand_idx]
            
            def lbfun(x_in):
                nonlocal calls_made
                if calls_made >= self.budget:
                    raise StopIteration
                calls_made += 1
                return func(x_in)

            try:
                remaining = budget - calls_made
                maxfun = min(remaining, 80)
                res = minimize(lbfun, cand_x, method='L-BFGS-B',
                               bounds=[(-5.0, 5.0) for _ in range(dim)],
                               options={'maxiter': 60, 'maxfun': maxfun,
                                        'ftol': 1e-12, 'gtol': 1e-12})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except StopIteration:
                break
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
