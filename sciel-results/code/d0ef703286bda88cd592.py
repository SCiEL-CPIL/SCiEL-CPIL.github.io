"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start with Nelder-Mead local refinement for Weierstrass f16."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin

        # Phase 1: Multi-start random sampling to find good basins
        # Use ~15% of budget for exploration
        explore_budget = max(10, int(budget * 0.15))
        best_x = None
        best_f = float('inf')
        candidates = []

        for i in range(explore_budget):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, n)
            fx = func(x)
            candidates.append((fx, x.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = rng.uniform(-5, 5, n)
            best_f = func(best_x)
            candidates.append((best_f, best_x.copy()))

        # Sort candidates by function value
        candidates.sort(key=lambda t: t[0])

        # Phase 2: Local optimization from top candidates using Nelder-Mead
        remaining_budget = budget - explore_budget
        # We'll try to refine the top few candidates
        # Each Nelder-Mead call uses ~10-20 evaluations
        # Allocate budget across top candidates

        # Determine how many local optimizations we can afford
        # Each NM run typically uses ~n*2 to n*5 evaluations
        evals_per_nm = 15
        num_local = min(len(candidates), max(1, remaining_budget // evals_per_nm))

        # Ensure we at least do one local optimization
        num_local = max(1, num_local)

        # Distribute remaining budget among local optimizations
        budget_per_local = remaining_budget // num_local
        budget_per_local = max(5, budget_per_local)

        for idx in range(num_local):
            if time.time() > deadline:
                break

            fx0, x0 = candidates[idx]

            # If this candidate is worse than current best by a lot, skip
            if fx0 > best_f + 5.0 and idx > 0:
                continue

            # Run Nelder-Mead from this starting point
            x_start = x0.copy()

            def objective(x):
                return func(np.clip(x, -5, 5))

            try:
                result = minimize(
                    objective,
                    x_start,
                    method='Nelder-Mead',
                    options={
                        'maxiter': budget_per_local,
                        'xatol': 1e-6,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5, 5).copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
