"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize


class Optimizer:
    """Hybrid Differential Evolution + Nelder-Mead for Rastrigin rotated."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = np.inf
        rng = np.random.default_rng(42)

        def wrapper(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() >= deadline:
                return np.inf
            val = func(x)
            calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: Differential Evolution for global exploration
        de_budget = int(budget * 0.6)
        popsize = 5
        pop_size = popsize * dim
        maxiter = max(1, de_budget // pop_size)
        
        try:
            res_de = scipy.optimize.differential_evolution(
                wrapper,
                bounds=[(-5, 5)] * dim,
                popsize=popsize,
                maxiter=maxiter,
                tol=1e-4,
                seed=42,
                init='latinhypercube',
                polish=False
            )
        except Exception:
            pass

        # Phase 2: Nelder-Mead for local exploitation
        if best_x is not None:
            remaining_budget = budget - calls
            if remaining_budget > 10:
                nm_maxiter = max(1, remaining_budget // (dim + 1))
                try:
                    res_nm = scipy.optimize.minimize(
                        wrapper,
                        best_x,
                        method='Nelder-Mead',
                        options={'maxiter': nm_maxiter, 'xatol': 1e-4, 'fatol': 1e-4}
                    )
                except Exception:
                    pass

        # Fallback
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = wrapper(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
