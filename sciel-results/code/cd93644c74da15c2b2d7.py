"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize


class Optimizer:
    """Optimizer using Differential Evolution followed by Nelder-Mead."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        calls = {'count': 0}
        
        def counted_func(x):
            calls['count'] += 1
            return func(x)

        # Phase 1: Global Search with Differential Evolution
        # Use ~60% of budget for global search
        de_budget = int(self.budget * 0.6)
        
        # SciPy DE population size is popsize * dim
        # We aim for a population of ~10-15
        popsize_multiplier = 2 # 2 * 5 = 10 individuals
        max_iter_de = de_budget // (popsize_multiplier * self.dim)
        
        if max_iter_de < 1:
            max_iter_de = 1

        x_best = np.random.uniform(-5, 5, self.dim)
        f_best = counted_func(x_best)
        
        try:
            result_de = scipy.optimize.differential_evolution(
                counted_func, 
                bounds=[(-5, 5)] * self.dim, 
                popsize=popsize_multiplier, 
                maxiter=max_iter_de,
                seed=0,
                tol=None,
                polish=False
            )
            if result_de.fun < f_best:
                x_best = result_de.x
                f_best = result_de.fun
        except Exception:
            pass

        # Phase 2: Local Refinement with Nelder-Mead
        remaining = self.budget - calls['count']
        if remaining > 0:
            try:
                result_local = scipy.optimize.minimize(
                    counted_func, 
                    x_best, 
                    method='Nelder-Mead',
                    options={'maxiter': remaining, 'xatol': 1e-9, 'fatol': 1e-9}
                )
                if result_local.fun < f_best:
                    x_best = result_local.x
                    f_best = result_local.fun
            except Exception:
                pass

        return x_best, f_best
# EVOLVE-BLOCK-END
