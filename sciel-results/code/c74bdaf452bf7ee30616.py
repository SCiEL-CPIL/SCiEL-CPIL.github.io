"""Optimizer for BBOB f14 (Different Powers) using Multi-start Nelder-Mead with Sobol initialization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc


class Optimizer:
    """
    Multi-start Nelder-Mead optimizer with Sobol sequence initialization.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget

        # 1. Initialization using Sobol sequence for better space-filling
        # We use 20 initial points to increase the chance of finding the global basin
        n_init = 20
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        points = sampler.random(n_init)
        
        # Map [0, 1] to [-5, 5]
        points = points * 10.0 - 5.0

        best_x = None
        best_f = np.inf
        calls_made = 0

        for p in points:
            if time.time() > deadline:
                break
            try:
                fp = func(p)
                calls_made += 1
                if fp < best_f:
                    best_f = fp
                    best_x = p.copy()
            except Exception:
                # If func fails for some reason, skip this point
                pass

        if best_x is None:
            # Fallback if all initial points failed
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made += 1

        if calls_made >= budget:
            return best_x, best_f

        remaining_budget = budget - calls_made

        # 2. Local Optimization using Nelder-Mead
        # We wrap the function to strictly enforce the remaining budget
        class BudgetExceeded(Exception):
            pass

        eval_counter = {'count': 0}

        def wrapped_func(x):
            if time.time() > deadline:
                raise BudgetExceeded()
            eval_counter['count'] += 1
            if eval_counter['count'] > remaining_budget:
                raise BudgetExceeded()
            return func(x)

        try:
            # Nelder-Mead is effective for low-dimensional smooth problems
            # adaptive=True helps with step size adjustment
            res = minimize(
                wrapped_func,
                best_x,
                method='Nelder-Mead',
                options={
                    'maxiter': 100,
                    'xatol': 1e-8,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            # Update best if the optimizer found a better solution
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x

        except BudgetExceeded:
            pass
        except Exception:
            # Catch any other errors from the optimizer
            pass

        return best_x, best_f
# EVOLVE-BLOCK-END
