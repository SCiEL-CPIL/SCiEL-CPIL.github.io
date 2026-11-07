"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        dim = self.dim
        budget = self.budget
        
        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                return 1e10
            if calls >= budget:
                return 1e10
            calls += 1
            x_clipped = np.clip(x, -5.0, 5.0)
            f_val = float(func(x_clipped))
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        # Initialize at the center of the box
        x0 = np.zeros(dim)
        
        # Construct initial simplex
        # A larger initial step helps cover the domain quickly, which is crucial for AOCC (early progress)
        # The minimum is in [-4, 4]^5, so a step of 3.0 from 0 covers [-3, 3] initially,
        # and NM will expand/contract as needed.
        delta = 3.0
        simplex = [x0.copy()]
        for i in range(dim):
            xi = x0.copy()
            xi[i] += delta
            simplex.append(xi)

        # Configure Nelder-Mead
        # maxiter is set high to allow the budget to be the limiting factor
        # xatol and fatol are set to small values to ensure we squeeze out precision
        # if we have remaining budget, but usually the budget runs out first.
        options = {
            'initial_simplex': simplex,
            'maxiter': 500, 
            'xatol': 1e-8, 
            'fatol': 1e-8,
            'adaptive': True
        }

        try:
            minimize(wrapped_func, x0, method='Nelder-Mead', options=options)
        except Exception:
            pass

        return best_x, best_f
# EVOLVE-BLOCK-END
