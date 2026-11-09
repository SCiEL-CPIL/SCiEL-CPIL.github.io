"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution

class Optimizer:
    """Differential Evolution (DE) using scipy for BBOB f15 (Rastrigin rotated)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget

        # Track function evaluations
        eval_count = [0]

        def wrapped_func(x):
            eval_count[0] += 1
            if eval_count[0] > budget:
                return np.inf
            # Clip to bounds as per task description
            x_clipped = np.clip(x, -5, 5)
            return func(x_clipped)

        # Bounds for differential_evolution
        bounds = [(-5, 5)] * dim

        # Run differential evolution
        # Use a relatively high popsize for robustness, but limit maxiter to fit budget
        # popsize=15, maxiter=30 -> 15*30 = 450 evals roughly
        # We let it run until it finishes or hits the time/budget limit via the wrapper
        
        try:
            result = differential_evolution(
                wrapped_func,
                bounds,
                maxiter=100,  # Will likely stop early due to budget or convergence
                popsize=15,
                tol=1e-6,
                mutation=(0.4, 1.0),
                recombination=0.7,
                seed=42,
                init='latinhypercube',
                polish=False,  # Avoid extra calls from L-BFGS-B which might not help with Rastrigin
                update_interval=1
            )
        except Exception:
            pass

        return result.x, result.fun
# EVOLVE-BLOCK-END
