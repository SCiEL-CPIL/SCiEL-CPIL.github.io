"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import dual_annealing


class Optimizer:
    """Dual annealing based black-box optimizer for BBOB f1 (Sphere)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        # Track number of calls to respect budget
        call_count = [0]
        max_calls = self.budget

        def wrapped_func(x):
            if call_count[0] >= max_calls:
                # Return a large value to discourage further exploration
                return 1e10
            call_count[0] += 1
            # Clip to box [-5, 5]^dim
            x_clipped = np.clip(x, -5, 5)
            val = func(x_clipped)
            return val

        # Define bounds
        bounds = [(-5, 5)] * self.dim

        # Run dual annealing
        # maxiter is not strictly the number of function evaluations, but we use
        # wrapped_func to hard-limit the number of calls.
        # We set maxiter high enough that the budget limit is the primary constraint.
        # initial_temp and restart_temp control the annealing schedule.
        # n_restart controls the number of restarts.
        
        # Dual annealing is stochastic. We use a fixed seed for reproducibility 
        # if needed, but the task allows any valid run. 
        # However, to be safe and deterministic in case of multiple instances 
        # needing different behaviors, we could vary seed per instance if we knew, 
        # but we don't. Using default random state is fine.
        
        # To ensure we don't exceed time, we rely on the function call limit.
        # Dual annealing can be slow if it doesn't converge, but the call limit
        # will stop it.
        
        try:
            result = dual_annealing(
                wrapped_func,
                bounds=bounds,
                initial_temp=5230.0,
                restart_temp=5000.0,
                maxiter=10000,
                seed=42
            )
        except Exception:
            # Fallback to a simple random search if dual_annealing fails
            rng = np.random.default_rng(0)
            x = rng.uniform(-5, 5, self.dim)
            fx = wrapped_func(x)
            for _ in range(max_calls - call_count[0]):
                if call_count[0] >= max_calls:
                    break
                y = rng.uniform(-5, 5, self.dim)
                fy = wrapped_func(y)
                if fy < fx:
                    x, fx = y, fy
            return x, fx

        return result.x, result.fun
# EVOLVE-BLOCK-END
