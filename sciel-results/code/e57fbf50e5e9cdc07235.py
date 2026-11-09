"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import dual_annealing


class Optimizer:
    """Dual Annealing optimizer on the box [-5, 5]^dim."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        # Dual annealing is a derivative-free global optimization method.
        # It uses a combination of simulated annealing and basin hopping.
        # We set maxiter and maxfev to stay within the budget.
        # The initial program used 500 calls. We aim to use at most 500.
        
        # dual_annealing accepts bounds as a list of (min, max) tuples
        bounds = [(-5.0, 5.0)] * self.dim
        
        # maxfev is the maximum number of function evaluations.
        # We set it to self.budget to ensure we don't exceed the limit.
        # Note: dual_annealing might use slightly more or fewer depending on implementation details,
        # but it is generally strict with maxfev.
        # To be safe, we can set it to self.budget - 5 to account for any internal overhead if any,
        # but typically maxfev is the hard cap.
        
        try:
            res = dual_annealing(
                func,
                bounds,
                maxfev=self.budget,
                seed=42  # Fixed seed for reproducibility if needed, though rng is internal
            )
            # res.fun is the best function value found
            # res.x is the best x found
            return res.x, res.fun
        except Exception:
            # If dual_annealing fails for some reason, fall back to a simple random search
            # to ensure we don't crash and produce an invalid result.
            rng = np.random.default_rng(0)
            best_x = rng.uniform(-5, 5, self.dim)
            best_f = func(best_x)
            for _ in range(self.budget - 1):
                x = rng.uniform(-5, 5, self.dim)
                f = func(x)
                if f < best_f:
                    best_x, best_f = x, f
            return best_x, best_f
# EVOLVE-BLOCK-END
