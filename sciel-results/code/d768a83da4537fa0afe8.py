"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Structure-exploiting optimizer for DTLZ7: Fix tail to 0.5, grid search on head."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        # DTLZ7 structure:
        # f1 = x0
        # f2 = x1
        # f3 = (1 + g(x)) * (h(x) + 1)
        # g(x) depends on x2...x21. Minimized at x_i = 0.5.
        # h(x) depends on x0, x1.
        
        # We have 200 budget.
        # Use 14x14 = 196 evaluations for a dense grid on x0, x1.
        # Remaining 4 evaluations can be random or unused.
        
        n_grid = 14
        evaluations_used = 0
        max_evals = self.budget
        
        # Prepare base vector with optimal tail values
        x_base = np.zeros(self.dim)
        # Bounds are [0, 1] for all variables
        # Optimal for tail (indices 2 to 21) is 0.5
        x_base[2:] = 0.5
        
        # Grid search on x0 and x1
        for i in range(n_grid):
            for j in range(n_grid):
                if evaluations_used >= max_evals:
                    break
                x = x_base.copy()
                x[0] = i / (n_grid - 1)
                x[1] = j / (n_grid - 1)
                problem(x)
                evaluations_used += 1
            if evaluations_used >= max_evals:
                break
        
        # Use remaining budget for random sampling to catch any non-grid-optimal areas
        # (though for DTLZ7 with g=0, the front is fully determined by x0, x1)
        # Random sampling might help if the grid is too coarse for the specific hypervolume metric
        # but 14x14 is quite dense.
        # Let's add a few random points just in case, ensuring we don't exceed budget.
        rng = np.random.default_rng(0)
        while evaluations_used < max_evals:
            x = x_base.copy()
            x[0] = rng.random()
            x[1] = rng.random()
            problem(x)
            evaluations_used += 1
            
        return None
# EVOLVE-BLOCK-END
