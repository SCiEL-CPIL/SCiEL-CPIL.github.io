"""
Grid-based coarse-to-fine optimizer for BBOB f3 (Rastrigin) on [-5, 5]^5.
Systematic hypercube partitioning to identify lowest-value region,
then local refinement around the best cell center.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin

        n = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        calls_made = 0
        rng = np.random.default_rng(42)

        def check_deadline():
            return (time.time() - start_time) > deadline or calls_made >= budget

        def update_best(x, f):
            nonlocal best_x, best_f, calls_made
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def grid_scan(center, half_width, cells_per_dim, max_calls):
            """
            Perform a grid scan around center with given half_width.
            Evaluates centers of hypercubes partitioning the box.
            Returns the best point found.
            """
            nonlocal best_x, best_f, calls_made

            # Generate grid points
            points = []
            for i in range(cells_per_dim):
                frac = (i + 0.5) / cells_per_dim
                offset = -half_width + 2 * half_width * frac
                points.append(offset)

            # Create all combinations
            from itertools import product
            grid_points = list(product(points, repeat=n))

            for offsets in grid_points:
                if check_deadline() or calls_made >= max_calls:
                    break
                x = np.array([center[d] + offsets[d] for d in range(n)])
                x = np.clip(x, -5, 5)
                f = func(x)
                update_best(x, f)

            return best_x, best_f

        # Phase 1: Coarse grid scan of the entire box [-5, 5]^5
        # Use a modest grid to stay within budget
        coarse_cells = 3  # 3^5 = 243 evaluations
        coarse_half = 5.0
        grid_scan(np.zeros(n), coarse_half, coarse_cells, budget)

        if best_x is None:
            x = np.zeros(n)
            f = func(x)
            update_best(x, f)
            return best_x, best_f

        # Phase 2: Medium-resolution refinement around best region
        remaining = budget - calls_made
        if remaining > 50 and not check_deadline():
            # Focus on a region around the best point found
            med_half = 2.0
            med_cells = 4  # 4^5 = 1024, but we'll limit by budget
            # Estimate how many cells we can afford
            max_cells = min(4**n, remaining)
            # Use smaller cells if we can afford more
            if max_cells >= 4**n:
                med_cells = 4
            elif max_cells >= 3**n:
                med_cells = 3
            else:
                med_cells = 2

            grid_scan(best_x, med_half, med_cells, remaining)

        # Phase 3: Fine local refinement around the best point
        remaining = budget - calls_made
        if remaining > 20 and not check_deadline():
            fine_half = 0.5
            fine_cells = 3  # 3^5 = 243
            # Limit by remaining budget
            if 3**n > remaining:
                fine_cells = 2

            grid_scan(best_x, fine_half, fine_cells, remaining)

        # Phase 4: Very fine local search if budget remains
        remaining = budget - calls_made
        if remaining > 10 and not check_deadline():
            vfine_half = 0.1
            vfine_cells = 3
            if 3**n > remaining:
                vfine_cells = 2

            grid_scan(best_x, vfine_half, vfine_cells, remaining)

        # Phase 5: Final stochastic perturbation if budget remains
        remaining = budget - calls_made
        if remaining > 5 and not check_deadline():
            for _ in range(min(remaining, 20)):
                if check_deadline():
                    break
                scale = 0.05
                x = np.clip(best_x + scale * rng.standard_normal(n), -5, 5)
                f = func(x)
                update_best(x, f)

        return best_x, best_f

# EVOLVE-BLOCK-END
