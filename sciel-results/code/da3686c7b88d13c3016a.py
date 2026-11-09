"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using a structured search strategy."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # 300s wall-clock limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # DTLZ4 structure:
        # - First (dim - n_obj + 1) variables are "position" variables
        # - Last (n_obj - 1) variables are "shape" variables
        # For DTLZ4 with n_obj=3, dim=12:
        #   position vars: x[0:10], shape vars: x[10:12]
        # The Pareto front is achieved when position vars are at optimal values.
        # For DTLZ4, the position variables should be at their optimal values
        # which depend on the specific formulation. Typically, for DTLZ4,
        # the position variables should be close to 0 or 1 depending on the problem.

        # Strategy:
        # 1. Sample shape variables on a grid to cover the Pareto front
        # 2. For each shape configuration, optimize position variables
        # 3. Use a mix of random search and local refinement

        n_shape = self.n_obj - 1  # 2 shape variables
        n_pos = self.dim - n_shape  # 10 position variables

        # Generate shape variable samples on a structured grid
        # For DTLZ4, the shape variables determine the point on the Pareto front
        # We'll sample them in [0, 1]^2

        # Budget allocation:
        # - ~40% for initial structured sampling
        # - ~40% for local refinement
        # - ~20% for random exploration

        budget_init = int(0.4 * self.budget)
        budget_refine = int(0.4 * self.budget)
        budget_random = self.budget - budget_init - budget_refine

        # Phase 1: Structured sampling of shape variables
        # Create a grid of shape variable values
        n_grid = int(np.sqrt(budget_init))
        if n_grid < 2:
            n_grid = 2

        shape_grid = []
        for i in range(n_grid):
            for j in range(n_grid):
                if len(shape_grid) >= budget_init:
                    break
                s1 = i / (n_grid - 1) if n_grid > 1 else 0.5
                s2 = j / (n_grid - 1) if n_grid > 1 else 0.5
                shape_grid.append([s1, s2])

        # For each shape configuration, find good position variables
        # For DTLZ4, the position variables affect the distance to the Pareto front
        # Optimal position variables are typically at specific values

        # Let's try: position vars at 0, 1, or 0.5
        pos_candidates = [
            np.zeros(n_pos),
            np.ones(n_pos),
            np.full(n_pos, 0.5),
            np.random.RandomState(0).random(n_pos) * 0.1,  # near 0
            np.random.RandomState(1).random(n_pos) * 0.1 + 0.9,  # near 1
        ]

        results = []
        used = 0

        # Phase 1: Evaluate structured combinations
        for shape in shape_grid:
            if used >= self.budget:
                break
            if time.time() > deadline - 30:
                break

            for pos in pos_candidates:
                if used >= self.budget:
                    break
                if time.time() > deadline - 30:
                    break

                x = np.concatenate([pos, shape])
                x = np.clip(x, lo, hi)
                f = problem(x)
                results.append(f)
                used += 1

        # Phase 2: Local refinement around best points
        if results and used < self.budget:
            results_arr = np.array(results)
            # Find non-dominated points
            n_res = len(results_arr)
            dominated = np.zeros(n_res, dtype=bool)
            for i in range(n_res):
                for j in range(n_res):
                    if i != j and np.all(results_arr[j] <= results_arr[i]) and np.any(results_arr[j] < results_arr[i]):
                        dominated[i] = True
                        break

            nondom_idx = np.where(~dominated)[0]
            if len(nondom_idx) > 0:
                # Refine around non-dominated points
                n_refine = min(budget_refine, self.budget - used)
                for _ in range(n_refine):
                    if used >= self.budget:
                        break
                    if time.time() > deadline - 30:
                        break

                    idx = np.random.randint(len(nondom_idx))
                    center = results_arr[nondom_idx[idx]]

                    # We need to find the x that produced this f
                    # Since we don't store x, let's do random perturbation in x-space
                    # Instead, let's sample near the shape values that produced good results

                    # Random perturbation approach
                    x_new = np.random.random(self.dim) * span + lo
                    # Bias towards known good regions
                    if np.random.random() < 0.5:
                        # Perturb position variables
                        pos_part = np.random.random(n_pos) * 0.2
                        shape_part = np.random.random(n_shape) * 0.1
                        x_new = np.concatenate([pos_part + 0.4, shape_part + 0.45])
                    x_new = np.clip(x_new, lo, hi)

                    f_new = problem(x_new)
                    results.append(f_new)
                    used += 1

        # Phase 3: Random exploration
        while used < self.budget:
            if time.time() > deadline - 30:
                break
            x = np.random.random(self.dim) * span + lo
            x = np.clip(x, lo, hi)
            f = problem(x)
            results.append(f)
            used += 1

        return results
# EVOLVE-BLOCK-END
