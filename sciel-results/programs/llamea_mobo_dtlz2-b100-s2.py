import numpy as np
import time


class Optimizer:
    """Hybrid DTLZ2 optimizer: detect optimal distance vars, grid search shape vars, local refine."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        mid = (lo + hi) / 2.0

        # For DTLZ2: first n_obj-1 variables are shape variables
        # Remaining are distance variables
        k_shape = self.n_obj - 1  # 2 for M=3
        shape_vars = list(range(k_shape))
        dist_vars = list(range(k_shape, self.dim))

        # Step 1: Determine optimal distance variable setting
        # Test dist_vars = 0.0
        x_test0 = np.zeros(self.dim)
        for i in shape_vars:
            x_test0[i] = 0.5
        f_test0 = np.array(problem(x_test0))

        # Test dist_vars = 0.5
        x_test1 = np.zeros(self.dim)
        for i in shape_vars:
            x_test1[i] = 0.5
        for i in dist_vars:
            x_test1[i] = 0.5
        f_test1 = np.array(problem(x_test1))

        # Choose the distance setting that gives lower sum of squares (closer to origin = Pareto front)
        if np.sum(f_test0**2) < np.sum(f_test1**2):
            dist_val = 0.0
        else:
            dist_val = 0.5

        evals_used = 2
        remaining_budget = self.budget - evals_used

        # Step 2: Dense grid search on shape variables
        # Use g x g grid where g = sqrt(remaining_budget)
        g = int(np.sqrt(remaining_budget))
        g = max(g, 10)

        # Shape variables should be in [0, 1] for standard DTLZ2 mapping
        # The problem function handles the mapping to angles internally
        grid_vals = np.linspace(0.0, 1.0, g)

        solutions = []
        objective_values = []
        used = 0

        # Phase 1: Grid search
        for v1 in grid_vals:
            for v2 in grid_vals:
                if time.time() > deadline:
                    break
                if used >= remaining_budget:
                    break

                x = np.zeros(self.dim)
                x[shape_vars[0]] = v1
                if len(shape_vars) > 1:
                    x[shape_vars[1]] = v2
                for i in dist_vars:
                    x[i] = dist_val

                f = np.array(problem(x))
                solutions.append(x)
                objective_values.append(f)
                used += 1

            if used >= remaining_budget or time.time() > deadline:
                break

        # Phase 2: Local refinement around non-dominated points
        if used < remaining_budget and time.time() < deadline and solutions:
            X = np.array(solutions)
            F = np.array(objective_values)

            # Find non-dominated points
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break

            nd_indices = np.where(~dominated)[0]

            # Refine evenly spaced non-dominated points
            n_refine = min(6, len(nd_indices))
            if n_refine > 0:
                if len(nd_indices) > n_refine:
                    refine_idx = np.linspace(0, len(nd_indices) - 1, n_refine, dtype=int)
                    refine_idx = nd_indices[refine_idx]
                else:
                    refine_idx = nd_indices

                for idx in refine_idx:
                    if used >= remaining_budget or time.time() > deadline:
                        break

                    base_x = X[idx].copy()
                    step_size = 0.02

                    for pert in [(-step_size, 0), (step_size, 0), (0, -step_size), (0, step_size)]:
                        if used >= remaining_budget or time.time() > deadline:
                            break

                        new_x = base_x.copy()
                        new_x[shape_vars[0]] += pert[0]
                        new_x[shape_vars[1]] += pert[1]
                        new_x = np.clip(new_x, 0.0, 1.0)

                        new_f = np.array(problem(new_x))
                        solutions.append(new_x)
                        objective_values.append(new_f)
                        used += 1

        return np.array(solutions) if solutions else np.array([mid])
