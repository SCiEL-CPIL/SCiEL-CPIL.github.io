"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Grid-decomposed multi-objective optimizer with local Nelder-Mead simplex search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 270.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Grid decomposition: partition 4D space into hypercubes
        # Use a 2x2x2x2 grid = 16 cells for good coverage
        grid_size = 2
        n_cells = grid_size ** dim  # 16 for dim=4

        # Allocate budget: some for grid center exploration, rest for local optimization
        # Each cell gets a simplex search with limited evaluations
        evals_per_cell = max(5, (budget - n_cells) // n_cells)
        # Reserve some evals for global refinement
        reserve = min(40, budget // 5)
        evals_per_cell = max(5, (budget - n_cells - reserve) // n_cells)

        # Track all evaluated points and their objectives
        all_points = []
        all_objectives = []
        eval_count = 0

        def evaluate(x):
            nonlocal eval_count
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            eval_count += 1
            all_points.append(x.copy())
            all_objectives.append(f)
            return x, f

        # Phase 1: Evaluate grid cell centers
        cell_centers = []
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    for l in range(grid_size):
                        idx = [i, j, k, l]
                        center = np.array([(lo[d] + hi[d]) / 2.0 * 0 + lo[d] + (hi[d] - lo[d]) * (idx[d] + 0.5) / grid_size
                                           for d in range(dim)])
                        cell_centers.append(center)

        for center in cell_centers:
            if eval_count >= budget or time.time() > deadline:
                break
            evaluate(center)

        # Phase 2: Local Nelder-Mead simplex search in each cell
        # For each cell, create a simplex around the cell center
        # and perform a few iterations of Nelder-Mead

        for center in cell_centers:
            if eval_count >= budget or time.time() > deadline:
                break

            # Create initial simplex: center + small perturbations
            cell_size = (hi - lo) / grid_size
            simplex = [center.copy()]
            for d in range(dim):
                point = center.copy()
                point[d] += 0.4 * cell_size[d]
                point = np.clip(point, lo, hi)
                simplex.append(point)

            # Evaluate simplex points (if not already evaluated)
            simplex_objs = []
            for pt in simplex:
                if eval_count >= budget or time.time() > deadline:
                    break
                _, f = evaluate(pt)
                simplex_objs.append(f)

            if len(simplex_objs) < dim + 1:
                continue

            # Nelder-Mead iterations (limited by budget)
            n_iterations = min(15, max(3, (budget - eval_count) // (n_cells * (dim + 1))))
            for _iter in range(n_iterations):
                if eval_count >= budget or time.time() > deadline:
                    break

                # Sort simplex by sum of objectives (simple scalarization)
                order = np.argsort(np.sum(simplex_objs, axis=1))
                simplex = [simplex[i] for i in order]
                simplex_objs = [simplex_objs[i] for i in order]

                # Compute centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]

                # Reflection
                alpha = 1.0
                reflected = centroid + alpha * (centroid - worst)
                reflected = np.clip(reflected, lo, hi)
                _, f_reflected = evaluate(reflected)

                if np.sum(f_reflected) < np.sum(simplex_objs[0]):
                    # Expansion
                    gamma = 2.0
                    expanded = centroid + gamma * (centroid - worst)
                    expanded = np.clip(expanded, lo, hi)
                    _, f_expanded = evaluate(expanded)
                    if np.sum(f_expanded) < np.sum(f_reflected):
                        simplex[-1] = expanded
                        simplex_objs[-1] = f_expanded
                    else:
                        simplex[-1] = reflected
                        simplex_objs[-1] = f_reflected
                elif np.sum(f_reflected) < np.sum(simplex_objs[-1]):
                    simplex[-1] = reflected
                    simplex_objs[-1] = f_reflected
                else:
                    # Contraction
                    rho = 0.5
                    contracted = centroid + rho * (worst - centroid)
                    contracted = np.clip(contracted, lo, hi)
                    _, f_contracted = evaluate(contracted)
                    if np.sum(f_contracted) < np.sum(simplex_objs[-1]):
                        simplex[-1] = contracted
                        simplex_objs[-1] = f_contracted
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lo, hi)
                            _, f_shrunk = evaluate(simplex[i])
                            simplex_objs[i] = f_shrunk

        # Phase 3: Use remaining budget for targeted refinement
        # Find the current Pareto front and explore gaps
        if eval_count < budget and time.time() < deadline:
            # Get current Pareto front
            if all_objectives:
                F = np.array(all_objectives)
                # Find non-dominated points
                n = len(F)
                dominated = np.zeros(n, dtype=bool)
                for i in range(n):
                    for j in range(n):
                        if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            dominated[i] = True
                            break
                pareto_idx = np.where(~dominated)[0]

                # For each Pareto point, try small perturbations
                rng = np.random.default_rng(42)
                for idx in pareto_idx:
                    if eval_count >= budget or time.time() > deadline:
                        break
                    base = all_points[idx]
                    for _ in range(3):
                        if eval_count >= budget or time.time() > deadline:
                            break
                        perturbation = 0.05 * (hi - lo) * rng.standard_normal(dim)
                        candidate = np.clip(base + perturbation, lo, hi)
                        evaluate(candidate)

        # Return (value is ignored, but must not raise)
        if all_points:
            return np.array(all_points)
        return np.empty((0, dim))
# EVOLVE-BLOCK-END
