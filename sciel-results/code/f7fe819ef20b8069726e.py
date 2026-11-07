"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Grid-partitioned local search for multi-objective optimization."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 270.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        evals_used = 0
        all_points = []
        all_objs = []

        def eval_point(x):
            nonlocal evals_used
            x = np.clip(x, lo, hi)
            obj = np.array(problem(x), dtype=float)
            evals_used += 1
            all_points.append(x.copy())
            all_objs.append(obj.copy())
            return obj

        # Phase 1: Structured grid exploration
        # Use a 3x3x3x3 grid = 81 points for initial coverage
        grid_per_dim = 3
        total_grid = grid_per_dim ** dim  # 81 for dim=4
        # Allocate budget: grid + local refinement
        grid_budget = min(total_grid, budget // 3)  # ~66 for budget=200
        # Adjust grid size to fit budget
        while grid_per_dim ** dim > grid_budget and grid_per_dim > 1:
            grid_per_dim -= 1
        total_grid = grid_per_dim ** dim

        # Generate grid points
        grid_axes = [np.linspace(lo[i], hi[i], grid_per_dim) for i in range(dim)]
        grid_points = []
        for i1 in range(grid_per_dim):
            for i2 in range(grid_per_dim):
                for i3 in range(grid_per_dim):
                    for i4 in range(grid_per_dim):
                        pt = np.array([grid_axes[0][i1], grid_axes[1][i2],
                                       grid_axes[2][i3], grid_axes[3][i4]])
                        grid_points.append(pt)

        # Evaluate grid points
        for pt in grid_points:
            if evals_used >= budget or time.time() > deadline:
                break
            eval_point(pt)

        # Phase 2: Local refinement of promising cells
        # Sort grid points by a scalarization (weighted sum with normalized objectives)
        if len(all_objs) > 0:
            objs_arr = np.array(all_objs)
            # Normalize objectives to [0,1] range based on observed min/max
            obj_min = objs_arr.min(axis=0)
            obj_max = objs_arr.max(axis=0)
            obj_range = obj_max - obj_min
            obj_range[obj_range == 0] = 1.0
            obj_norm = (objs_arr - obj_min) / obj_range

            # Use multiple weight vectors to cover the front
            # For 2 objectives, use weights along the Pareto front
            n_weights = 5
            weights = np.linspace(0.0, 1.0, n_weights).reshape(-1, 1)
            # Also add intermediate weights
            weights_extra = np.array([[0.25, 0.25], [0.5, 0.5], [0.75, 0.75]])
            weights = np.vstack([weights, weights_extra])

            # For each weight, find the best grid point and refine locally
            remaining_budget = budget - evals_used
            if remaining_budget > 10 and time.time() < deadline:
                # Allocate refinement budget evenly among weight directions
                n_refine = min(n_weights + 3, remaining_budget // 8)  # ~8 evals per local search
                per_refine = max(5, remaining_budget // max(n_refine, 1))

                refined_count = 0
                for w in weights:
                    if evals_used >= budget - 5 or time.time() > deadline:
                        break
                    if refined_count >= n_refine:
                        break

                    # Find best grid point for this weight
                    scalar = obj_norm @ w
                    best_idx = np.argmin(scalar)
                    start_pt = all_points[best_idx].copy()

                    # Run Nelder-Mead on scalarized objective
                    def scalar_obj(x):
                        x = np.clip(x, lo, hi)
                        obj = np.array(problem(x), dtype=float)
                        return obj @ w

                    # Track evaluations for local search
                    local_evals = [0]

                    def tracked_scalar_obj(x):
                        local_evals[0] += 1
                        return scalar_obj(x)

                    try:
                        result = minimize(
                            tracked_scalar_obj,
                            start_pt,
                            method='Nelder-Mead',
                            bounds=list(zip(lo, hi)),
                            options={'maxiter': per_refine, 'xatol': 1e-6, 'fatol': 1e-6}
                        )
                        # Record all evaluations from the local search
                        # We need to track them properly
                        if result.x is not None:
                            x_final = np.clip(result.x, lo, hi)
                            obj_final = np.array(problem(x_final), dtype=float)
                            evals_used += 1
                            all_points.append(x_final.copy())
                            all_objs.append(obj_final.copy())
                    except Exception:
                        pass

                    refined_count += 1

        # Phase 3: If budget remains, do additional random sampling + local refinement
        if evals_used < budget - 5 and time.time() < deadline:
            rng = np.random.default_rng(42)
            n_extra = min(budget - evals_used - 10, 30)
            for _ in range(n_extra):
                if evals_used >= budget or time.time() > deadline:
                    break
                x_rand = lo + (hi - lo) * rng.random(dim)
                obj_rand = np.array(problem(x_rand), dtype=float)
                evals_used += 1
                all_points.append(x_rand.copy())
                all_objs.append(obj_rand.copy())

                # Quick local refinement
                if evals_used < budget - 5 and time.time() < deadline:
                    # Use the point with best scalarization among recent points
                    if len(all_objs) >= 2:
                        objs_arr = np.array(all_objs)
                        obj_min = objs_arr.min(axis=0)
                        obj_max = objs_arr.max(axis=0)
                        obj_range = obj_max - obj_min
                        obj_range[obj_range == 0] = 1.0
                        obj_norm = (objs_arr - obj_min) / obj_range
                        # Use equal weights
                        scalar = obj_norm @ np.ones(n_obj) / n_obj
                        best_idx = np.argmin(scalar)
                        start_pt = all_points[best_idx].copy()

                        def scalar_obj2(x):
                            x = np.clip(x, lo, hi)
                            obj = np.array(problem(x), dtype=float)
                            return obj @ (np.ones(n_obj) / n_obj)

                        try:
                            result = minimize(
                                scalar_obj2,
                                start_pt,
                                method='Nelder-Mead',
                                bounds=list(zip(lo, hi)),
                                options={'maxiter': 10, 'xatol': 1e-6, 'fatol': 1e-6}
                            )
                            if result.x is not None:
                                x_final = np.clip(result.x, lo, hi)
                                obj_final = np.array(problem(x_final), dtype=float)
                                evals_used += 1
                                all_points.append(x_final.copy())
                                all_objs.append(obj_final.copy())
                        except Exception:
                            pass

        return all_points
# EVOLVE-BLOCK-END
