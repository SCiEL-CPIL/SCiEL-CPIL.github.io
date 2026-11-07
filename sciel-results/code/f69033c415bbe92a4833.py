"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic adaptive grid refinement multi-objective optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280.0  # 300s limit, 20s safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        def time_check():
            return time.time() < deadline

        # Phase 1: Initial coarse grid
        # Use a small grid to get initial coverage
        grid_size = 3  # 3^4 = 81 points for dim=4
        if budget <= 100:
            grid_size = 2

        points = []
        values = []
        evals_used = 0

        def evaluate(x):
            nonlocal evals_used
            fx = problem(x)
            evals_used += 1
            return fx

        # Generate grid points
        grid_coords = np.linspace(0.0, 1.0, grid_size)
        grid_points = []
        for c1 in grid_coords:
            for c2 in grid_coords:
                for c3 in grid_coords:
                    for c4 in grid_coords:
                        x = lo + (hi - lo) * np.array([c1, c2, c3, c4])
                        grid_points.append(x)

        # Evaluate initial grid
        for x in grid_points:
            if not time_check():
                break
            if evals_used >= budget:
                break
            f = evaluate(x)
            points.append(x)
            values.append(f)

        # Phase 2: Adaptive refinement
        # Keep refining the best (non-dominated) cells
        # For simplicity, we'll do a few rounds of refinement on promising regions

        # Sort by first objective to find trade-off region
        if len(values) > 0:
            values_arr = np.array(values)
            points_arr = np.array(points)

            # Find non-dominated points
            def is_dominated(f, all_f):
                # f is dominated if there exists another point that is better in all objectives
                # and strictly better in at least one
                for other_f in all_f:
                    if np.all(other_f <= f) and np.any(other_f < f):
                        return True
                return False

            non_dom_indices = []
            for i, f in enumerate(values_arr):
                if not is_dominated(f, values_arr):
                    non_dom_indices.append(i)

            # Refine around non-dominated points
            if non_dom_indices and time_check():
                # For each non-dominated point, generate local perturbations
                step = (hi - lo) / grid_size
                for idx in non_dom_indices[:10]:  # Limit to top 10
                    if not time_check() or evals_used >= budget:
                        break
                    center = points_arr[idx]
                    # Generate 8 neighbors (corners of a hypercube around center)
                    for d1 in [-1, 0, 1]:
                        for d2 in [-1, 0, 1]:
                            for d3 in [-1, 0, 1]:
                                for d4 in [-1, 0, 1]:
                                    if d1 == 0 and d2 == 0 and d3 == 0 and d4 == 0:
                                        continue
                                    if not time_check() or evals_used >= budget:
                                        break
                                    x = center + step * np.array([d1, d2, d3, d4])
                                    x = np.clip(x, lo, hi)
                                    f = evaluate(x)
                                    points.append(x)
                                    values.append(f)

        # Phase 3: Use remaining budget for random sampling in promising regions
        if time_check() and evals_used < budget:
            # Use scipy or simple random sampling
            rng = np.random.default_rng(42)
            remaining = budget - evals_used
            for _ in range(remaining):
                if not time_check():
                    break
                if evals_used >= budget:
                    break
                # Sample near the center of the domain or near known good points
                if len(points) > 0:
                    # Mix: 50% random, 50% near best points
                    if rng.random() < 0.5:
                        x = lo + (hi - lo) * rng.random(dim)
                    else:
                        # Pick a random existing point and perturb
                        base_idx = rng.integers(len(points))
                        base = np.array(points[base_idx])
                        sigma = 0.05 * (hi - lo)
                        x = base + sigma * rng.standard_normal(dim)
                        x = np.clip(x, lo, hi)
                else:
                    x = lo + (hi - lo) * rng.random(dim)
                f = evaluate(x)
                points.append(x)
                values.append(f)

        return points
# EVOLVE-BLOCK-END
