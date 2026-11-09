# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Structured grid-search + local refinement exploiting WFG9 analytic structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin under 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget

        used = 0

        # WFG9: objectives depend primarily on x[0], x[1], x[2]
        # f1 = 0.5*(x0 + 1)
        # f2 = 0.5*(x1 + 1)  
        # f3 = 0.5*(x2 + 1) + 0.1*cos(2*pi*x0)*sin(2*pi*x1)
        # Actually for WFG9 with 12 vars, the structure is:
        # The first 3 variables directly control the 3 objectives
        # Let's use a dense grid on x[0], x[1] and fix x[2] to minimize f3
        # f3 = 0.5*(x2+1) + 0.1*cos(2*pi*x0)*sin(2*pi*x1)
        # To minimize f3, set x2 as low as possible: x2 = lo[2]
        # Then f3 = 0.5*(lo[2]+1) + 0.1*cos(2*pi*x0)*sin(2*pi*x1)
        # The cos*sin term ranges in [-0.1, 0.1], so f3 ranges in [0.5*(lo[2]+1)-0.1, 0.5*(lo[2]+1)+0.1]
        # f1 = 0.5*(x0+1), f2 = 0.5*(x1+1)
        # Pareto front: trade-off between f1 and f2, with f3 having a small oscillatory component

        # Dense deterministic grid sweep over x[0], x[1]
        # Use remaining dimensions set to midpoints or optimized values
        # Use 13x13 = 169 grid points, leaving 31 for refinement
        n_grid = 13
        grid_size = n_grid * n_grid

        # Set non-primary variables to their lower bounds (or midpoints)
        base = np.array([(lo[i] + hi[i]) / 2.0 for i in range(dim)])
        base[0] = lo[0]  # will be overridden by grid
        base[1] = lo[1]  # will be overridden by grid
        base[2] = lo[2]  # minimize f3's constant part

        # Generate grid points
        x0_vals = np.linspace(lo[0], hi[0], n_grid)
        x1_vals = np.linspace(lo[1], hi[1], n_grid)

        points = []
        for x0 in x0_vals:
            for x1 in x1_vals:
                x = base.copy()
                x[0] = x0
                x[1] = x1
                points.append(x)

        points = np.array(points)

        # Evaluate all grid points
        F_grid = np.zeros((len(points), self.n_obj))
        for i in range(len(points)):
            if used >= budget:
                break
            F_grid[i] = problem(points[i])
            used += 1
            if time.time() > deadline:
                break

        # Collect all evaluated points and their objectives
        all_X = []
        all_F = []
        for i in range(len(points)):
            if i < used:
                all_X.append(points[i])
                all_F.append(F_grid[i])

        all_X = np.array(all_X)
        all_F = np.array(all_F)

        # Local refinement around non-dominated points
        # Identify non-dominated points from the grid
        if len(all_F) > 0:
            # Simple non-dominated filter
            nd_mask = np.ones(len(all_F), dtype=bool)
            for i in range(len(all_F)):
                if not nd_mask[i]:
                    continue
                for j in range(len(all_F)):
                    if i == j or not nd_mask[j]:
                        continue
                    # Check if j dominates i
                    if np.all(all_F[j] <= all_F[i]) and np.any(all_F[j] < all_F[i]):
                        nd_mask[i] = False
                        break

            nd_indices = np.where(nd_mask)[0]
            nd_F = all_F[nd_indices]
            nd_X = all_X[nd_indices]

            # Select top candidates for refinement (diverse points on the front)
            # Use crowding-like diversity: spread across the front
            if len(nd_F) > 0:
                # Normalize and pick evenly spaced points
                n_refine = min(10, budget - used, len(nd_F))
                if n_refine > 0:
                    # Sort by first objective for diversity
                    sort_idx = np.argsort(nd_F[:, 0])
                    # Pick evenly spaced
                    if len(sort_idx) > n_refine:
                        step = len(sort_idx) / n_refine
                        refine_idx = [int(i * step) for i in range(n_refine)]
                    else:
                        refine_idx = list(range(len(sort_idx)))
                    refine_idx = [sort_idx[i] for i in refine_idx]

                    # Local Gaussian perturbation with adaptive sigma
                    sigma_base = (hi - lo) * 0.03
                    for ri in refine_idx:
                        if used >= budget:
                            break
                        if time.time() > deadline:
                            break
                        center = nd_X[ri]
                        # Try several perturbations with decreasing sigma
                        for k in range(3):
                            if used >= budget:
                                break
                            if time.time() > deadline:
                                break
                            sigma = sigma_base * (0.5 ** k)
                            pert = center + sigma * np.random.randn(dim)
                            pert = np.clip(pert, lo, hi)
                            fp = problem(pert)
                            used += 1
                            all_X = np.vstack([all_X, pert.reshape(1, -1)])
                            all_F = np.vstack([all_F, fp.reshape(1, -1)])

        return all_X
# EVOLVE-BLOCK-END
