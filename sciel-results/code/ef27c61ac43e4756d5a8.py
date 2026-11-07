"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hierarchical grid search exploiting Katsuura's periodic structure."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0

        best_x = None
        best_f = np.inf
        calls_made = 0

        def counted_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= self.budget:
                raise RuntimeError("Budget exceeded")
            if time.time() > deadline - 2:
                raise RuntimeError("Time limit")
            calls_made += 1
            val = float(func(x))
            if val < best_f:
                best_f = val
                best_x = np.array(x, copy=True)
            return val

        dim = self.dim
        budget = self.budget

        def hierarchical_grid_search(lo, hi, grid_sizes):
            """
            Perform hierarchical grid search.
            lo, hi: arrays of shape (dim,) giving the current search box
            grid_sizes: list of integers, one per level (e.g., [3, 5, 7, 9])
            At each level, evaluate a grid of grid_sizes[i]^dim points in the current box,
            find the best point, then shrink the box to the sub-cell containing it.
            """
            nonlocal calls_made, best_f, best_x

            for level, n_grid in enumerate(grid_sizes):
                if calls_made >= budget or time.time() > deadline - 5:
                    break

                # Generate grid points for this level
                # For each dimension, create n_grid evenly spaced points
                grid_coords = []
                for d in range(dim):
                    grid_coords.append(np.linspace(lo[d], hi[d], n_grid))

                # Create full grid using meshgrid
                grids = np.meshgrid(*grid_coords, indexing='ij')
                points = np.stack([g.ravel() for g in grids], axis=-1)

                # Evaluate all grid points
                level_best_f = np.inf
                level_best_idx = 0
                for idx, x in enumerate(points):
                    if calls_made >= budget or time.time() > deadline - 5:
                        break
                    try:
                        fval = counted_func(x)
                    except RuntimeError:
                        return
                    if fval < level_best_f:
                        level_best_f = fval
                        level_best_idx = idx

                if calls_made >= budget or time.time() > deadline - 5:
                    break

                # Shrink the box to the sub-cell containing the best point
                # The best point's index in the flattened grid tells us its coordinates
                # in the multi-dimensional grid
                best_coords = np.unravel_index(level_best_idx, [n_grid] * dim)

                # Compute the sub-cell bounds
                for d in range(dim):
                    idx_d = best_coords[d]
                    cell_width = (hi[d] - lo[d]) / n_grid
                    lo[d] = lo[d] + idx_d * cell_width
                    hi[d] = lo[d] + cell_width

                # If we're at the last level, we could do a final refinement
                # but the grid itself gives us the best point

        # Start with the full domain
        lo = np.array([-5.0] * dim)
        hi = np.array([5.0] * dim)

        # Use a hierarchical grid: start coarse, get finer
        # Level 0: 3^5 = 243 points
        # Level 1: 5^5 = 3125 points (but we only have 500 budget total)
        # So we need to be careful with budget allocation

        # Given 500 total budget, let's do:
        # Level 0: 3^5 = 243 points (uses ~243 calls)
        # Level 1: 5^5 would be 3125, too many. Instead, do a smaller refinement.
        # Let's do 3 levels with smaller grids:
        # Level 0: 3^5 = 243
        # Level 1: 4^5 = 1024, still too many
        # 
        # Better approach: use adaptive grid sizes based on remaining budget
        # Level 0: 3^5 = 243 points
        # After that, we have ~257 calls left
        # Level 1: 3^5 = 243 points in the shrunk box (uses ~243, total ~486)
        # Level 2: 2^5 = 32 points in the further shrunk box (uses ~32, total ~518, over budget)
        #
        # So let's do:
        # Level 0: 3^5 = 243
        # Level 1: 3^5 = 243
        # That's 486 calls, leaving 14 for a final Nelder-Mead polish or just stop.

        # Actually, let's be more careful. Let's do:
        # Level 0: 3^5 = 243 points over [-5,5]^5
        # Level 1: 3^5 = 243 points in the best sub-cell
        # Total: 486 points, 14 remaining for a quick local search

        hierarchical_grid_search(lo, hi, [3, 3])

        # Use remaining budget for a quick Nelder-Mead polish
        if best_x is not None and calls_made < budget and time.time() < deadline - 5:
            n = dim
            simplex = [np.array(best_x, dtype=float, copy=True)]
            # Estimate a reasonable step size based on current box
            step_size = 0.1
            for i in range(n):
                xi = np.array(best_x, dtype=float, copy=True)
                xi[i] += step_size
                simplex.append(xi)
            simplex = [np.clip(x, -5, 5) for x in simplex]

            fvals = []
            for x in simplex:
                try:
                    fvals.append(counted_func(x))
                except RuntimeError:
                    break

            if len(fvals) == n + 1:
                alpha, gamma, rho = 1.0, 2.0, -0.5
                while calls_made < budget and time.time() < deadline - 2:
                    order = np.argsort(fvals)
                    simplex = [simplex[i] for i in order]
                    fvals = [fvals[i] for i in order]

                    if max(fvals) - min(fvals) < 1e-10:
                        break

                    centroid = np.mean(simplex[:-1], axis=0)
                    xr = centroid + alpha * (centroid - simplex[-1])
                    xr = np.clip(xr, -5, 5)
                    try:
                        fr = counted_func(xr)
                    except RuntimeError:
                        break

                    if fr < fvals[0]:
                        xe = centroid + gamma * (centroid - simplex[-1])
                        xe = np.clip(xe, -5, 5)
                        try:
                            fe = counted_func(xe)
                        except RuntimeError:
                            break
                        if fe < fr:
                            simplex[-1] = xe
                            fvals[-1] = fe
                        else:
                            simplex[-1] = xr
                            fvals[-1] = fr
                    elif fr < fvals[-2]:
                        simplex[-1] = xr
                        fvals[-1] = fr
                    else:
                        if fr < fvals[-1]:
                            xc = centroid + 0.5 * (xr - centroid)
                        else:
                            xc = centroid - 0.5 * (simplex[-1] - centroid)
                        xc = np.clip(xc, -5, 5)
                        try:
                            fc = counted_func(xc)
                        except RuntimeError:
                            break
                        if fc < fvals[-1]:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                try:
                                    fvals[i] = counted_func(simplex[i])
                                except RuntimeError:
                                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
# EVOLVE-BLOCK-END
