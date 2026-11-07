"""Hierarchical grid search exploiting Katsuura's periodic structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hierarchical grid search with local refinement for Katsuura."""

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

        def nelder_mead(x0, max_calls):
            nonlocal calls_made, best_f, best_x
            n = dim
            # Build initial simplex
            simplex = [np.array(x0, dtype=float, copy=True)]
            for i in range(n):
                xi = np.array(x0, dtype=float, copy=True)
                step = 0.1 if abs(xi[i]) < 0.5 else 0.05 * max(0.5, abs(xi[i]))
                xi[i] += step
                simplex.append(xi)
            simplex = [np.clip(x, -5, 5) for x in simplex]

            # Evaluate initial simplex
            fvals = []
            for x in simplex:
                try:
                    fvals.append(counted_func(x))
                except RuntimeError:
                    return

            alpha, gamma, rho = 1.0, 2.0, -0.5

            while calls_made < budget and time.time() < deadline - 5:
                # Sort by fitness
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Check convergence
                if max(fvals) - min(fvals) < 1e-14:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                try:
                    fr = counted_func(xr)
                except RuntimeError:
                    return

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    try:
                        fe = counted_func(xe)
                    except RuntimeError:
                        return
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + 0.5 * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - 0.5 * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    try:
                        fc = counted_func(xc)
                    except RuntimeError:
                        return
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink: move all points toward best
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            try:
                                fvals[i] = counted_func(simplex[i])
                            except RuntimeError:
                                return

        def hierarchical_grid_search(lo, hi, grid_sizes):
            """
            Perform hierarchical grid search.
            lo, hi: arrays of shape (dim,) giving the current search box
            grid_sizes: list of integers, one per level (e.g., [3, 5, 7, 9])
            At each level, evaluate a grid of grid_sizes[i]^dim points in the current box,
            keep the best sub-box (1/grid_sizes[i] of the current box in each dimension),
            and recurse.
            """
            nonlocal calls_made, best_f, best_x

            for level, gsize in enumerate(grid_sizes):
                if calls_made >= budget or time.time() > deadline - 5:
                    return

                # Generate grid points
                n_per_dim = gsize
                total_points = n_per_dim ** dim

                # Check if we can afford this many evaluations
                if total_points > (budget - calls_made):
                    # Reduce grid size to fit
                    max_total = budget - calls_made
                    n_per_dim = max(2, int(max_total ** (1.0 / dim)))
                    total_points = n_per_dim ** dim

                # Create grid
                grid_vals = []
                for d in range(dim):
                    lo_d, hi_d = lo[d], hi[d]
                    grid_vals.append(np.linspace(lo_d, hi_d, n_per_dim))

                # Evaluate all grid points
                best_local_x = None
                best_local_f = np.inf
                best_indices = None

                # Use itertools.product to iterate over grid
                import itertools
                for idx_tuple in itertools.product(range(n_per_dim), repeat=dim):
                    if calls_made >= budget or time.time() > deadline - 5:
                        return
                    x = np.array([grid_vals[d][idx_tuple[d]] for d in range(dim)])
                    try:
                        f = counted_func(x)
                    except RuntimeError:
                        return
                    if f < best_local_f:
                        best_local_f = f
                        best_local_x = x
                        best_indices = idx_tuple

                if best_local_x is None:
                    return

                # Determine the sub-box containing the best point
                # The sub-box is centered on the best point with half the grid spacing
                new_lo = np.zeros(dim)
                new_hi = np.zeros(dim)
                for d in range(dim):
                    span = hi[d] - lo[d]
                    spacing = span / n_per_dim
                    center = best_local_x[d]
                    new_lo[d] = max(lo[d], center - spacing / 2)
                    new_hi[d] = min(hi[d], center + spacing / 2)

                lo = new_lo
                hi = new_hi

        # Phase 1: Coarse hierarchical grid search
        # Start with the full domain [-5, 5]^dim
        lo = np.full(dim, -5.0)
        hi = np.full(dim, 5.0)

        # Use progressively finer grids: 3^5=243, then refine
        # Budget: 500 calls. 3^5 = 243 for first level, leaves ~257 for refinement
        grid_sizes = [3, 3, 3]
        hierarchical_grid_search(lo, hi, grid_sizes)

        # Phase 2: Local refinement with Nelder-Mead from the best point
        if best_x is not None and calls_made < budget:
            nelder_mead(best_x, budget - calls_made)

        # Phase 3: If budget remains, do a finer local grid search around best
        if best_x is not None and calls_made < budget:
            # Local grid search in a small box around best_x
            local_lo = np.maximum(-5.0, best_x - 0.5)
            local_hi = np.minimum(5.0, best_x + 0.5)
            grid_sizes_local = [5, 5]
            hierarchical_grid_search(local_lo, local_hi, grid_sizes_local)

        # Phase 4: Final Nelder-Mead polish
        if best_x is not None and calls_made < budget:
            nelder_mead(best_x, budget - calls_made)

        return best_x, best_f
# EVOLVE-BLOCK-END
# EVOLVE-BLOCK-END
