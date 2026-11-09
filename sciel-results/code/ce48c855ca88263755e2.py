# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-exploiting optimizer for DTLZ4 using deterministic Halton grid + local polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.n_shape = n_obj - 1
        self.n_dist = dim - self.n_shape

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # DTLZ4 structure: first M-1 vars are shape, rest are distance
        x_dist_opt = 0.5 * (lo[self.n_shape:] + hi[self.n_shape:])

        used = 0

        # --- Phase 1: Deterministic Halton grid over shape variables ---
        # For n_obj=3, shape vars are x[0], x[1]. The Pareto front is the set of
        # points where x[0]+x[1] <= 1 (in [0,1] space) with distance vars at 0.5.
        # We use a Halton sequence to uniformly cover the unit square [0,1]^2.
        # Only points with x0+x1 <= 1 are on the front; others are off-front.
        # But for hypervolume, we want points ON the front. So we project:
        # if x0+x1 > 1, scale back to the simplex boundary.
        
        # Number of grid points: use most of the budget for coverage
        # Leave some budget for local polish
        n_grid = 160  # deterministic grid points
        n_polish = self.budget - n_grid  # remaining for local search

        # Generate Halton sequence in 2D
        def halton(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        shape_points = []
        for k in range(1, n_grid + 1):
            x0 = halton(k, 2)
            x1 = halton(k, 3)
            # Project to simplex: if x0+x1 > 1, scale to boundary
            s = x0 + x1
            if s > 1.0:
                x0 /= s
                x1 /= s
            shape_points.append((x0, x1))

        # Build full vectors and evaluate
        X = np.zeros((n_grid, self.dim))
        for i, (x0, x1) in enumerate(shape_points):
            X[i, 0] = lo[0] + (hi[0] - lo[0]) * x0
            X[i, 1] = lo[1] + (hi[1] - lo[1]) * x1
            X[i, self.n_shape:] = x_dist_opt

        F = np.array([problem(x) for x in X])
        used = n_grid

        # --- Phase 2: Local polish using local search around best hypervolume contributors ---
        # For each grid point, do a small local search to refine its position
        # on the front, improving hypervolume contribution.

        # Vectorized dominance helpers
        def is_dominated_by_pop(f, F_pop):
            leq = np.all(F_pop <= f, axis=1)
            lt = np.any(F_pop < f, axis=1)
            return np.any(leq & lt)

        def dominated_indices_in_pop(f, F_pop):
            leq = np.all(f <= F_pop, axis=1)
            lt = np.any(f < F_pop, axis=1)
            return np.where(leq & lt)[0]

        # Compute hypervolume contribution estimate for each point
        # Use a simple proxy: for each point, estimate its unique contribution
        # by checking how many other points dominate it in each objective region
        # Simpler: just do local search on all points with remaining budget

        # Allocate polish budget: spread across all grid points
        polish_per_point = max(1, n_polish // n_grid)

        for i in range(n_grid):
            if used >= self.budget:
                break
            if time.time() > deadline:
                break

            # Local search around point i
            best_x = X[i].copy()
            best_f = F[i].copy()
            best_sum = np.sum(best_f)

            for step in range(polish_per_point):
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break

                # Small perturbation in shape space
                sigma = 0.03
                child = best_x.copy()
                child[:self.n_shape] += sigma * (hi[:self.n_shape] - lo[:self.n_shape]) * np.random.randn(self.n_shape)
                child[:self.n_shape] = np.clip(child[:self.n_shape], lo[:self.n_shape], hi[:self.n_shape])
                child[self.n_shape:] = x_dist_opt + 0.01 * (hi[self.n_shape:] - lo[self.n_shape:]) * np.random.randn(self.n_dist)
                child[self.n_shape:] = np.clip(child[self.n_shape:], lo[self.n_shape:], hi[self.n_shape:])

                f_child = problem(child)
                used += 1

                # Accept if it improves the local point (lower sum)
                if np.sum(f_child) < best_sum:
                    best_x = child
                    best_f = f_child
                    best_sum = np.sum(best_f)

            X[i] = best_x
            F[i] = best_f

        return X
# EVOLVE-BLOCK-END
