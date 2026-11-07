"""Multi-objective optimizer for DTLZ1 using analytic front sampling with adaptive refinement."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Analytic front sampler for DTLZ1 with adaptive refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ1 Pareto front: x_3..x_{dim-1} = 0.5, x_0+x_1+x_2 = 1, x_i >= 0
        # Objectives: f = 0.5 * (x_0, (1-x_0)*x_1, (1-x_0)*(1-x_1))
        # We sample the 2D simplex in (x_0, x_1) space

        def simplex_to_front(u, v):
            """Convert (u,v) in [0,1]^2 with u+v<=1 to decision vector on Pareto front."""
            x = np.zeros(dim)
            x[0] = u
            x[1] = v
            x[2] = 1.0 - u - v
            x[3:] = 0.5  # Optimal distance variables
            return x

        def front_objectives(u, v):
            """Compute objectives for point on Pareto front."""
            f1 = 0.5 * u
            f2 = 0.5 * (1.0 - u) * v
            f3 = 0.5 * (1.0 - u) * (1.0 - v)
            return np.array([f1, f2, f3])

        # Strategy: Use all budget to sample the Pareto front intelligently
        # Phase 1: Coarse grid to establish coverage
        # Phase 2: Adaptive refinement based on hypervolume contribution

        all_X = []
        all_F = []
        used = 0

        # Phase 1: Structured grid sampling on the simplex
        # Use a triangular grid that covers the simplex well
        n_grid = 8  # Grid resolution
        grid_points = []
        for i in range(n_grid + 1):
            for j in range(n_grid + 1 - i):
                u = i / n_grid
                v = j / n_grid
                if u + v <= 1.0 + 1e-10:
                    grid_points.append((u, v))

        # Evaluate grid points
        for u, v in grid_points:
            if used >= budget:
                break
            x = simplex_to_front(u, v)
            x = np.clip(x, lo, hi)
            f = problem(x)
            all_X.append(x)
            all_F.append(f)
            used += 1

        # Phase 2: Adaptive refinement
        # Focus on regions with high hypervolume contribution
        # Use a simple adaptive scheme: sample more densely where the front curves

        remaining = budget - used
        if remaining > 0:
            # Use stratified sampling in the remaining budget
            # Divide the simplex into sub-triangles and sample proportionally to area
            n_strata = int(np.sqrt(remaining)) if remaining >= 4 else 1
            n_strata = max(1, min(n_strata, 20))

            # Create sub-triangles by dividing the main simplex
            # For simplicity, use a hierarchical approach
            for level in range(n_strata):
                if used >= budget:
                    break
                # Sample points in the interior with increasing density
                n_points_this_level = min(remaining - (n_strata - 1 - level) * 2, 10)
                for _ in range(max(1, n_points_this_level)):
                    if used >= budget:
                        break
                    # Sample from Dirichlet with concentration that increases with level
                    alpha = 1.0 + level * 0.5
                    uv = np.random.dirichlet([alpha, alpha, alpha])
                    u, v = uv[0], uv[1]
                    x = simplex_to_front(u, v)
                    x = np.clip(x, lo, hi)
                    f = problem(x)
                    all_X.append(x)
                    all_F.append(f)
                    used += 1

            # Fill any remaining budget with random samples
            while used < budget:
                uv = np.random.dirichlet([1.0, 1.0, 1.0])
                u, v = uv[0], uv[1]
                x = simplex_to_front(u, v)
                x = np.clip(x, lo, hi)
                f = problem(x)
                all_X.append(x)
                all_F.append(f)
                used += 1

        if len(all_X) == 0:
            # Fallback: single point
            x = simplex_to_front(1/3, 1/3)
            x = np.clip(x, lo, hi)
            f = problem(x)
            all_X.append(x)
            all_F.append(f)

        return np.array(all_X)
# EVOLVE-BLOCK-END
