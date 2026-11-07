"""Multi-start Nelder-Mead optimizer for BBOB f17 (Schaffers F7)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Stochastic multi-start Nelder-Mead with 10 independent runs."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Deterministic set of 10 starting points spread in [-4, 4]^5
        # We use a Halton-like or simply grid-based deterministic points
        starts = []
        n_starts = 10
        # Create deterministic points using a simple stratified approach
        # For 5 dimensions, we can use combinations of {-3, -1, 1, 3}
        # But let's use a more systematic approach:
        # Generate points based on indices 0..9 mapped to the box
        for i in range(n_starts):
            # Use a quasi-random deterministic sequence
            # Simple approach: use fractional parts of i * golden ratio in each dim
            point = np.zeros(dim)
            for d in range(dim):
                # Different deterministic values for each dimension
                # Use a hash-like deterministic value
                val = ((i * 7 + d * 13 + 3) % 10) / 10.0 * 8.0 - 4.0
                point[d] = val
            starts.append(point)

        # Ensure we have distinct points; add more variety if needed
        # Let's use a better deterministic set:
        starts = []
        # Use 10 points from a stratified sampling approach
        # Divide [-4, 4] into segments and pick centers
        # For simplicity, use fixed offsets
        base_points = [
            [-3.0, -3.0, -3.0, -3.0, -3.0],
            [3.0, 3.0, 3.0, 3.0, 3.0],
            [-3.0, 3.0, -3.0, 3.0, -3.0],
            [3.0, -3.0, 3.0, -3.0, 3.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [-2.0, 2.0, -2.0, 2.0, -2.0],
            [2.0, -2.0, 2.0, -2.0, 2.0],
            [-1.0, -1.0, 1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, -1.0, 1.0],
            [-4.0, 4.0, -4.0, 4.0, -4.0],
        ]
        starts = [np.array(p, dtype=float) for p in base_points]

        # Budget per run
        budget_per_run = budget // n_starts
        remaining = budget - budget_per_run * n_starts
        # Distribute remaining to first runs
        for i in range(remaining):
            starts[i] = starts[i]  # just to track, we'll adjust budget in loop

        best_x = None
        best_f = np.inf

        # Track total calls made
        calls_made = 0

        for i in range(n_starts):
            if calls_made >= budget:
                break

            x0 = starts[i].copy()
            x0 = np.clip(x0, -5, 5)

            # Build initial simplex
            simplex = [x0.copy()]
            fx_list = [func(x0)]
            calls_made += 1

            # Add simplex points with small perturbations
            step = 0.5
            for d in range(dim):
                xp = x0.copy()
                xp[d] += step
                xp = np.clip(xp, -5, 5)
                simplex.append(xp)
                fx_list.append(func(xp))
                calls_made += 1
                if calls_made >= budget:
                    break

            if calls_made >= budget:
                # Find best in simplex
                min_idx = np.argmin(fx_list)
                if fx_list[min_idx] < best_f:
                    best_f = fx_list[min_idx]
                    best_x = simplex[min_idx].copy()
                break

            # Nelder-Mead iterations
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            iters_in_run = budget_per_run - (len(simplex) - 1)
            for _ in range(max(0, iters_in_run)):
                if calls_made >= budget:
                    break

                # Sort simplex by function value
                order = np.argsort(fx_list)
                simplex = [simplex[i] for i in order]
                fx_list = [fx_list[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                calls_made += 1
                if calls_made >= budget:
                    break

                if fr < fx_list[0]:
                    # Try expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    calls_made += 1
                    if calls_made >= budget:
                        break
                    if fe < fr:
                        simplex[-1] = xe
                        fx_list[-1] = fe
                    else:
                        simplex[-1] = xr
                        fx_list[-1] = fr
                elif fr < fx_list[-1]:
                    # Accept reflection
                    simplex[-1] = xr
                    fx_list[-1] = fr
                else:
                    # Contraction
                    if fr < fx_list[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (xr - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    calls_made += 1
                    if calls_made >= budget:
                        break

                    if fc <= min(fr, fx_list[-1]):
                        simplex[-1] = xc
                        fx_list[-1] = fc
                    else:
                        # Shrink
                        for j in range(1, dim + 1):
                            simplex[j] = simplex[0] + sigma * (simplex[j] - simplex[0])
                            simplex[j] = np.clip(simplex[j], -5, 5)
                            fx_list[j] = func(simplex[j])
                            calls_made += 1
                            if calls_made >= budget:
                                break
                        if calls_made >= budget:
                            break

                # Track best
                min_idx = np.argmin(fx_list)
                if fx_list[min_idx] < best_f:
                    best_f = fx_list[min_idx]
                    best_x = simplex[min_idx].copy()

            # Update global best after each run
            min_idx = np.argmin(fx_list)
            if fx_list[min_idx] < best_f:
                best_f = fx_list[min_idx]
                best_x = simplex[min_idx].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
