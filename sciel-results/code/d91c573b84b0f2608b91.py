"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start local search with Sobol global exploration and Nelder-Mead refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # Sobol quasi-random sequence for global exploration
        # We'll use a simple deterministic Sobol-like sequence
        # For 5D, we can use a precomputed set of well-spread points

        # Generate Sobol-like points using a simple method
        def sobol_points(n, dim, seed=42):
            """Generate n quasi-random points in [0,1]^dim using a simple method."""
            rng = np.random.default_rng(seed)
            # Use a combination of deterministic and random points
            # First, use a grid-based approach with jitter
            points = []
            # Use a stratified sampling approach
            per_dim = max(1, int(round(n ** (1.0 / dim))))
            grid = np.linspace(0, 1, per_dim + 1)[:-1] + 0.5 / per_dim
            # Create grid points
            grid_points = np.array(list(np.meshgrid(*([grid] * dim), indexing='ij'))).T.reshape(-1, dim)
            if len(grid_points) >= n:
                # Shuffle and take first n
                idx = rng.permutation(len(grid_points))[:n]
                return grid_points[idx]
            else:
                # Pad with random points
                remaining = n - len(grid_points)
                random_points = rng.uniform(0, 1, (remaining, dim))
                return np.vstack([grid_points, random_points])

        # Global exploration phase: sample with Sobol sequence
        n_global = min(100, budget // 5)  # Use first 20% of budget for global exploration
        global_points = sobol_points(n_global, dim)
        # Map from [0,1]^dim to [-5,5]^dim
        global_points = global_points * 10.0 - 5.0

        best_x = None
        best_f = np.inf
        call_count = 0

        # Evaluate global points
        for x in global_points:
            if time.time() > deadline:
                break
            x = np.clip(x, -5, 5)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Local refinement phase: Nelder-Mead on top candidates
        # We'll refine the best point found so far
        # Use a simple Nelder-Mead implementation

        def nelder_mead(func, x0, dim, max_iter=100, tol=1e-6, time_limit=None):
            """Simple Nelder-Mead simplex method."""
            if time_limit is not None and time.time() > time_limit:
                return x0, func(x0)

            # Initialize simplex
            alpha = 1.5  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma = 0.5  # shrink

            # Initial simplex
            simplex = [x0.copy()]
            fvals = [func(x0)]
            call_count_local = 1

            for i in range(dim):
                xi = x0.copy()
                xi[i] += 0.5 if x0[i] < 4.5 else -0.5
                xi = np.clip(xi, -5, 5)
                simplex.append(xi)
                fvals.append(func(xi))
                call_count_local += 1
                if time_limit is not None and time.time() > time_limit:
                    break

            fvals = np.array(fvals)
            simplex = np.array(simplex)

            for _ in range(max_iter):
                if time_limit is not None and time.time() > time_limit:
                    break

                # Sort by function value
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]

                # Check convergence
                if np.ptp(fvals) < tol:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                call_count_local += 1

                if fvals[0] <= fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                elif fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (xr - centroid)
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    call_count_local += 1
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    call_count_local += 1
                    if fc < min(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func(simplex[i])
                            call_count_local += 1
                            if time_limit is not None and time.time() > time_limit:
                                break
                        if time_limit is not None and time.time() > time_limit:
                            break

            # Return best point
            best_idx = np.argmin(fvals)
            return simplex[best_idx], fvals[best_idx]

        # Refine the best point from global exploration
        if best_x is not None and call_count < budget:
            # Use remaining budget for local refinement
            remaining_budget = budget - call_count
            # Allocate a portion for Nelder-Mead
            nm_max_iter = min(100, remaining_budget // 2)

            x_refined, f_refined = nelder_mead(
                func, best_x, dim,
                max_iter=nm_max_iter,
                tol=1e-6,
                time_limit=deadline
            )

            if f_refined < best_f:
                best_f = f_refined
                best_x = x_refined

            # If we still have budget, do additional local search with smaller steps
            remaining = budget - call_count - nm_max_iter * (dim + 2)
            if remaining > 0 and time.time() < deadline:
                # Do a fine local search around the best point
                sigma_local = 0.1
                for _ in range(min(remaining, 50)):
                    if time.time() > deadline:
                        break
                    y = np.clip(best_x + sigma_local * np.random.randn(dim), -5, 5)
                    fy = func(y)
                    if fy < best_f:
                        best_f = fy
                        best_x = y
                        sigma_local *= 1.2
                    else:
                        sigma_local *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
