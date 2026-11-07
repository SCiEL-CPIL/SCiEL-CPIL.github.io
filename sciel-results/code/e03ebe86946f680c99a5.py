"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with time and call budget guards."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin before 300s limit
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls = 0
        max_calls = self.budget

        def call_func(x):
            nonlocal calls, best_x, best_f
            if calls >= max_calls:
                return best_f
            x_clipped = np.clip(x, -5.0, 5.0)
            fx = func(x_clipped)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # Generate initial points
        n_starts = 10
        start_points = []
        # First point at origin
        start_points.append(np.zeros(self.dim))
        # Random points
        for _ in range(n_starts - 1):
            start_points.append(rng.uniform(-4, 4, self.dim))

        # For each start point, run Nelder-Mead
        for sp in start_points:
            if calls >= max_calls:
                break
            if time.time() > deadline:
                break

            x0 = np.clip(sp, -5.0, 5.0)
            f0 = call_func(x0)

            # Set up simplex
            simplex = [x0.copy()]
            fvals = [f0]

            # Create simplex around x0
            for i in range(self.dim):
                xi = x0.copy()
                if xi[i] != 0:
                    xi[i] += 0.05 * abs(xi[i])
                else:
                    xi[i] += 0.5
                xi = np.clip(xi, -5.0, 5.0)
                fi = call_func(xi)
                simplex.append(xi)
                fvals.append(fi)

            # Nelder-Mead parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5

            max_nm_iters = 200
            for _ in range(max_nm_iters):
                if calls >= max_calls:
                    break
                if time.time() > deadline:
                    break

                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Check convergence
                if abs(fvals[-1] - fvals[0]) < 1e-10:
                    break

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5.0, 5.0)
                fr = call_func(xr)

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5.0, 5.0)
                    fe = call_func(xe)
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
                    # Contraction
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + rho * (simplex[-1] - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5.0, 5.0)
                    fc = call_func(xc)
                    if fc < min(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            fvals[i] = call_func(simplex[i])

            # Update best
            local_best_idx = np.argmin(fvals)
            if fvals[local_best_idx] < best_f:
                best_f = fvals[local_best_idx]
                best_x = simplex[local_best_idx].copy()

        if best_x is None:
            best_x = np.zeros(self.dim)

        return best_x, best_f
# EVOLVE-BLOCK-END
