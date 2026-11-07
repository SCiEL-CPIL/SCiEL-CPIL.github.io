"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid coordinate-descent + Nelder-Mead with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s

        def time_ok():
            return time.time() < deadline

        # Track calls and best
        self._calls = 0
        self._best_x = None
        self._best_f = np.inf

        def call_func(x):
            self._calls += 1
            fx = float(func(x))
            if fx < self._best_f:
                self._best_f = fx
                self._best_x = np.array(x, dtype=float)
            return fx

        dim = self.dim
        budget = self.budget

        # Phase 1: Coordinate-wise 1-D search (ternary-like with golden ratio)
        # Start from a few random initial points, do coordinate descent
        rng = np.random.default_rng(42)

        # Try a small set of starting points
        n_starts = 3
        best_x = None
        best_f = np.inf

        for s in range(n_starts):
            if self._calls >= budget or not time_ok():
                break

            if s == 0:
                x0 = np.zeros(dim)
            else:
                x0 = rng.uniform(-4, 4, dim)

            fx = call_func(x0)
            if fx < best_f:
                best_f = fx
                best_x = np.array(x0)

            # Coordinate descent with golden-section 1-D search
            # For each coordinate, do a few golden-section steps
            for coord in range(dim):
                if self._calls >= budget or not time_ok():
                    break
                # Determine bracket around current value
                # Try expanding to find a bracket
                center = best_x[coord]
                step = 1.0
                # Evaluate at center - step, center, center + step
                # We already have center value (best_f)
                # Try to find a direction that improves
                improved = False
                for _ in range(10):
                    if self._calls >= budget or not time_ok():
                        break
                    x_minus = np.array(best_x)
                    x_plus = np.array(best_x)
                    x_minus[coord] = np.clip(center - step, -5, 5)
                    x_plus[coord] = np.clip(center + step, -5, 5)

                    f_minus = call_func(x_minus)
                    if f_minus < best_f:
                        best_f = f_minus
                        best_x = x_minus
                        improved = True
                        # Continue searching in this direction
                        center = x_minus[coord]
                        step *= 0.5
                        continue

                    f_plus = call_func(x_plus)
                    if f_plus < best_f:
                        best_f = f_plus
                        best_x = x_plus
                        improved = True
                        center = x_plus[coord]
                        step *= 0.5
                        continue

                    # No improvement in either direction, shrink step
                    step *= 0.5
                    if step < 0.01:
                        break

        if best_x is None:
            # Fallback: just do random evaluations
            for _ in range(min(budget - self._calls, 100)):
                if self._calls >= budget or not time_ok():
                    break
                x = rng.uniform(-5, 5, dim)
                call_func(x)
            return self._best_x, self._best_f

        # Phase 2: Nelder-Mead refinement
        if self._calls < budget and time_ok():
            best_x = self._best_x.copy()
            best_f = self._best_f

            # Initialize simplex
            simplex = [best_x.copy()]
            f_vals = [best_f]

            for i in range(dim):
                if self._calls >= budget or not time_ok():
                    break
                xi = best_x.copy()
                if xi[i] < 4.0:
                    xi[i] += 0.5
                else:
                    xi[i] -= 0.5
                xi[i] = np.clip(xi[i], -5, 5)
                fi = call_func(xi)
                simplex.append(xi)
                f_vals.append(fi)

            # Nelder-Mead iterations
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            max_nm_iter = 200
            for _ in range(max_nm_iter):
                if self._calls >= budget or not time_ok():
                    break

                # Sort simplex by function value
                order = np.argsort(f_vals)
                simplex = [simplex[i] for i in order]
                f_vals = [f_vals[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = call_func(xr)

                if fr < f_vals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = call_func(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        f_vals[-1] = fe
                    else:
                        simplex[-1] = xr
                        f_vals[-1] = fr
                elif fr < f_vals[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    f_vals[-1] = fr
                else:
                    # Contraction
                    if fr < f_vals[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                    xc = np.clip(xc, -5, 5)
                    fc = call_func(xc)
                    if fc < max(fr, f_vals[-1]):
                        simplex[-1] = xc
                        f_vals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_vals[i] = call_func(simplex[i])

        return self._best_x, self._best_f
# EVOLVE-BLOCK-END
