"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Coordinate-wise golden-section search followed by Nelder-Mead polish,
    with a deadline guard to stay within wall-clock limits."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim
        self.deadline = time.time() + 280  # safety margin before 300s limit
        self.calls_made = 0

    def _eval(self, func, x):
        """Evaluate func at x, tracking call count and deadline."""
        self.calls_made += 1
        if self.calls_made >= self.budget:
            return func(x)  # last allowed call
        val = func(x)
        if time.time() > self.deadline:
            # We'll stop soon, but this call is already made
            pass
        return val

    def _golden_section_1d(self, func, x_fixed, dim_idx, lo, hi, tol=1e-8, max_iter=60):
        """Golden-section search along one coordinate."""
        phi = (np.sqrt(5.0) - 1.0) / 2.0
        a, b = lo, hi
        c = b - phi * (b - a)
        d = a + phi * (b - a)

        # Build points
        x_c = x_fixed.copy()
        x_c[dim_idx] = c
        x_d = x_fixed.copy()
        x_d[dim_idx] = d

        fc = self._eval(func, x_c)
        fd = self._eval(func, x_d)

        for _ in range(max_iter):
            if time.time() > self.deadline or self.calls_made >= self.budget:
                break
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - phi * (b - a)
                x_c = x_fixed.copy()
                x_c[dim_idx] = c
                fc = self._eval(func, x_c)
            else:
                a = c
                c = d
                fc = fd
                d = a + phi * (b - a)
                x_d = x_fixed.copy()
                x_d[dim_idx] = d
                fd = self._eval(func, x_d)
            if abs(b - a) < tol:
                break

        best_val = min(fc, fd)
        best_pt = c if fc < fd else d
        return best_pt, best_val

    def _nelder_mead(self, func, x0, max_iter=200, tol=1e-8):
        """Nelder-Mead simplex method."""
        d = self.dim
        # Build initial simplex
        simplex = [x0.copy() for _ in range(d + 1)]
        fvals = [self._eval(func, simplex[0])]
        for i in range(1, d + 1):
            simplex[i] = x0.copy()
            if simplex[i][i - 1] != 0:
                simplex[i][i - 1] += 0.05 * abs(simplex[i][i - 1])
            else:
                simplex[i][i - 1] += 0.05
            simplex[i] = np.clip(simplex[i], -5, 5)
            fvals.append(self._eval(func, simplex[i]))

        alpha, gamma, rho = 1.0, 2.0, 0.5  # reflection, expansion, contraction

        for _ in range(max_iter):
            if time.time() > self.deadline or self.calls_made >= self.budget:
                break

            # Sort by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            if abs(fvals[-1] - fvals[0]) < tol:
                break

            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = self._eval(func, x_r)

            if f_r < fvals[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = self._eval(func, x_e)
                if f_e < f_r:
                    simplex[-1], fvals[-1] = x_e, f_e
                else:
                    simplex[-1], fvals[-1] = x_r, f_r
            elif f_r < fvals[-2]:
                simplex[-1], fvals[-1] = x_r, f_r
            else:
                # Contraction
                x_c = centroid + rho * (simplex[-1] - centroid)
                x_c = np.clip(x_c, -5, 5)
                f_c = self._eval(func, x_c)
                if f_c < fvals[-1]:
                    simplex[-1], fvals[-1] = x_c, f_c
                else:
                    # Shrink
                    for i in range(1, d + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = self._eval(func, simplex[i])

        # Return best point
        best_idx = np.argmin(fvals)
        return simplex[best_idx], fvals[best_idx]

    def __call__(self, func):
        d = self.dim
        self.deadline = time.time() + 280
        self.calls_made = 0

        # Phase 1: Coarse grid to find a good starting region
        # Use a small grid per dimension to identify promising areas
        n_grid = 5
        grid_vals = np.linspace(-5, 5, n_grid)
        best_x = np.zeros(d)
        best_f = np.inf

        # Evaluate a coarse grid (limited calls)
        # We'll do a coordinate-wise scan: for each dim, try grid values
        # while keeping other dims at 0 initially
        for dim_idx in range(d):
            if time.time() > self.deadline or self.calls_made >= self.budget:
                break
            for gv in grid_vals:
                if time.time() > self.deadline or self.calls_made >= self.budget:
                    break
                x_test = best_x.copy()
                x_test[dim_idx] = gv
                x_test = np.clip(x_test, -5, 5)
                f_test = self._eval(func, x_test)
                if f_test < best_f:
                    best_f = f_test
                    best_x = x_test.copy()

        # Phase 2: Coordinate-wise golden-section refinement
        for _ in range(3):  # 3 passes of coordinate refinement
            if time.time() > self.deadline or self.calls_made >= self.budget:
                break
            for dim_idx in range(d):
                if time.time() > self.deadline or self.calls_made >= self.budget:
                    break
                # Golden section search along this coordinate
                x_fixed = best_x.copy()
                pt, val = self._golden_section_1d(func, x_fixed, dim_idx, -5, 5, tol=1e-6, max_iter=30)
                if val < best_f:
                    best_f = val
                    best_x[dim_idx] = pt

        # Phase 3: Nelder-Mead polish
        if time.time() < self.deadline and self.calls_made < self.budget - 10:
            x_nm, f_nm = self._nelder_mead(func, best_x, max_iter=100, tol=1e-8)
            if f_nm < best_f:
                best_f = f_nm
                best_x = x_nm

        return best_x, best_f
# EVOLVE-BLOCK-END
