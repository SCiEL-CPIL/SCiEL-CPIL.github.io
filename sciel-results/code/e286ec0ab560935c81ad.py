"""Optimizer for BBOB-style Schwefel function f20 on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Global grid exploration followed by multi-start Nelder-Mead local search."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._calls = 0

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        self._calls = 0
        deadline = time.time() + 290  # safety margin before 300s limit

        def f(x):
            self._calls += 1
            return func(np.clip(x, -5, 5))

        # Phase 1: Systematic grid exploration
        # 3^5 = 243 points at {-5, 0, 5}^5
        grid_vals = np.array([-5.0, 0.0, 5.0])
        results = []

        for i0 in range(3):
            for i1 in range(3):
                for i2 in range(3):
                    for i3 in range(3):
                        for i4 in range(3):
                            if self._calls >= budget:
                                break
                            x = np.array([grid_vals[i0], grid_vals[i1],
                                          grid_vals[i2], grid_vals[i3],
                                          grid_vals[i4]])
                            fx = f(x)
                            results.append((fx, x.copy()))
                        if self._calls >= budget:
                            break
                    if self._calls >= budget:
                        break
                if self._calls >= budget:
                    break
            if self._calls >= budget:
                break

        if not results:
            # Fallback: single random point
            x = np.random.uniform(-5, 5, dim)
            fx = f(x)
            return x, fx

        # Sort by function value
        results.sort(key=lambda t: t[0])
        best_f = results[0][0]
        best_x = results[0][1]

        if self._calls >= budget:
            return best_x, best_f

        # Phase 2: Nelder-Mead from top starting points
        # Use top 3 starts to conserve budget
        n_starts = min(3, len(results))
        for idx in range(n_starts):
            if self._calls >= budget or time.time() > deadline:
                break
            start_x = results[idx][1].copy()
            x_opt, f_opt = self._nelder_mead(f, start_x, deadline)
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()

        return best_x, best_f

    def _nelder_mead(self, f, x0, deadline):
        """Nelder-Mead simplex method."""
        dim = self.dim

        # Initial simplex
        simplex = [x0.copy()]
        step = 0.5
        for i in range(dim):
            xi = x0.copy()
            xi[i] += step
            simplex.append(xi)

        # Evaluate initial simplex
        fvals = []
        for v in simplex:
            fv = f(v)
            fvals.append(fv)
            if self._calls >= self.budget or time.time() > deadline:
                break

        if len(fvals) < dim + 1:
            # Not enough budget for full simplex, return best so far
            best_idx = np.argmin(fvals) if fvals else 0
            if fvals:
                return simplex[best_idx], fvals[best_idx]
            return x0, np.inf

        # Sort simplex by function value
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]

        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5

        # Limit iterations based on remaining budget
        remaining = self.budget - self._calls
        max_iter = min(150, remaining // 2)  # ~2 calls per iteration

        for _ in range(max_iter):
            if self._calls >= self.budget or time.time() > deadline:
                break

            # Check convergence
            if max(fvals) - min(fvals) < 1e-12:
                break

            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            f_r = f(x_r)

            if f_r < fvals[0]:
                # Expansion
                x_e = centroid + gamma * (centroid - simplex[-1])
                f_e = f(x_e)
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                # Accept reflection
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                # Contraction
                if f_r < fvals[-1]:
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    f_c = f(x_c)
                    if f_c <= f_r:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        break
                else:
                    x_c = centroid - rho * (simplex[-1] - centroid)
                    f_c = f(x_c)
                    if f_c < fvals[-1]:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        break

            # Re-sort
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

        best_idx = np.argmin(fvals)
        return simplex[best_idx], fvals[best_idx]
# EVOLVE-BLOCK-END
