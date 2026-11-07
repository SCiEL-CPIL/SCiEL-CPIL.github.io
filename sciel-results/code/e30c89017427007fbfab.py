"""Rewritten optimizer using Nelder-Mead simplex with adaptive reset and multi-start diversification."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Derivative-free trust-region / simplex-based optimizer with multi-start."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0

    def __call__(self, func):
        deadline = time.time() + 280  # safety margin before 300s limit
        self._call_count = 0
        best_x = None
        best_f = np.inf

        def counted_func(x):
            nonlocal self
            if self._call_count >= self.budget:
                return np.inf
            self._call_count += 1
            return func(np.clip(x, -5, 5))

        def nelder_mead_search(start_x, max_iter):
            nonlocal best_x, best_f, deadline, self
            x0 = np.clip(start_x, -5, 5)
            f0 = counted_func(x0)
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

            # Initialize simplex
            simplex = [x0.copy()]
            for i in range(self.dim):
                xi = x0.copy()
                xi[i] = x0[i] + 0.5 if x0[i] < 4.5 else x0[i] - 0.5
                simplex.append(xi)

            simplex_f = [f0]
            for i in range(1, self.dim + 1):
                simplex_f.append(counted_func(simplex[i]))
                if simplex_f[-1] < best_f:
                    best_f = simplex_f[-1]
                    best_x = simplex[i].copy()

            alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5

            for _ in range(max_iter):
                if time.time() > deadline or self._call_count >= self.budget:
                    break

                # Sort simplex by function value
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                fr = counted_func(xr)
                if fr < best_f:
                    best_f = fr
                    best_x = xr.copy()

                if fr < simplex_f[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    fe = counted_func(xe)
                    if fe < best_f:
                        best_f = fe
                        best_x = xe.copy()
                    if fe < fr:
                        simplex[-1] = xe
                        simplex_f[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                elif fr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                else:
                    # Contraction
                    xc = centroid + rho * (simplex[-1] - centroid)
                    fc = counted_func(xc)
                    if fc < best_f:
                        best_f = fc
                        best_x = xc.copy()
                    if fc < fr:
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, self.dim + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex_f[i] = counted_func(simplex[i])
                            if simplex_f[i] < best_f:
                                best_f = simplex_f[i]
                                best_x = simplex[i].copy()

            return best_x, best_f

        # Multi-start: use diverse initial points
        rng = np.random.default_rng(42)
        starts = []
        # Center
        starts.append(np.zeros(self.dim))
        # Random points
        for _ in range(5):
            starts.append(rng.uniform(-4, 4, self.dim))

        # Allocate budget to each start
        remaining = self.budget
        for i, start in enumerate(starts):
            if self._call_count >= self.budget or time.time() > deadline:
                break
            alloc = remaining // (len(starts) - i) if i < len(starts) - 1 else remaining
            alloc = max(alloc, 10)
            nelder_mead_search(start, alloc)
            remaining = self.budget - self._call_count

        return best_x, best_f
# EVOLVE-BLOCK-END
