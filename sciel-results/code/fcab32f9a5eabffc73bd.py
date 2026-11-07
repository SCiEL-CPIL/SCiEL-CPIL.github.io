"""Rewritten optimizer using Nelder-Mead simplex with adaptive reset and multi-start diversification."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Derivative-free trust-region / simplex-based optimizer with multi-start."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0

    def __call__(self, func):
        self._call_count = 0
        dim = self.dim
        budget = self.budget

        # Wrapper to count calls
        def counted_func(x):
            self._call_count += 1
            return float(func(x))

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        # Multi-start: try a few initial points, allocate remaining budget to best one
        # First, do a quick random search to find a decent starting point
        n_random = min(20, budget // 5)
        candidates = []
        for _ in range(n_random):
            x0 = rng.uniform(-5, 5, dim)
            f0 = counted_func(x0)
            candidates.append((f0, x0))
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

        # Sort candidates and pick the best few for local optimization
        candidates.sort(key=lambda t: t[0])

        # Allocate remaining budget among top candidates
        remaining = budget - n_random
        if remaining <= 0:
            return best_x, best_f

        # Use Nelder-Mead on the best candidate(s)
        # We'll do a single strong local search from the best point,
        # with adaptive restarts if stuck

        x_current = candidates[0][1].copy()
        f_current = candidates[0][0]

        # Nelder-Mead parameters
        initial_alpha = 0.5  # initial simplex size
        max_iter_per_run = 300
        min_sigma = 1e-4

        # Adaptive Nelder-Mead implementation with budget tracking
        def nelder_mead_search(x_start, f_start, max_calls):
            nonlocal self
            dim = self.dim

            # Initialize simplex
            x0 = x_start.copy()
            f0 = f_start

            # Build simplex
            simplex_x = [x0.copy()]
            simplex_f = [f0]

            for i in range(dim):
                xi = x0.copy()
                # Perturb in direction i
                step = initial_alpha if abs(x0[i]) < 1e-10 else 0.5 * abs(x0[i])
                xi[i] += step
                xi = np.clip(xi, -5, 5)
                fi = counted_func(xi)
                simplex_x.append(xi)
                simplex_f.append(fi)

            # Track progress
            best_f_so_far = min(simplex_f)
            best_x_so_far = simplex_x[simplex_f.index(best_f_so_far)].copy()
            no_improve_count = 0
            patience = 50

            # Nelder-Mead iterations
            while self._call_count < max_calls:
                # Sort simplex by function value
                order = np.argsort(simplex_f)
                simplex_x = [simplex_x[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                # Check for convergence
                f_spread = simplex_f[-1] - simplex_f[0]
                x_spread = max(np.max(np.abs(np.array(simplex_x) - np.array(simplex_x[0]))))

                if f_spread < 1e-10 and x_spread < 1e-8:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex_x[:-1], axis=0)

                # Reflection
                xr = centroid + (centroid - simplex_x[-1])
                xr = np.clip(xr, -5, 5)
                fr = counted_func(xr)

                if fr < simplex_f[0]:
                    # Expansion
                    xe = centroid + 2 * (centroid - simplex_x[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = counted_func(xe)
                    if fe < fr:
                        simplex_x[-1] = xe
                        simplex_f[-1] = fe
                    else:
                        simplex_x[-1] = xr
                        simplex_f[-1] = fr
                elif fr < simplex_f[-2]:
                    # Accept reflection
                    simplex_x[-1] = xr
                    simplex_f[-1] = fr
                else:
                    # Contraction
                    if fr < simplex_f[-1]:
                        # Outside contraction
                        xc = centroid + 0.5 * (xr - centroid)
                        xc = np.clip(xc, -5, 5)
                        fc = counted_func(xc)
                        if fc <= fr:
                            simplex_x[-1] = xc
                            simplex_f[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex_x[i] = simplex_x[0] + 0.5 * (simplex_x[i] - simplex_x[0])
                                simplex_x[i] = np.clip(simplex_x[i], -5, 5)
                                simplex_f[i] = counted_func(simplex_x[i])
                    else:
                        # Inside contraction
                        xc = centroid - 0.5 * (centroid - simplex_x[-1])
                        xc = np.clip(xc, -5, 5)
                        fc = counted_func(xc)
                        if fc < simplex_f[-1]:
                            simplex_x[-1] = xc
                            simplex_f[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex_x[i] = simplex_x[0] + 0.5 * (simplex_x[i] - simplex_x[0])
                                simplex_x[i] = np.clip(simplex_x[i], -5, 5)
                                simplex_f[i] = counted_func(simplex_x[i])

                # Update best
                cur_best_f = min(simplex_f)
                if cur_best_f < best_f_so_far - 1e-10:
                    best_f_so_far = cur_best_f
                    best_x_so_far = simplex_x[simplex_f.index(cur_best_f)].copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                if no_improve_count > patience:
                    break

            return best_x_so_far, best_f_so_far

        # Run local search from best candidate
        x_local, f_local = nelder_mead_search(x_current, f_current, budget)
        if f_local < best_f:
            best_f = f_local
            best_x = x_local

        # If we have budget left, try another start from second-best candidate
        if self._call_count < budget and len(candidates) > 1:
            x_start2 = candidates[1][1].copy()
            f_start2 = candidates[1][0]
            x_local2, f_local2 = nelder_mead_search(x_start2, f_start2, budget)
            if f_local2 < best_f:
                best_f = f_local2
                best_x = x_local2

        # Final polish with a very fine Nelder-Mead if budget remains
        if self._call_count < budget and best_x is not None:
            # Use scipy's Nelder-Mead for final polish
            try:
                result = minimize(
                    counted_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 100,
                        'maxfev': max(1, budget - self._call_count),
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
