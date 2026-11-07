"""Refined multi-start Nelder-Mead with coordinate-descent polish for BBOB-style f6."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead simplex with coordinate-descent local search."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 275.0  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def time_ok():
            return time.time() < deadline

        def budget_ok():
            return calls_made < budget - 1

        # Multi-start Nelder-Mead simplex
        n_starts = 5
        start_seeds = [42, 123, 789, 1024, 5555]
        start_scales = [2.0, 3.0, 4.0, 2.5, 1.5]

        for start_idx in range(n_starts):
            if not (budget_ok() and time_ok()):
                break
            if calls_made >= budget - 15:
                break

            rng = np.random.default_rng(start_seeds[start_idx])
            scale = start_scales[start_idx]

            # Create initial simplex
            center = rng.uniform(-4.0, 4.0, dim)
            simplex = []
            fvals = []

            v0 = np.clip(center, -5, 5)
            f0 = func(v0)
            calls_made += 1
            simplex.append(v0.copy())
            fvals.append(f0)

            if f0 < best_f:
                best_f = f0
                best_x = v0.copy()

            for i in range(dim):
                if not (budget_ok() and time_ok()):
                    break
                vi = v0.copy()
                vi[i] += scale
                vi = np.clip(vi, -5, 5)
                fi = func(vi)
                calls_made += 1
                simplex.append(vi.copy())
                fvals.append(fi)

                if fi < best_f:
                    best_f = fi
                    best_x = vi.copy()

            if len(simplex) < dim + 1:
                break

            simplex = np.array(simplex[:dim + 1])
            fvals = np.array(fvals[:dim + 1])

            # Nelder-Mead parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_nm = 0.5

            max_iters = 80
            for iteration in range(max_iters):
                if not (budget_ok() and time_ok()):
                    break
                if calls_made >= budget - 10:
                    break

                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]

                if np.max(fvals) - np.min(fvals) < 1e-12:
                    break

                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                worst_f = fvals[-1]

                xr = centroid + alpha * (centroid - worst)
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                calls_made += 1

                if fr < fvals[0]:
                    xe = centroid + gamma * (centroid - worst)
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    calls_made += 1

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
                    if fr < worst_f:
                        xc = centroid + rho * (xr - centroid)
                    else:
                        xc = centroid - rho * (centroid - worst)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    calls_made += 1

                    if fc < min(fr, worst_f):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fi = func(simplex[i])
                            calls_made += 1
                            fvals[i] = fi

                min_idx = np.argmin(fvals)
                if fvals[min_idx] < best_f:
                    best_f = fvals[min_idx]
                    best_x = simplex[min_idx].copy()

                if np.linalg.norm(simplex[0] - simplex[-1]) < 1e-10:
                    break

        # Coordinate descent polish: sweep each coordinate with adaptive step
        if best_x is not None and budget_ok() and time_ok():
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            sigma_min = 1e-8
            sigma_max = 2.0

            max_cd_iters = min(50, max(0, budget - calls_made - 2))
            for cd_iter in range(max_cd_iters):
                if not (budget_ok() and time_ok()):
                    break

                improved = False
                for i in range(dim):
                    if not (budget_ok() and time_ok()):
                        break

                    # Try positive and negative steps along coordinate i
                    for direction in (1.0, -1.0):
                        y = x.copy()
                        y[i] += direction * sigma
                        y = np.clip(y, -5, 5)
                        fy = func(y)
                        calls_made += 1

                        if fy < fx:
                            x, fx = y, fy
                            improved = True
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()

                if not improved:
                    sigma *= 0.5
                    if sigma < sigma_min:
                        break
                    continue
                else:
                    sigma = min(sigma * 1.1, sigma_max)

        # Final random restart polish with small perturbations
        if best_x is not None and budget_ok() and time_ok():
            rng = np.random.default_rng(777)
            x = best_x.copy()
            fx = best_f
            sigma = max(1e-4, 0.05)

            max_polish = min(30, max(0, budget - calls_made - 2))
            for _ in range(max_polish):
                if not (budget_ok() and time_ok()):
                    break

                direction = rng.standard_normal(dim)
                direction /= (np.linalg.norm(direction) + 1e-15)
                y = np.clip(x + sigma * direction, -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma = min(sigma * 1.1, 0.5)
                else:
                    sigma = max(sigma * 0.7, 1e-8)

        # Ensure valid result
        if best_x is None:
            best_x = np.zeros(dim)
            if calls_made < budget:
                best_f = func(best_x)
                calls_made += 1
            else:
                best_f = 0.0

        return best_x, best_f
# EVOLVE-BLOCK-END
