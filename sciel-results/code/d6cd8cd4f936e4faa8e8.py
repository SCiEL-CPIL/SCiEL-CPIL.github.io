"""Optimizer for BBOB f17 (Schaffer's F7) using Nelder-Mead with multi-start."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead based optimizer for BBOB f17 (Schaffer's F7)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 330s limit, leave 50s margin

        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_fx = np.inf
        calls_made = 0

        def call_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                return np.inf
            fx = float(func(x))
            calls_made += 1
            return fx

        def time_ok():
            return time.time() < deadline and calls_made < budget

        def nelder_mead(x0, max_calls, alpha=1.0, gamma=2.0, rho=0.5, sigma=2.0):
            """Nelder-Mead simplex method with adaptive restart."""
            nonlocal best_x, best_fx, calls_made

            n = d
            # Build initial simplex
            simplex = [x0.copy()]
            fvals = [call_func(x0)]
            step = 0.5
            for i in range(n):
                xi = x0.copy()
                xi[i] += step
                xi = np.clip(xi, -5, 5)
                simplex.append(xi)
                fvals.append(call_func(xi))

            local_best_x = x0.copy()
            local_best_fx = min(fvals)
            if local_best_fx < best_fx:
                best_fx = local_best_fx
                best_x = local_best_x.copy()

            max_iter = max(10, (max_calls - calls_made) // (n + 1))
            for _ in range(max_iter):
                if not time_ok():
                    break

                # Sort simplex by fitness
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Update best
                if fvals[0] < local_best_fx:
                    local_best_fx = fvals[0]
                    local_best_x = simplex[0].copy()
                    if local_best_fx < best_fx:
                        best_fx = local_best_fx
                        best_x = local_best_x.copy()

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = call_func(xr)

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
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
                        xc = centroid + 0.5 * (xr - centroid)
                        fc = call_func(np.clip(xc, -5, 5))
                        if fc <= fr:
                            simplex[-1] = np.clip(xc, -5, 5)
                            fvals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                fvals[i] = call_func(np.clip(simplex[i], -5, 5))
                    else:
                        # Inside contraction
                        xc = centroid - 0.5 * (centroid - simplex[-1])
                        fc = call_func(np.clip(xc, -5, 5))
                        if fc < fvals[-1]:
                            simplex[-1] = np.clip(xc, -5, 5)
                            fvals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                fvals[i] = call_func(np.clip(simplex[i], -5, 5))

                # Check convergence
                if max(fvals) - min(fvals) < 1e-10:
                    break

            return local_best_x, local_best_fx

        # Phase 1: Multi-start initial sampling
        n_starts = 8
        starts = [np.zeros(d)]
        for i in range(n_starts - 1):
            starts.append(rng.uniform(-4, 4, d))

        for x in starts:
            if not time_ok():
                break
            x = np.clip(x, -5, 5)
            fx = call_func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(d)
            best_fx = call_func(best_x)

        # Phase 2: Nelder-Mead from best point
        if calls_made < budget and time.time() < deadline:
            remaining = budget - calls_made
            nm_calls = min(remaining, 300)
            _, _ = nelder_mead(best_x.copy(), calls_made + nm_calls)

        # Phase 3: Additional Nelder-Mead runs from diverse starts
        for _ in range(3):
            if not time_ok():
                break
            remaining = budget - calls_made
            if remaining < 50:
                break
            # Perturb from best
            x0 = best_x + rng.normal(0, 0.5, d)
            x0 = np.clip(x0, -5, 5)
            fx = call_func(x0)
            if fx < best_fx:
                best_fx = fx
                best_x = x0.copy()

            remaining = budget - calls_made
            if remaining < 30:
                break
            nm_calls = min(remaining, 80)
            _, _ = nelder_mead(x0.copy(), calls_made + nm_calls)

        # Phase 4: Final polish with small perturbations
        if calls_made < budget and time.time() < deadline:
            polish_calls = min(30, budget - calls_made)
            for _ in range(polish_calls):
                if not time_ok():
                    break
                y = best_x + 0.05 * rng.standard_normal(d)
                y = np.clip(y, -5, 5)
                fy = call_func(y)
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
