# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid Nelder-Mead + (1+1)-ES for BBOB-style optimization.
    
    Phase 1: Multiple Nelder-Mead runs from diverse starting points
    Phase 2: (1+1)-ES polish from the best point found
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 295

        rng = np.random.default_rng(123)

        def nelder_mead(func, x0, budget_nm, deadline):
            """Run Nelder-Mead simplex method with limited budget."""
            calls = 0
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_shrink = 0.5

            # Initialize simplex
            simplex = [x0.copy()]
            fvals = [func(x0)]
            calls += 1

            for i in range(dim):
                xi = x0.copy()
                step = 0.5 if abs(x0[i]) < 1e-10 else 0.1 * abs(x0[i])
                xi[i] += step
                xi = np.clip(xi, -5, 5)
                simplex.append(xi)
                fvals.append(func(xi))
                calls += 1

            # Order by fitness
            order = np.argsort(fvals)
            simplex = [simplex[i].copy() for i in order]
            fvals = [fvals[i] for i in order]

            best_x = simplex[0].copy()
            best_f = fvals[0]

            while calls < budget_nm:
                if time.time() > deadline:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                calls += 1
                if calls >= budget_nm:
                    break

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    calls += 1
                    if calls >= budget_nm:
                        break
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
                        xc = centroid + rho * (xr - centroid)
                    else:
                        xc = centroid - rho * (centroid - simplex[-1])
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    calls += 1
                    if calls >= budget_nm:
                        break
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            if calls >= budget_nm:
                                break
                            simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func(simplex[i])
                            calls += 1

                # Reorder
                order = np.argsort(fvals)
                simplex = [simplex[i].copy() for i in order]
                fvals = [fvals[i] for i in order]

                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = simplex[0].copy()

            return best_x, best_f, calls

        # Phase 1: Nelder-Mead from diverse starting points
        n_starts = 4
        nm_budget_each = 35

        start_points = [np.zeros(dim)]
        for _ in range(n_starts - 1):
            pt = rng.uniform(-3, 3, dim)
            start_points.append(pt)

        best_x = None
        best_f = np.inf
        total_calls = 0

        for sp in start_points:
            if total_calls >= budget:
                break
            if time.time() > deadline:
                break
            remaining = budget - total_calls
            nm_budget = min(nm_budget_each, remaining)
            if nm_budget < dim + 2:
                break
            x_nm, f_nm, calls_nm = nelder_mead(func, sp, nm_budget, deadline)
            total_calls += calls_nm
            if f_nm < best_f:
                best_f = f_nm
                best_x = x_nm.copy()

        # Phase 2: (1+1)-ES polish from best point
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            total_calls += 1

        x = best_x.copy()
        fx = best_f
        sigma = 0.5
        sigma_min = 0.001
        sigma_max = 3.0
        success_window = []
        window_size = 10
        stagnation = 0
        no_improve_count = 0

        while total_calls < budget:
            if time.time() > deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            total_calls += 1

            success = fy < fx
            success_window.append(success)
            if len(success_window) > window_size:
                success_window.pop(0)

            if len(success_window) >= window_size:
                success_rate = sum(success_window) / len(success_window)
                if success_rate > 0.3:
                    sigma = min(sigma * 1.3, sigma_max)
                elif success_rate < 0.1:
                    sigma = max(sigma * 0.6, sigma_min)

            if success:
                x, fx = y, fy
                stagnation = 0
            else:
                stagnation += 1

            if fy < best_f:
                best_f = fy
                best_x = y.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1

            if stagnation >= 30 or no_improve_count >= 50:
                local_sigma = max(sigma * 1.5, 0.3)
                x = np.clip(best_x + local_sigma * rng.standard_normal(dim), -5, 5)
                fx = func(x)
                total_calls += 1
                stagnation = 0
                no_improve_count = 0
                success_window = []
                sigma = local_sigma

                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

                if total_calls >= budget:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
