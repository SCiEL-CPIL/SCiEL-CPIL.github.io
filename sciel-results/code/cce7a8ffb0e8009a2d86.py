"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead + coordinate descent for BBOB f19."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls = 0

        def try_call(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def nelder_mead(x0, f0, max_calls, deadline):
            nonlocal calls, best_x, best_f

            if calls >= max_calls or time.time() >= deadline:
                return best_x, best_f

            simplex = [x0.copy()]
            simplex_f = [f0]

            step = 0.3
            for i in range(dim):
                perturbation = x0.copy()
                perturbation[i] += step
                simplex.append(perturbation)

            for i in range(1, dim + 1):
                if calls >= max_calls or time.time() >= deadline:
                    return best_x, best_f
                fx_i = try_call(simplex[i])
                if fx_i is None:
                    return best_x, best_f
                simplex_f.append(fx_i)

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_nm = 0.5

            for _ in range(150):
                if calls >= max_calls or time.time() >= deadline:
                    break

                sorted_idx = np.argsort(simplex_f)
                simplex = [simplex[i].copy() for i in sorted_idx]
                simplex_f = [simplex_f[i] for i in sorted_idx]

                if simplex_f[0] < best_f:
                    best_f = simplex_f[0]
                    best_x = simplex[0].copy()

                spread = max(simplex_f) - min(simplex_f)
                if spread < 1e-14:
                    break

                centroid = np.mean(simplex[:-1], axis=0)

                x_r = centroid + alpha * (centroid - simplex[-1])
                f_r = try_call(x_r)
                if f_r is None:
                    break

                if f_r < simplex_f[0]:
                    x_e = centroid + gamma * (x_r - centroid)
                    f_e = try_call(x_e)
                    if f_e is None:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < simplex_f[-2]:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    f_c = try_call(x_c)
                    if f_c is None:
                        break
                    if f_c < simplex_f[-1]:
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex_f[i] = try_call(simplex[i])
                            if simplex_f[i] is None:
                                break
                        if simplex_f[-1] is None:
                            break

            return best_x, best_f

        def coordinate_descent(x0, max_calls, deadline):
            nonlocal calls, best_x, best_f
            if calls >= max_calls or time.time() >= deadline:
                return best_x, best_f

            x = x0.copy()
            f = best_f
            step = 0.1
            no_improve = 0

            for _ in range(80):
                if calls >= max_calls or time.time() >= deadline:
                    break
                if no_improve > 20:
                    break

                improved = False
                for i in range(dim):
                    for direction in [1, -1]:
                        x_try = x.copy()
                        x_try[i] += direction * step
                        x_try = np.clip(x_try, -5.0, 5.0)
                        f_try = try_call(x_try)
                        if f_try is None:
                            return best_x, best_f
                        if f_try < f - 1e-13:
                            x = x_try
                            f = f_try
                            best_f = f
                            best_x = x.copy()
                            improved = True
                            no_improve = 0
                            break
                    if calls >= max_calls or time.time() >= deadline:
                        break

                if not improved:
                    step *= 0.5
                    no_improve += 1
                    if step < 1e-9:
                        break
                else:
                    step = max(step * 1.2, 0.01)

            return best_x, best_f

        # Multi-start strategy: generate diverse starting points
        # First evaluate origin
        origin = np.zeros(dim)
        f_origin = try_call(origin)
        if f_origin is None:
            return best_x, best_f

        # Generate several random starts spread across the domain
        n_starts = 8
        starts = [origin]
        for i in range(n_starts - 1):
            start = rng.uniform(-4.0, 4.0, size=dim)
            starts.append(start)

        # Allocate budget: first start gets more, subsequent ones get less
        base_nm_budget = 30
        base_cd_budget = 15
        total_per_start = base_nm_budget + base_cd_budget
        first_start_budget = min(int(0.4 * budget), budget - (n_starts - 1) * total_per_start)

        # Process each start
        for idx, start in enumerate(starts):
            if calls >= budget or time.time() >= deadline:
                break

            # Evaluate the start point
            if idx == 0:
                f_start = f_origin
            else:
                f_start = try_call(start)
                if f_start is None:
                    break

            # Determine budget for this start
            if idx == 0:
                nm_budget = first_start_budget
            else:
                nm_budget = base_nm_budget

            # Run Nelder-Mead from this start
            nm_end_budget = calls + nm_budget
            best_x, best_f = nelder_mead(start, f_start, nm_end_budget, deadline)

            # Run coordinate descent polish
            if best_x is not None and calls < budget and time.time() < deadline:
                cd_end_budget = calls + base_cd_budget
                best_x, best_f = coordinate_descent(best_x, best_f, cd_end_budget, deadline)

        # Final polish if budget remains
        if best_x is not None and calls < budget and time.time() < deadline:
            best_x, best_f = coordinate_descent(best_x, best_f, budget, deadline)

        return best_x, best_f
# EVOLVE-BLOCK-END
