"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead simplex search with L-BFGS-B polish for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start_time = time.time()
        time_limit = 285.0  # Safety margin for 300s limit
        calls_made = 0
        best_x = None
        best_f = np.inf
        dim = self.dim
        budget = self.budget

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                raise StopIteration
            calls_made += 1
            fx = float(func(x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def check_time():
            return time.time() - start_time < time_limit

        def nelder_mead(x0, max_calls=80):
            """Run Nelder-Mead simplex from x0, using safe_call for evaluations."""
            nonlocal calls_made, best_f, best_x

            # Build initial simplex
            simplex = [x0.copy()]
            base_step = 1.0
            for i in range(dim):
                xi = x0.copy()
                xi[i] += base_step
                simplex.append(xi)

            # Evaluate initial simplex
            fvals = []
            for v in simplex:
                try:
                    fv = safe_call(v)
                except StopIteration:
                    return x0, best_f
                fvals.append(fv)

            alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
            max_iter = max_calls // (dim + 1)

            for it in range(max_iter):
                if not check_time() or calls_made >= budget - 30:
                    break

                # Sort simplex by function value
                idx = np.argsort(fvals)
                simplex = [simplex[i] for i in idx]
                fvals = [fvals[i] for i in idx]

                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                try:
                    fr = safe_call(xr)
                except StopIteration:
                    break

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    try:
                        fe = safe_call(xe)
                    except StopIteration:
                        break
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    xc = centroid + rho * (simplex[-1] - centroid)
                    try:
                        fc = safe_call(xc)
                    except StopIteration:
                        break
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            try:
                                fvals[i] = safe_call(simplex[i])
                            except StopIteration:
                                return simplex[np.argmin(fvals)], best_f
                        break

            # Return best point in simplex
            best_idx = np.argmin(fvals)
            return simplex[best_idx], fvals[best_idx]

        # Phase 1: Multi-start Nelder-Mead exploration
        num_starts = 5
        nm_budget_per_start = (budget - 100) // num_starts

        # Starting points: structured for Rosenbrock
        starts = []
        starts.append(np.zeros(dim))
        for i in range(num_starts - 1):
            starts.append(rng.uniform(-4, 4, dim))

        for s_idx, x0 in enumerate(starts):
            if not check_time() or calls_made >= budget - 50:
                break
            try:
                _ = safe_call(x0)
            except StopIteration:
                break
            x_best, _ = nelder_mead(x0, max_calls=nm_budget_per_start)

        # Phase 2: L-BFGS-B polish from best point
        if check_time() and calls_made < budget - 10 and best_x is not None:
            def lbfun(x_in):
                nonlocal calls_made
                if calls_made >= budget:
                    raise StopIteration
                calls_made += 1
                return float(func(x_in))

            try:
                res = minimize(lbfun, best_x, method='L-BFGS-B',
                               bounds=[(-5.0, 5.0) for _ in range(dim)],
                               options={'maxiter': 200, 'maxfun': budget - calls_made})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except StopIteration:
                pass
            except Exception:
                pass

        # Phase 3: Second polish if first didn't converge well
        if check_time() and calls_made < budget - 10 and best_f > 1.0:
            x2 = np.clip(best_x + 0.3 * rng.standard_normal(dim), -5, 5)
            try:
                f2 = safe_call(x2)
                if f2 < best_f:
                    best_f = f2
                    best_x = x2.copy()

                def lbfun2(x_in):
                    nonlocal calls_made
                    if calls_made >= budget:
                        raise StopIteration
                    calls_made += 1
                    return float(func(x_in))

                res2 = minimize(lbfun2, best_x, method='L-BFGS-B',
                                bounds=[(-5.0, 5.0) for _ in range(dim)],
                                options={'maxiter': 100, 'maxfun': budget - calls_made})
                if res2.fun < best_f:
                    best_f = res2.fun
                    best_x = res2.x.copy()
            except StopIteration:
                pass
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
