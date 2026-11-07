"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Nelder-Mead multi-start + L-BFGS-B polish for Rosenbrock (BBOB f8)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start_time = time.time()
        time_limit = 280.0  # Safety margin for 300s/330s limit
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

        # --- Phase 1: Multi-start Nelder-Mead exploration ---
        # Rosenbrock has a narrow valley; Nelder-Mead navigates it well
        # without gradients. Multiple diverse starts increase basin coverage.
        num_starts = 8
        nm_maxiter = 80  # iterations per NM run (each iter ~1 call)

        # Diverse starting points: valley-aligned + random
        starts = []
        valley_dir = np.ones(dim) / np.sqrt(dim)
        for offset in [0.0, 2.0, -2.0, 3.0, -3.0, 4.0]:
            starts.append(np.clip(offset * valley_dir + rng.uniform(-0.8, 0.8, dim), -5, 5))
        starts.append(rng.uniform(-4, 4, dim))
        starts.append(rng.uniform(-4, 4, dim))

        for s_idx, x0 in enumerate(starts):
            if not check_time() or calls_made >= budget - 40:
                break

            x = np.clip(x0, -5, 5)
            try:
                fx = safe_call(x)
            except StopIteration:
                break

            # Nelder-Mead simplex method
            # Build initial simplex
            simplex = [x.copy()]
            for j in range(dim):
                dx = x.copy()
                step = 0.5 if abs(x[j]) < 1.0 else 0.2
                dx[j] += step
                simplex.append(np.clip(dx, -5, 5))

            # Evaluate simplex vertices
            simplex_f = [fx]
            for v in simplex[1:]:
                if not check_time() or calls_made >= budget - 40:
                    break
                try:
                    simplex_f.append(safe_call(v))
                except StopIteration:
                    break

            # Nelder-Mead iterations
            alpha, gamma, rho, sigma_nm = 1.0, 2.0, -0.5, 0.5
            for it in range(nm_maxiter):
                if not check_time() or calls_made >= budget - 40:
                    break
                if len(simplex_f) < dim + 1:
                    break

                # Sort by function value
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                try:
                    fxr = safe_call(xr)
                except StopIteration:
                    break

                if simplex_f[0] <= fxr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fxr
                elif fxr < simplex_f[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    try:
                        fxe = safe_call(xe)
                    except StopIteration:
                        break
                    if fxe < fxr:
                        simplex[-1] = xe
                        simplex_f[-1] = fxe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fxr
                else:
                    # Contraction
                    if fxr < simplex_f[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                        xc = np.clip(xc, -5, 5)
                    else:
                        # Inside contraction
                        xc = centroid + rho * (simplex[-1] - centroid)
                        xc = np.clip(xc, -5, 5)
                    try:
                        fxc = safe_call(xc)
                    except StopIteration:
                        break
                    if fxc < min(fxr, simplex_f[-1]):
                        simplex[-1] = xc
                        simplex_f[-1] = fxc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                        try:
                            simplex_f[i] = safe_call(simplex[i])
                        except StopIteration:
                            break

        # --- Phase 2: L-BFGS-B polish from the best point ---
        if check_time() and calls_made < budget - 5 and best_x is not None:
            def lbfun(x_in):
                nonlocal calls_made
                if calls_made >= budget:
                    raise StopIteration
                calls_made += 1
                return float(func(x_in))

            try:
                res = minimize(lbfun, best_x, method='L-BFGS-B',
                               bounds=[(-5.0, 5.0) for _ in range(dim)],
                               options={'maxiter': 300, 'maxfun': budget - calls_made,
                                        'gtol': 1e-8})
                if res.fun < best_f:
                    best_f = float(res.fun)
                    best_x = res.x.copy()
            except StopIteration:
                pass
            except Exception:
                pass

        # --- Phase 3: Second L-BFGS-B from perturbed best if still not optimal ---
        if check_time() and calls_made < budget - 5 and best_f > 0.1 and best_x is not None:
            x2 = np.clip(best_x + 0.5 * rng.standard_normal(dim), -5, 5)
            try:
                safe_call(x2)

                def lbfun2(x_in):
                    nonlocal calls_made
                    if calls_made >= budget:
                        raise StopIteration
                    calls_made += 1
                    return float(func(x_in))

                res2 = minimize(lbfun2, x2, method='L-BFGS-B',
                                bounds=[(-5.0, 5.0) for _ in range(dim)],
                                options={'maxiter': 200, 'maxfun': budget - calls_made,
                                         'gtol': 1e-8})
                if res2.fun < best_f:
                    best_f = float(res2.fun)
                    best_x = res2.x.copy()
            except StopIteration:
                pass
            except Exception:
                pass

        # --- Phase 4: Final short random refinement ---
        if check_time() and calls_made < budget - 2 and best_x is not None:
            for _ in range(min(30, budget - calls_made)):
                if not check_time() or calls_made >= budget:
                    break
                x3 = np.clip(best_x + 0.1 * rng.standard_normal(dim), -5, 5)
                try:
                    safe_call(x3)
                except StopIteration:
                    break

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = 0.0

        return best_x, best_f
# EVOLVE-BLOCK-END
