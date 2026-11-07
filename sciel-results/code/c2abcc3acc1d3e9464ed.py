"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import dual_annealing

class Optimizer:
    """Global optimization using Dual Annealing followed by Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0

        best_x = None
        best_f = np.inf
        calls_made = 0

        def counted_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= self.budget:
                raise RuntimeError("Budget exceeded")
            if time.time() > deadline - 2:
                raise RuntimeError("Time limit")
            calls_made += 1
            # Ensure x is a numpy array for consistent handling
            x = np.array(x, dtype=float)
            val = float(func(x))
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        dim = self.dim
        budget = self.budget
        bounds = [(-5.0, 5.0)] * dim

        # Phase 1: Global Search using Dual Annealing
        # Dual Annealing is a robust global optimizer for multimodal functions.
        # We let it run until it either finishes or hits the budget/time limit.
        try:
            # maxiter is set high to allow the budget to be the limiting factor.
            # seed is fixed for reproducibility.
            res = dual_annealing(
                func=counted_func,
                bounds=bounds,
                maxiter=10000,
                initial_temp=52.5,
                restart_temp_ratio=1.0,
                accept_temp=52.5,
                seed=42
            )
            # If it finished normally, update best
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x
        except RuntimeError:
            pass
        except Exception:
            pass

        # Phase 2: Local Polish using Nelder-Mead
        # Use remaining budget to refine the best point found.
        if best_x is not None and calls_made < budget and time.time() < deadline - 5:
            n = dim
            # Build initial simplex
            simplex = [np.array(best_x, dtype=float, copy=True)]
            # Adaptive step size
            for i in range(n):
                xi = np.array(best_x, dtype=float, copy=True)
                # Start with a small step, as we are likely close to the minimum
                step = 0.1
                xi[i] += step
                simplex.append(xi)
            simplex = [np.clip(x, -5, 5) for x in simplex]

            # Evaluate initial simplex
            fvals = []
            for x in simplex:
                try:
                    fvals.append(counted_func(x))
                except RuntimeError:
                    break

            if len(fvals) == n + 1:
                alpha, gamma, rho = 1.0, 2.0, -0.5
                while calls_made < budget and time.time() < deadline - 2:
                    order = np.argsort(fvals)
                    simplex = [simplex[i] for i in order]
                    fvals = [fvals[i] for i in order]

                    if max(fvals) - min(fvals) < 1e-12:
                        break

                    centroid = np.mean(simplex[:-1], axis=0)
                    xr = centroid + alpha * (centroid - simplex[-1])
                    xr = np.clip(xr, -5, 5)
                    try:
                        fr = counted_func(xr)
                    except RuntimeError:
                        break

                    if fr < fvals[0]:
                        xe = centroid + gamma * (centroid - simplex[-1])
                        xe = np.clip(xe, -5, 5)
                        try:
                            fe = counted_func(xe)
                        except RuntimeError:
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
                        if fr < fvals[-1]:
                            xc = centroid + 0.5 * (xr - centroid)
                        else:
                            xc = centroid - 0.5 * (simplex[-1] - centroid)
                        xc = np.clip(xc, -5, 5)
                        try:
                            fc = counted_func(xc)
                        except RuntimeError:
                            break
                        if fc < fvals[-1]:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                try:
                                    fvals[i] = counted_func(simplex[i])
                                except RuntimeError:
                                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
