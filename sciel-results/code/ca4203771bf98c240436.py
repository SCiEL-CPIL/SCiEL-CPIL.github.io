"""Deterministic Nelder-Mead simplex optimizer for BBOB f18."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Nelder-Mead simplex method with adaptive restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Nelder-Mead parameters
        alpha = 1.0    # reflection
        gamma = 2.0    # expansion
        rho = 0.5      # contraction
        sigma = 0.5    # shrink

        # Track best overall
        best_x = None
        best_f = np.inf
        calls = 0

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def build_simplex(center, step_size):
            """Build initial simplex around center with given step size."""
            simplex = []
            fvals = []
            v0 = center.copy()
            simplex.append(v0)
            f0 = func(v0)
            calls += 1
            fvals.append(f0)
            update_best(v0, f0)

            for i in range(dim):
                vi = center.copy()
                vi[i] += step_size
                vi = np.clip(vi, -5, 5)
                simplex.append(vi)
                fi = func(vi)
                calls += 1
                fvals.append(fi)
                update_best(vi, fi)
            return simplex, fvals

        # Initial simplex at centroid of the box
        center = np.zeros(dim)
        simplex, fvals = build_simplex(center, 1.0)

        # Main optimization loop
        iteration = 0
        stall_count = 0
        best_f_prev = best_f

        while calls < budget:
            # Sort simplex by function values
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            # Check for convergence
            f_spread = max(fvals) - min(fvals)
            if f_spread < 1e-10:
                stall_count += 1
            else:
                stall_count = 0

            # Track progress
            if best_f < best_f_prev - 1e-10:
                best_f_prev = best_f
                stall_count = 0

            # If stalled too long, restart with larger steps
            if stall_count > 30 and calls < budget - (dim + 1):
                # Try different step sizes for restart
                for step_size in [0.5, 2.0, 5.0]:
                    if calls >= budget - (dim + 1):
                        break
                    simplex, fvals = build_simplex(best_x, step_size)
                    stall_count = 0
                    break
            elif stall_count > 60 and calls < budget - (dim + 1):
                # More aggressive restart
                simplex, fvals = build_simplex(best_x, 1.0)
                stall_count = 0

            # Centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]

            # Reflection
            xr = centroid + alpha * (centroid - worst)
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            calls += 1
            update_best(xr, fr)

            if fr < fvals[0]:
                # Try expansion
                xe = centroid + gamma * (centroid - worst)
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                calls += 1
                update_best(xe, fe)

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
                if fr < fvals[-1]:
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (centroid - worst)

                xc = np.clip(xc, -5, 5)
                fc = func(xc)
                calls += 1
                update_best(xc, fc)

                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        if calls >= budget:
                            break
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fi = func(simplex[i])
                        fvals[i] = fi
                        calls += 1
                        update_best(simplex[i], fi)
                    if calls >= budget:
                        break

            iteration += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
