"""Optimized BBOB f14 solver using Nelder-Mead simplex with restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex optimization for BBOB f14 on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 320.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)

        best_x_global = None
        best_f_global = np.inf
        calls_made = 0

        # Nelder-Mead parameters
        alpha = 1.0    # reflection
        gamma = 2.0    # expansion
        rho = 0.5      # contraction
        sigma = 0.5    # shrink

        def nelder_mead(func, x0, budget_remaining, deadline, calls_counter):
            """Run Nelder-Mead starting from x0, using up to budget_remaining calls."""
            nonlocal calls_made
            n = dim
            # Initialize simplex: x0 and n points perturbed in each direction
            simplex = [x0.copy()]
            for i in range(n):
                xi = x0.copy()
                # Perturb in dimension i
                step = 0.5 if x0[i] >= 0 else -0.5
                xi[i] += step
                xi = np.clip(xi, -5, 5)
                simplex.append(xi)

            # Evaluate initial simplex
            fvals = []
            for i in range(n + 1):
                if calls_made >= budget or time.time() > deadline:
                    break
                fvals.append(func(simplex[i]))
                calls_made += 1

            if len(fvals) < n + 1:
                # Not enough budget to initialize full simplex
                best_idx = np.argmin(fvals) if fvals else 0
                if fvals:
                    return simplex[best_idx], fvals[best_idx]
                return x0, np.inf

            # Main loop
            max_iter = budget - calls_made
            for _ in range(max_iter):
                if calls_made >= budget or time.time() > deadline:
                    break

                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i].copy() for i in order]
                fvals = [fvals[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = func(x_ref)
                calls_made += 1

                if f_ref < fvals[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = func(x_exp)
                    calls_made += 1

                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        fvals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        fvals[-1] = f_ref
                elif f_ref < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < fvals[-1]:
                        # Outside contraction
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - rho * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = func(x_con)
                    calls_made += 1

                    if f_con < min(f_ref, fvals[-1]):
                        simplex[-1] = x_con
                        fvals[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func(simplex[i])
                            calls_made += 1

                        if calls_made >= budget or time.time() > deadline:
                            break

                # Check for convergence: if simplex is very small, stop
                spread = np.max(simplex, axis=0) - np.min(simplex, axis=0)
                if np.all(spread < 1e-6):
                    break

            # Return best point in simplex
            best_idx = np.argmin(fvals)
            return simplex[best_idx], fvals[best_idx]

        # Strategy: Multiple restarts with decreasing step sizes
        # Phase 1: One initial Nelder-Mead from origin
        x0 = np.zeros(dim)
        x_best, f_best = nelder_mead(func, x0, budget, deadline, 0)
        if f_best < best_f_global:
            best_f_global = f_best
            best_x_global = x_best.copy()

        # Phase 2: Several restarts from random points around the best
        n_restarts = 4
        for r in range(n_restarts):
            if calls_made >= budget or time.time() > deadline:
                break

            # Generate restart point near current best
            if best_x_global is not None:
                # Perturb around best with decreasing magnitude
                scale = 2.0 / (r + 1)
                x0 = best_x_global + rng.standard_normal(dim) * scale
                x0 = np.clip(x0, -5, 5)
            else:
                x0 = rng.uniform(-5, 5, dim)

            x_best, f_best = nelder_mead(func, x0, budget, deadline, 0)
            if f_best < best_f_global:
                best_f_global = f_best
                best_x_global = x_best.copy()

        # If we still have budget, do a final fine-tune around the global best
        if calls_made < budget and best_x_global is not None and time.time() < deadline:
            x0 = best_x_global.copy()
            x_best, f_best = nelder_mead(func, x0, budget, deadline, 0)
            if f_best < best_f_global:
                best_f_global = f_best
                best_x_global = x_best.copy()

        return best_x_global, best_f_global
# EVOLVE-BLOCK-END
