"""Rewritten program using Nelder-Mead simplex optimization."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Nelder-Mead simplex optimizer for the BBOB-style f6 (Attractive sector)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Initialize simplex: 6 points (dim + 1)
        # Start with a reasonable initial point and build simplex around it
        x0 = np.zeros(dim)
        f0 = func(x0)

        # Build initial simplex
        # Use a scaled coordinate basis
        step = 1.0
        simplex = [x0.copy()]
        for i in range(dim):
            xi = x0.copy()
            xi[i] += step
            simplex.append(xi)

        # Evaluate all simplex points
        simplex_f = [f0]
        for i in range(1, dim + 1):
            simplex_f.append(func(simplex[i]))

        calls_used = dim + 1

        # Nelder-Mead parameters
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink

        # Track best
        best_x = simplex[0].copy()
        best_f = f0

        # Sort simplex by function value
        def sort_simplex():
            nonlocal simplex, simplex_f
            idx = np.argsort(simplex_f)
            simplex = [simplex[i] for i in idx]
            simplex_f = [simplex_f[i] for i in idx]

        sort_simplex()
        best_x = simplex[0].copy()
        best_f = simplex_f[0]

        while calls_used < budget:
            # Check convergence
            f_spread = simplex_f[-1] - simplex_f[0]
            x_spread = np.max(np.abs(np.array(simplex) - np.array(simplex[0])))
            if f_spread < 1e-10 and x_spread < 1e-8:
                # Perturb to escape potential stagnation
                simplex[0] = simplex[0] + 0.1 * np.random.randn(dim)
                simplex[0] = np.clip(simplex[0], -5, 5)
                simplex_f[0] = func(simplex[0])
                calls_used += 1
                sort_simplex()
                continue

            # Compute centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            calls_used += 1

            if f_r < simplex_f[0]:
                # Try expansion
                if calls_used >= budget:
                    break
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                calls_used += 1
                if f_e < f_r:
                    simplex[-1] = x_e
                    simplex_f[-1] = f_e
                else:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
            elif f_r < simplex_f[-2]:
                # Accept reflection
                simplex[-1] = x_r
                simplex_f[-1] = f_r
            else:
                # Contraction
                if calls_used >= budget:
                    break
                # Outside contraction if f_r < f_w, else inside contraction
                if f_r < simplex_f[-1]:
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    x_c = centroid - rho * (simplex[-1] - centroid)
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                calls_used += 1

                if f_c < min(f_r, simplex_f[-1]):
                    simplex[-1] = x_c
                    simplex_f[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        simplex_f[i] = func(simplex[i])
                        calls_used += 1
                        if calls_used >= budget:
                            break
                    if calls_used >= budget:
                        break

            # Update best
            min_idx = np.argmin(simplex_f)
            if simplex_f[min_idx] < best_f:
                best_f = simplex_f[min_idx]
                best_x = simplex[min_idx].copy()

            sort_simplex()

        return best_x, best_f
# EVOLVE-BLOCK-END
