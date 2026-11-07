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

        # Initialize simplex with a larger step to cover the domain better
        x0 = np.zeros(dim)
        initial_step = 2.0
        current_step = initial_step

        simplex = [x0.copy()]
        simplex_f = [func(x0)]
        calls_used = 1

        for i in range(dim):
            xi = x0.copy()
            xi[i] += current_step
            simplex.append(xi)
            simplex_f.append(func(xi))
            calls_used += 1

        best_x = simplex[0].copy()
        best_f = simplex_f[0]

        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5

        def sort_simplex():
            nonlocal simplex, simplex_f
            idx = np.argsort(simplex_f)
            simplex = [simplex[i] for i in idx]
            simplex_f = [simplex_f[i] for i in idx]

        sort_simplex()
        best_x = simplex[0].copy()
        best_f = simplex_f[0]

        while calls_used < budget:
            # Check for convergence/stagnation
            f_spread = simplex_f[-1] - simplex_f[0]
            x_spread = np.max(np.abs(np.array(simplex) - np.array(simplex[0])))
            
            if f_spread < 1e-6 and x_spread < 1e-3:
                # Restart simplex around the best point to escape local minima
                if calls_used + dim < budget:
                    current_step = 2.0
                    simplex = [best_x.copy()]
                    simplex_f = [best_f]
                    
                    for i in range(dim):
                        xi = best_x.copy()
                        xi[i] += current_step
                        xi = np.clip(xi, -5, 5)
                        simplex.append(xi)
                        simplex_f.append(func(xi))
                        calls_used += 1
                        
                    sort_simplex()
                    if simplex_f[0] < best_f:
                        best_f = simplex_f[0]
                        best_x = simplex[0].copy()
                    continue

            # Compute centroid
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            calls_used += 1

            if f_r < simplex_f[0]:
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
                simplex[-1] = x_r
                simplex_f[-1] = f_r
            else:
                if calls_used >= budget:
                    break
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
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        simplex_f[i] = func(simplex[i])
                        calls_used += 1
                        if calls_used >= budget:
                            break
                    if calls_used >= budget:
                        break

            min_idx = np.argmin(simplex_f)
            if simplex_f[min_idx] < best_f:
                best_f = simplex_f[min_idx]
                best_x = simplex[min_idx].copy()

            sort_simplex()

        return best_x, best_f
# EVOLVE-BLOCK-END
