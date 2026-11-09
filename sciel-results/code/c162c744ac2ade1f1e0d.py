"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with diverse initializations."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 290  # safety margin before 300s limit
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls = 0

        def safe_call(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return None
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Coarse Random Search (20% of budget)
        n_random = max(10, budget // 5)
        for _ in range(n_random):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            safe_call(x)

        if best_x is None:
            # Fallback if no calls were made (should not happen)
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls = 1

        # Phase 2: Nelder-Mead Local Optimization
        # Initialize simplex around best_x
        alpha, gamma, rho = 1.0, 2.0, -0.5
        x0 = best_x.copy()
        simplex = [x0.copy()]
        fvals = [best_f]
        
        # Create initial simplex neighbors
        step_size = 0.5
        for i in range(dim):
            if calls >= budget or time.time() > deadline:
                break
            xi = x0.copy()
            # Expand in direction i, respecting bounds
            if xi[i] < 4.5:
                xi[i] += step_size
            else:
                xi[i] -= step_size
            xi = np.clip(xi, -5, 5)
            fi = safe_call(xi)
            if fi is None:
                break
            simplex.append(xi)
            fvals.append(fi)
        
        if len(simplex) < dim + 1:
            # Not enough budget to form a simplex, return best found
            return best_x, best_f

        # Nelder-Mead Loop
        while calls < budget and time.time() < deadline:
            # Sort simplex
            order = np.argsort(fvals)
            simplex = [simplex[i].copy() for i in order]
            fvals = [fvals[i] for i in order]
            
            # If converged, break
            if abs(fvals[0] - fvals[-1]) < 1e-10:
                break

            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = safe_call(x_r)
            if f_r is None:
                break

            if f_r < fvals[0]:
                # Expansion
                if calls >= budget or time.time() > deadline:
                    break
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = safe_call(x_e)
                if f_e is None:
                    break
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                # Contraction
                if calls >= budget or time.time() > deadline:
                    break
                x_c = centroid + rho * (simplex[-1] - centroid)
                x_c = np.clip(x_c, -5, 5)
                f_c = safe_call(x_c)
                if f_c is None:
                    break
                if f_c < fvals[-1]:
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        if calls >= budget or time.time() > deadline:
                            break
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        f_i = safe_call(simplex[i])
                        if f_i is None:
                            break
                        fvals[i] = f_i

        return best_x, best_f
# EVOLVE-BLOCK-END
