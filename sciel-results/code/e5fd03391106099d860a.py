"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with Nelder-Mead polish for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300  # 300s safety margin
        budget = self.budget
        dim = self.dim

        def time_left():
            return time.time() < deadline

        # Deterministic multi-start points: center + vertices of a regular simplex
        # inscribed in [-5, 5]^dim, plus a few deterministic offsets
        rng = np.random.default_rng(42)

        # Generate starting points: origin, and deterministic spread
        starts = [np.zeros(dim)]
        # Add vertices of a hypercube corners (deterministic subset)
        for i in range(1, 8):
            pt = rng.uniform(-4.5, 4.5, dim)
            starts.append(pt)

        best_x = None
        best_f = np.inf
        calls = 0

        def call_func(x):
            nonlocal calls
            if not time_left():
                return None
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            fx = func(x)
            calls += 1
            return fx

        # Phase 1: Multi-start (1+1)-ES with adaptive sigma
        # Each start gets a portion of the budget
        n_starts = len(starts)
        budget_per_start = max(30, (budget - 100) // n_starts)  # reserve 100 for polish

        for s_idx, s0 in enumerate(starts):
            if calls >= budget or not time_left():
                break

            x = np.clip(s0.copy(), -5, 5)
            fx = call_func(x)
            if fx is None:
                break
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 1.5
            remaining = min(budget - calls, budget_per_start)
            for _ in range(remaining):
                if not time_left():
                    break
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = call_func(y)
                if fy is None:
                    break
                if fy < fx:
                    x = y.copy()
                    fx = fy
                    sigma *= 1.3
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85
                    if sigma < 0.01:
                        break

        # Phase 2: Nelder-Mead polish from the best point
        if best_x is not None and calls < budget and time_left():
            x_nm = best_x.copy()
            fx_nm = best_f
            # Initialize simplex
            simplex = [x_nm.copy()]
            for i in range(dim):
                pt = x_nm.copy()
                step = 0.5 if abs(x_nm[i]) > 1e-8 else 0.5
                pt[i] += step
                pt = np.clip(pt, -5, 5)
                simplex.append(pt)

            # Evaluate simplex
            simplex_f = [fx_nm]
            for i in range(1, dim + 1):
                if not time_left():
                    break
                f_i = call_func(simplex[i])
                if f_i is None:
                    break
                simplex_f.append(f_i)

            # Nelder-Mead iterations
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_nm = 0.5
            max_nm_iters = min(budget - calls, 80)

            for _ in range(max_nm_iters):
                if not time_left() or calls >= budget:
                    break
                # Sort simplex by function value
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = call_func(x_r)
                if f_r is None:
                    break

                if f_r < simplex_f[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = call_func(x_e)
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
                    # Contraction
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = call_func(x_c)
                    if f_c is None:
                        break
                    if f_c < simplex_f[-1]:
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if not time_left():
                                break
                            f_i = call_func(simplex[i])
                            if f_i is None:
                                break
                            simplex_f[i] = f_i
                        if not time_left():
                            break

                # Track best
                min_idx = np.argmin(simplex_f)
                if simplex_f[min_idx] < best_f:
                    best_f = simplex_f[min_idx]
                    best_x = simplex[min_idx].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
