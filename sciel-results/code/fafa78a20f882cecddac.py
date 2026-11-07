"""Nelder-Mead based optimizer for BBOB f14 (Different powers)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Nelder-Mead simplex method with multi-start and adaptive step sizes.
    - Multiple random starts to handle hidden shifts/rotations.
    - Nelder-Mead for efficient local convergence on smooth functions.
    - Deadline guard for safety.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 295
        rng = np.random.default_rng(42)
        d = self.dim
        max_calls = self.budget

        best_x = None
        best_f = float('inf')
        calls_used = 0

        def try_eval(x):
            nonlocal best_x, best_f, calls_used
            if calls_used >= max_calls:
                return None
            if time.time() > deadline - 2:
                return None
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        def nelder_mead(start_x, max_iters=200):
            """
            Nelder-Mead simplex optimization.
            Returns (best_x_local, best_f_local)
            """
            nonlocal calls_used, best_x, best_f

            # Initialize simplex
            simplex = [start_x.copy()]
            for i in range(d):
                v = start_x.copy()
                v[i] += 0.5
                simplex.append(v)

            fvals = []
            for v in simplex:
                fx = try_eval(v)
                if fx is None:
                    return start_x, start_f
                fvals.append(fx)

            start_f = min(fvals)

            # Nelder-Mead parameters
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            for it in range(max_iters):
                if calls_used >= max_calls or time.time() > deadline - 3:
                    break

                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Check convergence
                f_range = fvals[-1] - fvals[0]
                if f_range < 1e-10:
                    break

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = try_eval(x_r)
                if f_r is None:
                    break

                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = try_eval(x_e)
                    if f_e is None:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    if f_r < fvals[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (x_r - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = try_eval(x_c)
                    if f_c is None:
                        break

                    if f_c < min(f_r, fvals[-1]):
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, d + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_i = try_eval(simplex[i])
                            if f_i is None:
                                return simplex[0], fvals[0]
                            fvals[i] = f_i

                # Update global best
                local_best_f = min(fvals)
                if local_best_f < best_f:
                    best_f = local_best_f
                    best_x = simplex[np.argmin(fvals)].copy()

            # Return best from simplex
            local_best_idx = np.argmin(fvals)
            return simplex[local_best_idx].copy(), fvals[local_best_idx]

        # Phase 1: Multi-start Nelder-Mead
        # Number of starts depends on budget
        n_starts = min(8, max(3, max_calls // 100))

        starts = []
        # Include origin and some structured points
        starts.append(np.zeros(d))
        for i in range(n_starts - 1):
            starts.append(rng.uniform(-4, 4, d))

        for i, start in enumerate(starts):
            if calls_used >= max_calls or time.time() > deadline - 10:
                break

            # Allocate remaining budget for this start
            remaining = max_calls - calls_used
            iters = min(150, max(20, remaining // (n_starts - i)))

            x_local, f_local = nelder_mead(start, max_iters=iters)

            if f_local < best_f:
                best_f = f_local
                best_x = x_local.copy()

        # Phase 2: Refine around best with a tighter Nelder-Mead
        if best_x is not None and calls_used < max_calls - 20 and time.time() < deadline - 5:
            # Shrink simplex around best
            tight_start = best_x.copy()
            x_refined, f_refined = nelder_mead(tight_start, max_iters=100)
            if f_refined < best_f:
                best_f = f_refined
                best_x = x_refined.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
