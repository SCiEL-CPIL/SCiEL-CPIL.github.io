"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead with multi-start for BBOB f22 (Gallagher 21 peaks)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf
        total_calls = 0

        def try_eval(x):
            nonlocal total_calls, best_x, best_f
            if total_calls >= budget:
                return None
            if time.time() > deadline - 3:
                return None
            x = np.clip(x, -5, 5)
            f = func(x)
            total_calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Nelder-Mead simplex method
        def nelder_mead(x0, max_calls):
            """Nelder-Mead with adaptive simplex."""
            nonlocal total_calls, best_x, best_f
            if max_calls < 10:
                return x0, best_f

            # Initialize simplex
            simplex = [x0.copy() + 0.1 * rng.standard_normal(n) for _ in range(n + 1)]
            simplex[0] = x0.copy()
            for i in range(1, n + 1):
                simplex[i] = x0.copy()
                simplex[i][i - 1] += 0.5

            fvals = []
            for i in range(n + 1):
                f = try_eval(simplex[i])
                if f is None:
                    return x0, best_f
                fvals.append(f)

            best_x_local = simplex[0].copy()
            best_f_local = min(fvals)
            best_idx = fvals.index(best_f_local)

            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            for _ in range(max_calls):
                if total_calls >= budget or time.time() > deadline - 5:
                    break

                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                best_x_local = simplex[0].copy()
                best_f_local = fvals[0]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                f_ref = try_eval(x_ref)
                if f_ref is None:
                    break

                if f_ref < fvals[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    f_exp = try_eval(x_exp)
                    if f_exp is None:
                        simplex[-1] = x_ref
                        fvals[-1] = f_ref
                        break
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        fvals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        fvals[-1] = f_ref
                elif f_ref < fvals[-2]:
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < fvals[-1]:
                        # Outside contraction
                        x_con = centroid + 0.5 * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - 0.5 * (simplex[-1] - centroid)
                    f_con = try_eval(x_con)
                    if f_con is None:
                        break
                    if f_con <= min(f_ref, fvals[-1]):
                        simplex[-1] = x_con
                        fvals[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f = try_eval(simplex[i])
                            if f is None:
                                return best_x_local, best_f_local
                            fvals[i] = f

                # Check convergence
                f_range = max(fvals) - min(fvals)
                x_range = np.max(np.std(simplex, axis=0))
                if f_range < 1e-10 and x_range < 1e-8:
                    break

            return best_x_local, best_f_local

        # --- Phase 1: Initial sampling ---
        n_initial = min(25, budget // 10)
        structured_points = [
            np.zeros(n),
            np.ones(n) * 2,
            -np.ones(n) * 2,
            np.ones(n) * 1,
            -np.ones(n) * 1,
            np.array([3.0, -3.0, 2.0, -2.0, 1.0]),
            np.array([-2.0, 3.0, -1.0, 2.0, -3.0]),
        ]
        for p in structured_points:
            if total_calls >= budget:
                break
            try_eval(p)

        for i in range(n_initial):
            if total_calls >= budget or time.time() > deadline - 5:
                break
            scale = 3.0 * (1.0 - i / max(1, n_initial)) + 0.5
            x_try = rng.normal(0, scale, n)
            try_eval(x_try)

        if best_x is None:
            return None, np.inf

        # --- Phase 2: Multi-start Nelder-Mead ---
        # Start from best point and a few diverse points
        starts = [best_x.copy()]
        # Add a few random starts
        for _ in range(3):
            x_rand = rng.uniform(-4, 4, n)
            f_rand = try_eval(x_rand)
            if f_rand is not None:
                starts.append(x_rand.copy())

        # Run Nelder-Mead on each start
        remaining_budget = budget - total_calls
        budget_per_start = max(15, remaining_budget // len(starts))
        for i, x_start in enumerate(starts):
            if total_calls >= budget - 5 or time.time() > deadline - 10:
                break
            remaining = budget - total_calls
            alloc = min(budget_per_start, remaining)
            if alloc < 10:
                break
            x_opt, f_opt = nelder_mead(x_start, alloc)
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()

        # --- Phase 3: Final polish ---
        if total_calls < budget and best_x is not None and time.time() < deadline - 5:
            # Small Nelder-Mead from best point
            if budget - total_calls >= 20:
                x_opt, f_opt = nelder_mead(best_x.copy(), min(30, budget - total_calls))
                if f_opt < best_f:
                    best_f = f_opt
                    best_x = x_opt.copy()

            # Tiny perturbations
            n_final = min(10, budget - total_calls)
            for i in range(n_final):
                if total_calls >= budget or time.time() > deadline - 3:
                    break
                perturbation = rng.normal(0, 0.01, n)
                x_try = best_x + perturbation
                f_try = try_eval(x_try)

        return best_x, best_f
# EVOLVE-BLOCK-END
