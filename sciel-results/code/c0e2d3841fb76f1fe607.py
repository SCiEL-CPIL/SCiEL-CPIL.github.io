"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid optimizer for BBOB f13 (Sharp Ridge):
    1. Low-discrepancy Halton initialization for broad early coverage (AOCC).
    2. Adaptive (1+1)-ES with ridge-aware step scaling.
    3. Nelder-Mead refinement for final convergence on narrow ridges.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        box_low = -5.0
        box_high = 5.0
        calls_made = 0

        def clip(x):
            return np.clip(x, box_low, box_high)

        def evaluate(x):
            nonlocal calls_made
            if calls_made >= budget or time.time() > deadline:
                return None
            x = np.asarray(x, dtype=float)
            if x.shape != (dim,):
                x = x.reshape(dim)
            x = clip(x)
            f = func(x)
            calls_made += 1
            return x, f

        # Halton sequence for low-discrepancy initialization
        def halton(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        def halton_point(index, dim, low, high):
            point = np.zeros(dim)
            for d in range(dim):
                base = d + 2  # 2, 3, 5, 7, 11, ...
                point[d] = low + (high - low) * halton(index + 1, base)
            return point

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Broad exploration with Halton sequence
        n_explore = min(80, budget // 6)
        
        for i in range(n_explore):
            if time.time() > deadline - 60:
                break
            x = halton_point(i, dim, box_low, box_high)
            res = evaluate(x)
            if res is None:
                break
            x, f = res
            update_best(x, f)

        # Also evaluate origin and a few structured points
        structured = [
            np.zeros(dim),
            np.ones(dim) * 2.0,
            np.ones(dim) * -2.0,
            np.array([1, -1, 1, -1, 1]) * 2.0,
            np.array([3, -3, 0, 3, -3]),
        ]
        for x in structured:
            if time.time() > deadline - 60:
                break
            res = evaluate(x)
            if res is None:
                break
            x, f = res
            update_best(x, f)

        if best_x is None:
            # Fallback: evaluate origin if no points were evaluated
            x0 = np.zeros(dim)
            res = evaluate(x0)
            if res is None:
                return x0, np.inf
            best_x, best_f = res

        # Phase 2: Adaptive (1+1)-ES refinement
        # This is effective for finding the general direction of the ridge
        if calls_made < budget and time.time() < deadline - 30:
            x = best_x.copy()
            fx = best_f
            sigma = 1.0
            successes = 0
            trials = 0
            alloc = min(100, budget - calls_made)

            for _ in range(alloc):
                if time.time() > deadline - 30:
                    break

                y = clip(x + sigma * np.random.default_rng(42).standard_normal(dim))
                res = evaluate(y)
                if res is None:
                    break
                y, fy = res

                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    sigma *= 1.25
                else:
                    sigma *= 0.85

                trials += 1

                if trials >= 10:
                    success_rate = successes / trials
                    if success_rate > 0.3:
                        sigma *= 1.1
                    elif success_rate < 0.1:
                        sigma *= 0.9
                    successes = 0
                    trials = 0

                update_best(x, fx)

        # Phase 3: Nelder-Mead refinement
        # This adapts the simplex shape to the sharp ridge geometry
        if calls_made < budget and time.time() < deadline - 10:
            x0 = best_x.copy()
            f0 = best_f

            # Initialize simplex
            simplex = [x0.copy()]
            fvals = [f0]

            # Expand simplex along each dimension
            step = 0.5
            for d in range(dim):
                if time.time() > deadline - 10 or calls_made >= budget:
                    break
                x_new = x0.copy()
                x_new[d] += step
                x_new = clip(x_new)
                res = evaluate(x_new)
                if res is None:
                    break
                x_new, f_new = res
                simplex.append(x_new)
                fvals.append(f_new)
                update_best(x_new, f_new)

            if len(simplex) < dim + 1:
                return best_x, best_f

            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma_nm = 0.5 # shrink

            max_iter = 200
            for _ in range(max_iter):
                if time.time() > deadline - 10 or calls_made >= budget:
                    break

                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i].copy() for i in order]
                fvals = [fvals[i] for i in order]

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = clip(x_r)
                res = evaluate(x_r)
                if res is None:
                    break
                x_r, f_r = res
                update_best(x_r, f_r)

                if f_r < fvals[0]:
                    # Try expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = clip(x_e)
                    res = evaluate(x_e)
                    if res is None:
                        break
                    x_e, f_e = res
                    update_best(x_e, f_e)
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
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    x_c = clip(x_c)
                    res = evaluate(x_c)
                    if res is None:
                        break
                    x_c, f_c = res
                    update_best(x_c, f_c)
                    if f_c < f_r:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = clip(simplex[i])
                            res = evaluate(simplex[i])
                            if res is None:
                                return best_x, best_f
                            simplex[i], fvals[i] = res
                            update_best(simplex[i], fvals[i])

                # Check convergence
                if abs(fvals[-1] - fvals[0]) < 1e-10:
                    break

            # Return best point in simplex
            best_idx = np.argmin(fvals)
            if fvals[best_idx] < best_f:
                best_f = fvals[best_idx]
                best_x = simplex[best_idx].copy()

        return best_x, best_f

# EVOLVE-BLOCK-END
