import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        x_best = None
        fx_best = float('inf')

        def track(x, fx):
            nonlocal x_best, fx_best
            if fx < fx_best:
                fx_best = fx
                x_best = x.copy()

        # Evaluate a few diverse initial points
        init_points = [np.zeros(dim)]
        for _ in range(4):
            init_points.append(rng.uniform(-5, 5, dim))

        for x0 in init_points:
            fx0 = func(x0)
            track(x0, fx0)
            if time.time() > deadline:
                break

        # Nelder-Mead helper
        def nelder_mead(x0, n_evals, sigma_scale=1.0):
            nonlocal x_best, fx_best
            if time.time() > deadline:
                return
            dim = self.dim
            # Initialize simplex
            simplex = [x0.copy()]
            fvals = [func(x0)]
            track(x0, fvals[0])
            n_used = 1

            for d in range(dim):
                x_new = x0.copy()
                x_new[d] += sigma_scale
                x_new = np.clip(x_new, -5, 5)
                f_new = func(x_new)
                track(x_new, f_new)
                simplex.append(x_new)
                fvals.append(f_new)
                n_used += 1
                if n_used >= n_evals or time.time() > deadline:
                    return

            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
            max_iter = n_evals - n_used

            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                # Sort simplex by fitness
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                track(simplex[0], fvals[0])

                x_avg = np.mean(simplex[:-1], axis=0)
                x_worst = simplex[-1]

                # Reflection
                x_ref = x_avg + alpha * (x_avg - x_worst)
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = func(x_ref)
                track(x_ref, f_ref)
                n_used += 1

                if fvals[0] <= f_ref < fvals[-2]:
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
                elif f_ref < fvals[0]:
                    # Expansion
                    x_exp = x_avg + gamma * (x_ref - x_avg)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = func(x_exp)
                    track(x_exp, f_exp)
                    n_used += 1
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        fvals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        fvals[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < fvals[-1]:
                        # Outside contraction
                        x_con = x_avg + sigma * (x_ref - x_avg)
                    else:
                        # Inside contraction
                        x_con = x_avg - sigma * (x_worst - x_avg)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = func(x_con)
                    track(x_con, f_con)
                    n_used += 1
                    if f_con < fvals[-1]:
                        simplex[-1] = x_con
                        fvals[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            fvals[i] = func(simplex[i])
                            track(simplex[i], fvals[i])
                            n_used += 1
                            if n_used > n_evals or time.time() > deadline:
                                return
                        if n_used > n_evals or time.time() > deadline:
                            return

                if n_used >= n_evals:
                    break

        # Phase 1: Multi-start Nelder-Mead with dense grid of diverse starting points
        n_starts = 25
        evals_per_start = 20

        # Generate diverse starting points
        start_points = []
        # Center
        start_points.append(np.zeros(dim))
        # Uniform random
        for _ in range(n_starts - 1):
            start_points.append(rng.uniform(-5, 5, dim))

        for sp in start_points:
            if time.time() > deadline:
                break
            nelder_mead(sp, evals_per_start, sigma_scale=1.0)

        # Phase 2: Refine the best point with a tighter Nelder-Mead
        if x_best is not None and time.time() < deadline:
            nelder_mead(x_best, 50, sigma_scale=0.1)

        # Phase 3: (1+1)-ES polish
        if x_best is not None and time.time() < deadline:
            x = x_best.copy()
            fx = fx_best
            sigma = 0.5
            remaining = budget - int((time.time() - start_time) * 10)  # rough
            remaining = max(0, min(remaining, 100))

            for _ in range(remaining):
                if time.time() > deadline:
                    break
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                track(y, fy)
                if fy < fx:
                    x = y
                    fx = fy
                    sigma = min(sigma * 1.1, 2.0)
                else:
                    sigma *= 0.85
                    if sigma < 1e-5:
                        break

        return x_best, fx_best
