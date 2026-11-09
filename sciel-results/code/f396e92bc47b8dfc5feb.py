"""Nelder-Mead + (1+1)-ES hybrid optimizer for BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex + (1+1)-ES hybrid for ill-conditioned rotated landscapes."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 310  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        def time_ok():
            return time.time() < deadline

        # --- Phase 1: Coarse random sampling to find a good starting point ---
        n_init = 20
        best_x = None
        best_f = np.inf
        evals = 0

        for _ in range(n_init):
            if evals >= budget or not time_ok():
                break
            x = rng.uniform(-4, 4, dim)
            fx = func(x)
            evals += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            evals += 1

        # --- Phase 2: Nelder-Mead simplex method ---
        # Initialize simplex around best_x
        x0 = best_x.copy()
        alpha = 0.5  # initial simplex size
        simplex = [x0.copy()]
        fvals = [best_f]
        # Generate other vertices
        for i in range(dim):
            xi = x0.copy()
            xi[i] += alpha
            simplex.append(xi.copy())
            fi = func(xi)
            evals += 1
            fvals.append(fi)

        # Nelder-Mead parameters
        alpha_nm = 1.0   # reflection
        gamma = 2.0      # expansion
        rho = 0.5        # contraction
        sigma_nm = 0.5   # shrink
        max_nm_iter = 150  # limit iterations to leave budget for ES phase

        nm_iter = 0
        while nm_iter < max_nm_iter and evals < budget - 50 and time_ok():
            nm_iter += 1
            n = len(simplex)

            # Sort simplex by fitness
            idx = np.argsort(fvals)
            simplex = [simplex[i].copy() for i in idx]
            fvals = [fvals[i] for i in idx]

            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]
            f_worst = fvals[-1]

            # Reflection
            x_r = centroid + alpha_nm * (centroid - worst)
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            evals += 1

            if f_r < fvals[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                evals += 1
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                # Simple reflection
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                # Contraction
                if f_r < f_worst:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    # Inside contraction
                    x_c = centroid - rho * (worst - centroid)
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                evals += 1
                if f_c < min(f_r, f_worst):
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, n):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        evals += 1

            # Update best
            min_idx = int(np.argmin(fvals))
            if fvals[min_idx] < best_f:
                best_f = fvals[min_idx]
                best_x = simplex[min_idx].copy()

        # --- Phase 3: (1+1)-ES with 1/5th success rule for fine-tuning ---
        x = best_x.copy()
        fx = best_f
        # Estimate initial sigma from simplex spread
        if len(simplex) > 1:
            dists = [np.linalg.norm(s - best_x) for s in simplex]
            sigma = max(0.01, np.mean(dists) * 0.3)
        else:
            sigma = 0.1

        # 1/5th success rule
        successes = []
        window = 10
        c = 0.9  # adaptation rate

        while evals < budget and time_ok():
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            evals += 1

            if fy < fx:
                x = y
                fx = fy
                successes.append(1)
            else:
                successes.append(0)

            if len(successes) > window:
                successes.pop(0)

            if len(successes) == window:
                rate = np.mean(successes)
                if rate > 0.2:
                    sigma *= (1.0 + c * 0.1)
                elif rate < 0.1:
                    sigma *= (1.0 - c * 0.1)

            sigma = np.clip(sigma, 0.001, 10.0)

        return x, fx
# EVOLVE-BLOCK-END
