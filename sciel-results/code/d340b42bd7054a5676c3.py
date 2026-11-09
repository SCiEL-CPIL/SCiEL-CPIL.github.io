"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex with deadline guard and fallback refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 280s safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        calls = 0

        def safe_func(x):
            nonlocal calls
            if time.time() > deadline:
                raise TimeoutError()
            calls += 1
            return func(np.asarray(x, dtype=np.float64))

        # --- Phase 1: Nelder-Mead Simplex ---
        # Initialize simplex: one point at origin-ish, others along axes
        x0 = np.zeros(dim)
        f0 = safe_func(x0)
        simplex = [x0.copy()]
        fvals = [f0]

        # Expand along each axis
        step = 2.0
        for i in range(dim):
            xi = x0.copy()
            xi[i] += step
            simplex.append(xi)
            fvals.append(safe_func(xi))

        # Nelder-Mead iterations
        alpha = 1.0    # reflection
        gamma = 2.0    # expansion
        rho = 0.5      # contraction
        sigma_nm = 0.5 # shrink

        for _ in range(budget - (dim + 1)):
            if time.time() > deadline:
                break
            if calls >= budget:
                break

            # Sort simplex by function value
            idx = np.argsort(fvals)
            simplex = [simplex[i] for i in idx]
            fvals = [fvals[i] for i in idx]

            # Check convergence
            if abs(fvals[-1] - fvals[0]) < 1e-10 * max(1.0, abs(fvals[0])):
                break

            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]
            f_worst = fvals[-1]

            # Reflection
            x_r = centroid + alpha * (centroid - worst)
            x_r = np.clip(x_r, -5, 5)
            f_r = safe_func(x_r)

            if f_r < fvals[0]:
                # Expansion
                x_e = centroid + gamma * (centroid - worst)
                x_e = np.clip(x_e, -5, 5)
                f_e = safe_func(x_e)
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
                if f_r < f_worst:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    # Inside contraction
                    x_c = centroid - rho * (centroid - worst)
                x_c = np.clip(x_c, -5, 5)
                f_c = safe_func(x_c)
                if f_c < min(f_r, f_worst):
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = safe_func(simplex[i])

        # Best point so far
        idx_best = np.argmin(fvals)
        x_best = np.array(simplex[idx_best])
        f_best = fvals[idx_best]

        # --- Phase 2: Refine with local search on remaining budget ---
        remaining = budget - calls
        if remaining > 0 and time.time() < deadline:
            # Use CMA-like or (1+1)-ES for fine-tuning
            sigma = 0.1
            for _ in range(remaining):
                if time.time() > deadline or calls >= budget:
                    break
                direction = rng.standard_normal(dim)
                direction /= (np.linalg.norm(direction) + 1e-15)
                x_trial = x_best + sigma * direction
                x_trial = np.clip(x_trial, -5, 5)
                f_trial = safe_func(x_trial)
                if f_trial < f_best:
                    x_best = x_trial
                    f_best = f_trial
                    sigma *= 1.3
                else:
                    sigma *= 0.8
                    sigma = max(sigma, 1e-6)

        return x_best, f_best
# EVOLVE-BLOCK-END
