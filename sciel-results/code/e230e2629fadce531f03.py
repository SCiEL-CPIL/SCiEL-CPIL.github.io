"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+4)-ES with success-rate sigma adaptation and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: (1+lambda)-ES with success-rate adaptation
        lam = 4
        # Reserve budget for polish
        polish_budget = 60
        es_budget = budget - polish_budget

        best_x = None
        best_f = np.inf

        # Multiple restarts
        n_restarts = 3
        budget_per_restart = es_budget // n_restarts

        for restart in range(n_restarts):
            x = rng.uniform(-4, 4, dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 1.5
            successes = 0
            window = 10
            success_count = 0

            for i in range(budget_per_restart - 1):
                if time.time() > deadline - polish_budget * 0.5:
                    break

                # Generate lambda offspring
                offspring = []
                for _ in range(lam):
                    y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                    fy = func(y)
                    offspring.append((fy, y))

                offspring.sort(key=lambda t: t[0])
                best_fy, best_y = offspring[0]

                # Count successes (any offspring better than parent)
                n_better = sum(1 for fy, _ in offspring if fy < fx)
                success_count += 1 if n_better > 0 else 0

                if best_fy < fx:
                    x = best_y
                    fx = best_fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()

                # Success rate adaptation
                if success_count >= window:
                    sr = success_count / window
                    if sr > 0.3:
                        sigma *= 1.2
                    elif sr < 0.15:
                        sigma *= 0.8
                    success_count = 0

                sigma = max(sigma, 1e-6)
                sigma = min(sigma, 10.0)

        # Phase 2: Nelder-Mead polish
        if best_x is not None and time.time() < deadline:
            x_nm = best_x.copy()
            f_nm = best_f
            # Initial simplex
            simplex = [x_nm.copy()]
            for i in range(dim):
                pt = x_nm.copy()
                step = 0.1 if abs(x_nm[i]) > 1e-3 else 0.1
                pt[i] += step
                pt = np.clip(pt, -5, 5)
                simplex.append(pt)

            fvals = [func(s) for s in simplex]
            fvals = [f if np.isfinite(f) else 1e10 for f in fvals]

            # Update best
            for s, f in zip(simplex, fvals):
                if f < f_nm:
                    f_nm = f
                    x_nm = s.copy()
                    if f_nm < best_f:
                        best_f = f_nm
                        best_x = x_nm.copy()

            iters = 0
            max_iters = polish_budget // (dim + 1)  # rough estimate
            for _ in range(max_iters):
                if time.time() > deadline:
                    break
                iters += 1

                # Sort simplex
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                f_worst = fvals[-1]

                # Reflection
                alpha = 1.0
                x_r = centroid + alpha * (centroid - worst)
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                if not np.isfinite(f_r):
                    f_r = 1e10

                if f_r < fvals[0]:
                    # Expansion
                    gamma = 2.0
                    x_e = centroid + gamma * (centroid - worst)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    if not np.isfinite(f_e):
                        f_e = 1e10
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
                    rho = 0.5
                    x_c = centroid + rho * (worst - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    if not np.isfinite(f_c):
                        f_c = 1e10
                    if f_c < f_worst:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func(simplex[i])
                            if not np.isfinite(fvals[i]):
                                fvals[i] = 1e10

                # Update best
                for s, f in zip(simplex, fvals):
                    if f < f_nm:
                        f_nm = f
                        x_nm = s.copy()
                        if f_nm < best_f:
                            best_f = f_nm
                            best_x = x_nm.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
