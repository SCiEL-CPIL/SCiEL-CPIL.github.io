import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES for BBOB f6 (Attractive Sector)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # 5s safety margin before 330s kill

        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def eval_and_update(x):
            nonlocal best_x, best_f, calls_made
            calls_made += 1
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def time_left():
            return deadline - time.time()

        # ---------- Phase 1: Coarse grid + random sampling ----------
        # Evaluate center
        eval_and_update(np.zeros(dim))

        # Coarse grid: 3 points per dimension (center, -2, +2)
        # That's 3^5 = 243 points, too many. Use a subset.
        # Instead, do a Latin-hypercube-like sampling of ~30 points.
        rng = np.random.default_rng(123)
        n_grid = 30
        for _ in range(n_grid):
            if calls_made >= budget or time_left() < 5.0:
                break
            x = rng.uniform(-5, 5, dim)
            eval_and_update(x)

        # Also evaluate a few structured points
        structured = [
            np.zeros(dim),
            np.full(dim, 2.0),
            np.full(dim, -2.0),
            np.array([5.0] + [0.0]*(dim-1)),
            np.array([-5.0] + [0.0]*(dim-1)),
        ]
        for x in structured:
            if calls_made >= budget or time_left() < 5.0:
                break
            eval_and_update(x)

        # ---------- Phase 2: CMA-ES from best point ----------
        def run_cmaes(x_start, sigma_start, max_calls, max_time):
            nonlocal best_x, best_f, calls_made
            t0 = time.time()

            # Population size
            lam = max(4, int(4 + 3 * np.log(dim)))
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights ** 2)

            cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            cs = (mu_eff + 2) / (dim + mu_eff + 5)
            c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            cmu = min(1 - c1, 2 * ((mu_eff + 2) / (mu_eff + 5) - 1)) / ((dim + 2) ** 2 + mu_eff)

            x_mean = x_start.copy()
            sigma = sigma_start
            C = np.eye(dim)
            # Cholesky factor of C: C = L @ L.T, so sample = x_mean + sigma * L @ z
            L = np.eye(dim)

            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            gen = 0
            while calls_made < budget and (time.time() - t0) < max_time and calls_made < max_calls:
                if time_left() < 3.0:
                    break

                # Generate population
                pop = []
                fits = []
                for i in range(lam):
                    if calls_made >= budget or time_left() < 3.0:
                        break
                    z = rng.standard_normal(dim)
                    # Sample: x = x_mean + sigma * L @ z
                    y = sigma * (L @ z)
                    x_i = np.clip(x_mean + y, -5, 5)
                    f_i = eval_and_update(x_i)
                    pop.append(x_i)
                    fits.append(f_i)

                if len(pop) < mu:
                    break

                # Sort
                idx = np.argsort(fits)
                pop = [pop[i] for i in idx]
                fits = [fits[i] for i in idx]

                # Weighted mean
                x_new_mean = np.sum([weights[i] * pop[i] for i in range(mu)], axis=0)

                # Delta for paths
                if sigma > 1e-15:
                    delta = (x_new_mean - x_mean) / sigma
                else:
                    delta = np.zeros(dim)

                # Update sigma path
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * delta

                # Sigma adaptation
                norm_p_sigma = np.linalg.norm(p_sigma)
                # Approximate the h_sigma condition
                h_sigma = 1.0 if norm_p_sigma / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * chi_n else 0.0
                if h_sigma > 0:
                    sigma *= np.exp(cs / (dim + 3) * (norm_p_sigma / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Update C path
                p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * mu_eff) * delta

                # h_c heuristic
                h_c = 1.0 if np.linalg.norm(p_c) / np.sqrt(1 - (1 - cc) ** (gen + 1)) < (1.4 + 2 / (dim + 1)) * chi_n else 0.0

                # Rank-one update
                rank_one = np.outer(p_c, p_c)

                # Rank-mu update
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    if sigma > 1e-15:
                        y_i = (pop[i] - x_mean) / sigma
                    else:
                        y_i = np.zeros(dim)
                    rank_mu += weights[i] * np.outer(y_i, y_i)

                # Update C
                C = (1 + c1 * h_c * (2 - cc) - c1) * C + c1 * rank_one + cmu * rank_mu
                C = (C + C.T) / 2

                # Cholesky with regularization
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Add small jitter
                    C += 1e-6 * np.eye(dim)
                    try:
                        L = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        L = np.eye(dim)

                x_mean = x_new_mean
                gen += 1

                # Stagnation check: if best in population hasn't improved in a while, break
                # (handled by caller via max_calls)

            return best_f

        # Run CMA-ES from the best point found
        x_start = best_x.copy() if best_x is not None else np.zeros(dim)
        # Estimate initial sigma from the spread of evaluated points
        # Use a moderate sigma to start
        sigma_start = 1.5
        run_cmaes(x_start, sigma_start, max_calls=budget - calls_made, max_time=time_left() - 5.0)

        # If we still have budget and time, do a few more CMA-ES restarts from random points
        # This helps if the first CMA-ES got stuck in a local optimum
        restarts = 0
        max_restarts = 3
        while (calls_made < budget - 10 and time_left() > 30.0 and restarts < max_restarts):
            # Pick a random starting point, biased toward the current best
            x_restart = np.clip(best_x + rng.standard_normal(dim) * 2.0, -5, 5)
            f_before = best_f
            run_cmaes(x_restart, 1.0, max_calls=min(100, budget - calls_made), max_time=min(30.0, time_left() - 5.0))
            if best_f < f_before - 1e-8:
                restarts += 1
            else:
                # No improvement, stop restarting
                break

        return best_x, best_f
