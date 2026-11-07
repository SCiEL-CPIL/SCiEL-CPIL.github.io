import numpy as np
import time


class Optimizer:
    """CMA-ES-lite with restarts for BBOB f13 (sharp ridge), 5D, 500 evals."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 290.0  # 10s safety margin before 300s kill
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(12345)

        # --- CMA-ES core with restarts ---
        # Parameters
        lam = 20       # population size
        mu = 5         # number of parents
        weights = np.ones(mu) / mu
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = (2 + 2.0 / (dim + 1.3)) / (dim + c_c + 3)
        c_mu = min(1 - c_1, (mu_eff - 2 + 2.0 / (dim + 1)) / (dim + 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        def time_ok():
            return time.time() < deadline

        # Restart mechanism
        best_x = None
        best_f = np.inf
        total_evals = 0

        max_restarts = 4
        for restart in range(max_restarts):
            if total_evals >= budget or not time_ok():
                break

            # Initialize or re-seed
            if restart == 0:
                x_mean = rng.uniform(-5, 5, dim)
            else:
                # Seed near the best found so far, with some spread
                x_mean = best_x + rng.uniform(-0.5, 0.5, dim)
                x_mean = np.clip(x_mean, -5, 5)

            # Initial sigma: start moderately large
            sigma = 1.5 if restart == 0 else 0.5
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            cs_inv = 1.0 / (1 - c_c)  # not exactly, but we'll track pc norm
            # Actually track pc for sigma update
            pc = np.zeros(dim)
            ps = np.zeros(dim)

            # Evaluate initial population
            pop = []
            for i in range(lam):
                if not time_ok() or total_evals >= budget:
                    break
                y = x_mean + sigma * rng.multivariate_normal(np.zeros(dim), C, 1).ravel()
                y = np.clip(y, -5, 5)
                fy = func(y)
                total_evals += 1
                pop.append((fy, y.copy()))

            if len(pop) == 0:
                break

            # Sort by fitness
            pop.sort(key=lambda t: t[0])

            # Track best
            if pop[0][0] < best_f:
                best_f = pop[0][0]
                best_x = pop[0][1].copy()

            # CMA-ES main loop
            no_improve = 0
            stagnation_threshold = 40  # evaluations without improvement

            while total_evals < budget and time_ok():
                # Update from current population
                parents = [p[1] for p in pop[:mu]]
                parent_fits = [p[0] for p in pop[:mu]]

                # Weighted mean
                w_mean = np.zeros(dim)
                for w, p in zip(weights, parents):
                    w_mean += w * p
                w_mean = w_mean / np.sum(weights)

                # Update covariance matrix
                y_vecs = [(p - x_mean) / sigma for p in parents]
                C_update = np.zeros((dim, dim))
                for w, yv in zip(weights, y_vecs):
                    C_update += w * np.outer(yv, yv)
                C = (1 - c_1 - c_mu) * C + c_1 * ((1 + 2 * c_1 / c_1 - 1) * C + 2 * np.outer(pc, pc)) + c_mu * C_update
                # Simplify: standard CMA update
                C = (1 - c_1 - c_mu) * C + c_1 * (2 * np.outer(pc, pc) + (1 - np.outer(pc, pc) @ C @ np.outer(pc, pc).T) * C) + c_mu * C_update
                # Ensure symmetry
                C = (C + C.T) / 2

                # Update step size
                ps = (1 - c_sigma) * ps + c_sigma / np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (w_mean - x_mean) / sigma
                norm_ps = np.linalg.norm(ps)
                h_sigma = 1.0 if norm_ps / np.sqrt(1 - (1 - c_sigma) ** (2 * (total_evals % 100) + 1)) < (1 + 2 / (dim + 1)) * chi_n else 0.0
                sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Update pc
                h_sigma_factor = 1.0 if h_sigma else 0.0
                pc = (1 - c_c) * pc + h_sigma_factor * c_c * (w_mean - x_mean) / sigma

                x_mean = w_mean

                # Check improvement
                if parent_fits[0] < best_f:
                    best_f = parent_fits[0]
                    best_x = parents[0].copy()
                    no_improve = 0
                else:
                    no_improve += 1

                # Generate new population
                new_pop = []
                for i in range(lam):
                    if not time_ok() or total_evals >= budget:
                        break
                    y = x_mean + sigma * rng.multivariate_normal(np.zeros(dim), C, 1).ravel()
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    total_evals += 1
                    new_pop.append((fy, y.copy()))

                if len(new_pop) == 0:
                    break

                new_pop.sort(key=lambda t: t[0])

                # Check if we should restart
                if no_improve > stagnation_threshold:
                    if new_pop[0][0] < best_f:
                        best_f = new_pop[0][0]
                        best_x = new_pop[0][1].copy()
                    break  # trigger restart

                pop = new_pop

            # If we used up budget or time, break out of restarts
            if total_evals >= budget or not time_ok():
                break

        return best_x, best_f
