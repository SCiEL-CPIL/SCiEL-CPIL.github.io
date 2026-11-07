"""CMA-ES with deadline guard, initial sampling, and restarts for BBOB f13."""
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s wall-clock kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Track best
        best_x = None
        best_f = np.inf
        call_count = 0

        def time_ok():
            return time.time() < deadline and call_count < budget

        # --- Phase 1: Initial coarse sampling to find a good region ---
        n_init = min(30, budget // 5)
        for i in range(n_init):
            if not time_ok():
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Also try center
        x0 = np.zeros(dim)
        fx0 = func(x0)
        call_count += 1
        if fx0 < best_f:
            best_f = fx0
            best_x = x0.copy()

        # --- Phase 2: CMA-ES with restarts ---
        def cmaes_run(x_start, f_start, max_evals, sigma_init=1.0):
            nonlocal call_count, best_f, best_x
            if call_count >= budget or not time_ok():
                return
            dim_local = dim
            popsize = max(4, 4 + int(3 * np.log(dim_local)))
            # Adapt popsize to remaining budget
            remaining = budget - call_count
            if remaining < popsize * 3:
                popsize = max(2, remaining // 3)
            if popsize < 2:
                popsize = 2

            # CMA-ES parameters
            lambda_ = popsize
            mu = lambda_ // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mu_eff + 2) / (dim_local + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim_local + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / dim_local) / (dim_local + 4 + 2 * mu_eff / dim_local)
            c_1 = (2 - mu_eff / (dim_local + 1.7)) / ((dim_local + 2) ** 2 + mu_eff / (dim_local + 1.7))
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim_local + 2) ** 2 + mu_eff))
            chi_N = np.sqrt(dim_local) * (1 - 1 / (4 * dim_local) + 1 / (21 * dim_local ** 2))

            x_mean = x_start.copy()
            sigma = sigma_init
            C = np.eye(dim_local)
            pc = np.zeros(dim_local)
            ps = np.zeros(dim_local)

            for gen in range(max_evals):
                if not time_ok():
                    break
                # Generate population
                offsprings = []
                for i in range(lambda_):
                    if not time_ok():
                        break
                    z = rng.standard_normal(dim_local)
                    # Approximate sqrt(C) via Cholesky (recompute each gen for simplicity)
                    try:
                        L = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        L = np.eye(dim_local)
                    y = sigma * (L @ z)
                    xi = np.clip(x_mean + y, -5, 5)
                    fi = func(xi)
                    call_count += 1
                    offsprings.append((fi, xi))
                    if fi < best_f:
                        best_f = fi
                        best_x = xi.copy()

                if len(offsprings) < mu:
                    break

                # Sort by fitness
                offsprings.sort(key=lambda t: t[0])
                selected = offsprings[:mu]

                # Update mean
                new_mean = sum(weights[i] * selected[i][1] for i in range(mu))
                new_mean = np.clip(new_mean, -5, 5)

                # Update sigma (CSA)
                y_mean = (new_mean - x_mean) / sigma if sigma > 1e-12 else np.zeros(dim_local)
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (gen + 1))) / chi_N < 1.4 + 2 / (dim_local + 1)
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_N - 1))
                sigma = max(sigma, 1e-12)

                # Update C
                if hsig:
                    pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean
                delta_hsig = (1 - hsig) * c_c * (2 - c_c)
                rank_1 = np.outer(pc, pc)
                rank_mu = np.zeros((dim_local, dim_local))
                for i in range(mu):
                    zi = (selected[i][1] - x_mean) / sigma if sigma > 1e-12 else np.zeros(dim_local)
                    rank_mu += weights[i] * np.outer(zi, zi)
                C = (1 + c_1 * delta_hsig - c_1 - c_mu) * C + c_1 * rank_1 + c_mu * rank_mu

                x_mean = new_mean

            return

        # Run CMA-ES with initial sigma based on spread
        sigma_init = 2.0
        max_evals_1 = min(budget - call_count, 300)
        cmaes_run(best_x, best_f, max_evals_1, sigma_init)

        # If budget remains, do a second CMA-ES run with smaller sigma (local refinement)
        if time_ok() and call_count < budget:
            sigma_init2 = 0.5
            max_evals_2 = min(budget - call_count, 150)
            cmaes_run(best_x, best_f, max_evals_2, sigma_init2)

        # If still budget remains, do a fine local search with Nelder-Mead
        if time_ok() and call_count < budget:
            remaining = budget - call_count
            if remaining >= 10:
                def wrapper(x):
                    nonlocal call_count
                    if call_count >= budget:
                        return best_f
                    x = np.clip(x, -5, 5)
                    fx = func(x)
                    call_count += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    return fx

                try:
                    res = minimize(wrapper, best_x, method='Nelder-Mead',
                                   options={'maxiter': min(remaining // 2, 50),
                                            'xatol': 1e-6, 'fatol': 1e-8})
                except Exception:
                    pass

        return best_x, best_f
