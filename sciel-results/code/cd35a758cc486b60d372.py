import numpy as np
import time

class Optimizer:
    """
    Optimizer using CMA-ES with adaptive restarts for BBOB f13 (Sharp Ridge).
    CMA-ES adapts its covariance matrix to the problem's geometry, which is
    particularly effective for sharp ridge functions where the search space
    has a narrow valley structure.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0  # Safety margin for 330s limit

        calls = 0
        best_x = None
        best_f = float('inf')
        rng = np.random.default_rng(42)

        def safe_call(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return None
            if time.time() > deadline:
                return None
            calls += 1
            try:
                x_clipped = np.clip(x, -5.0, 5.0)
                f_val = func(x_clipped)
                if f_val < best_f:
                    best_f = f_val
                    best_x = x_clipped.copy()
                return f_val
            except Exception:
                return None

        # Phase 1: Global Exploration
        # Sample points to find the basin of attraction.
        n_random = 15
        for _ in range(n_random):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, size=dim)
            f_val = safe_call(x)
            if f_val is None:
                break

        if best_x is None:
            # Fallback to origin if no valid call was made
            x0 = np.zeros(dim)
            f0 = safe_call(x0)
            if f0 is None:
                return best_x, best_f
            best_x = x0
            best_f = f0

        # Phase 2: CMA-ES with restarts
        def cma_es(x_start, f_start, max_evals=200):
            nonlocal calls, best_f, best_x

            if calls >= budget or time.time() > deadline:
                return

            # CMA-ES parameters
            lambda_ = 4 * dim  # Population size
            mu = lambda_ // 2  # Weighting
            cc = (4. + dim / 4.) / (dim + 4. + 2.)
            cs = (dim + 1.5 + mu_eff) / (dim + 5. + 2. * mu_eff)
            mu_eff = 1. / sum((i + 1) for i in range(mu))
            sigma = 0.5  # Initial step size
            c_sigma = cs
            c_cov = (4. + mu_eff / dim) / (dim + 4. + 2. * mu_eff / dim)
            c_c = 2. / (dim + 3.) ** 2 + c_cov

            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()

            # State
            mean = x_start.copy()
            C = np.eye(dim)
            C_sqrt = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1. + 1. / (4. * dim) - 1. / (21. * dim ** 2))

            evals_this_run = 0
            stagnation = 0
            max_stagnation = 30

            while evals_this_run < max_evals and calls < budget and time.time() < deadline:
                # Generate population
                population = np.zeros((lambda_, dim))
                f_values = np.zeros(lambda_)

                for i in range(lambda_):
                    if calls >= budget or time.time() > deadline:
                        break
                    z = rng.standard_normal(dim)
                    x_i = mean + sigma * (C_sqrt @ z)
                    x_i = np.clip(x_i, -5.0, 5.0)
                    f_i = safe_call(x_i)
                    if f_i is None:
                        return
                    population[i] = x_i
                    f_values[i] = f_i
                    evals_this_run += 1

                if calls >= budget or time.time() > deadline:
                    break

                # Sort by function value
                idx = np.argsort(f_values)
                population = population[idx]
                f_values = f_values[idx]

                # Update mean
                new_mean = np.sum(weights[:, np.newaxis] * population[:mu], axis=0)

                # Update evolution paths
                y = (new_mean - mean) / sigma
                ps = (1. - c_sigma) * ps + np.sqrt(c_sigma * (2. - c_sigma) * mu_eff) * y
                pc = (1. - c_c) * pc + np.sqrt(c_c * (2. - c_c) * mu_eff) * (C_sqrt.T @ y)

                # Update covariance matrix
                delta_hs = 1. if (np.dot(ps, ps) / (1. - (1. - c_sigma) ** (2 * (evals_this_run + 1)))) < (1. + 4. / dim) * chi_n ** 2 else 0.
                rank_mu = np.outer(y, y)
                rank_one = np.outer(ps, ps)
                C = (1. + c_cov * (c_c - c_cov + delta_hs * (cs * (2. - c_sigma))) * (rank_one - C)) + c_cov * (c_c - c_cov) * rank_mu

                # Ensure C is positive definite
                C = (C + C.T) / 2.
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = C + 1e-10 * np.eye(dim)
                    try:
                        C_sqrt = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        C_sqrt = np.eye(dim)

                # Update step size
                sigma *= np.exp((c_sigma / cs) * (np.linalg.norm(ps) / chi_n - 1.))
                sigma = max(sigma, 1e-10)
                sigma = min(sigma, 10.0)

                mean = new_mean

                # Check convergence
                if f_values[0] < best_f:
                    best_f = f_values[0]
                    best_x = population[0].copy()
                    stagnation = 0
                else:
                    stagnation += 1

                if stagnation > max_stagnation:
                    break

                if np.max(f_values[:mu]) - np.min(f_values[:mu]) < 1e-12:
                    break

        # Run CMA-ES with restarts
        restart_count = 0
        max_restarts = 8

        while calls < budget and time.time() < deadline and restart_count < max_restarts:
            if restart_count == 0:
                x_start = best_x.copy()
                f_start = best_f
            else:
                # Perturb the best point found so far
                scale = 1.0 * (1.0 + 0.5 * restart_count)
                x_start = best_x + rng.normal(0, scale, dim)
                x_start = np.clip(x_start, -5.0, 5.0)
                f_start = safe_call(x_start)
                if f_start is None:
                    break

            # Run CMA-ES for a limited number of evaluations to allow restarts
            remaining = budget - calls
            if remaining < 20:
                break

            max_evals = min(200, remaining - 10)
            cma_es(x_start, f_start, max_evals=max_evals)

            restart_count += 1

        return best_x, best_f
