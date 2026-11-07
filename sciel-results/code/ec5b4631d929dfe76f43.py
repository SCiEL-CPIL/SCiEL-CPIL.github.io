"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize as scipy_minimize


class Optimizer:
    """CMA-ES based optimizer with multi-start and L-BFGS-B polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        global_x = None
        global_fx = np.inf
        calls_used = 0

        def evaluate(x):
            nonlocal calls_used, global_x, global_fx
            if calls_used >= budget:
                return np.inf
            calls_used += 1
            fx = func(x)
            if fx < global_fx:
                global_fx = fx
                global_x = x.copy()
            return fx

        def cma_es(x0, sigma0, remaining_budget):
            """Run CMA-ES from x0 with initial sigma. Returns best found."""
            nonlocal calls_used, global_x, global_fx
            if calls_used >= budget or not time_ok():
                return global_x, global_fx

            # CMA-ES parameters
            n = dim
            lam = max(4, int(np.ceil(4 + 3 * np.log(n))))  # population size
            mu = lam // 2  # number of parents
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

            x = np.array(x0, dtype=float)
            sigma = sigma0
            B = np.eye(n)
            D = np.ones(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

            fx = evaluate(x)
            best_x = x.copy()
            best_fx = fx

            max_iter = remaining_budget // lam
            for it in range(max_iter):
                if not time_ok() or calls_used >= budget:
                    break

                # Generate population
                offspring = []
                for i in range(lam):
                    z = np.random.standard_normal(n)
                    y = np.clip(x + sigma * (B @ (D * z)), -5, 5)
                    fy = evaluate(y)
                    offspring.append((fy, y))

                # Sort by fitness
                offspring.sort(key=lambda t: t[0])

                # Update best
                if offspring[0][0] < best_fx:
                    best_fx = offspring[0][0]
                    best_x = offspring[0][1].copy()

                # Compute weighted mean of best mu
                y_w = np.zeros(n)
                for i in range(mu):
                    y_w += weights[i] * offspring[i][1]
                x_old = x.copy()
                x = y_w

                # Update evolution paths
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (B @ (D * (y_w - x_old)))
                if np.linalg.norm(ps) / chi_n < (1.4 + 1 / c_sigma) * c_sigma:
                    pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (y_w - x_old) / sigma

                # Update covariance matrix
                delta_h = (1 + np.sqrt(c_sigma) * (1 - c_sigma) / c_sigma) * (np.linalg.norm(ps) / chi_n < (1.4 + 1 / c_sigma)) - 1
                diff = np.zeros((n, n))
                for i in range(mu):
                    diff += weights[i] * np.outer(offspring[i][1] - x_old, offspring[i][1] - x_old)
                C = (1 - c_1 - c_mu + delta_h * c_1 * c_c * (2 - c_c)) * np.eye(n) + c_1 * (pc @ pc + np.eye(n) * (1 - c_c) - np.eye(n) * (1 - c_c)) + c_mu * diff

                # Eigen decomposition for B and D
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-20, None)
                D = np.sqrt(eigvals)
                B = eigvecs

                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-12, 50.0)

                # Termination criteria
                if sigma < 1e-12:
                    break
                spread = np.linalg.norm(x - x_old) / sigma if sigma > 0 else 0
                if spread < 1e-12 and it > 5:
                    break
                # Check if we're stuck
                if it > 0 and best_fx > global_fx - 1e-10 and it % 20 == 0:
                    # Check progress over last 20 iterations
                    pass

            return best_x, best_fx

        # Phase 1: Multi-start with CMA-ES
        # First, do a few random evaluations to get a rough sense of the landscape
        rng = np.random.default_rng(42)
        n_initial = min(10, budget // 5)
        for i in range(n_initial):
            if not time_ok() or calls_used >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            evaluate(x)

        # Run CMA-ES from the best point found
        if global_x is not None:
            remaining = budget - calls_used
            if remaining > 20:
                sigma0 = 1.5
                _, _ = cma_es(global_x, sigma0, remaining)

        # Phase 2: More CMA-ES restarts from perturbed best
        n_restarts = 5
        for r in range(n_restarts):
            if not time_ok() or calls_used >= budget:
                break
            if global_x is None:
                break
            remaining = budget - calls_used
            if remaining < 30:
                break

            # Perturb global best
            noise_scale = 1.0 + 0.5 * r
            x0 = np.clip(global_x + noise_scale * rng.standard_normal(dim), -5, 5)
            sigma0 = 0.8 * (1 + 0.3 * r)
            _, _ = cma_es(x0, sigma0, remaining)

        # Phase 3: Final polish with L-BFGS-B
        if global_x is not None and calls_used < budget:
            polish_budget = budget - calls_used
            if polish_budget > 5:
                try:
                    def polished_func(z):
                        nonlocal calls_used
                        if calls_used >= budget:
                            return func(z)
                        calls_used += 1
                        return func(z)

                    res = scipy_minimize(polished_func, global_x, method='L-BFGS-B',
                                         bounds=[(-5, 5)] * dim,
                                         options={'maxiter': min(polish_budget, 50),
                                                  'maxfun': min(polish_budget, 50),
                                                  'gtol': 1e-8,
                                                  'ftol': 1e-12})
                    if res.fun < global_fx:
                        global_fx = res.fun
                        global_x = res.x.copy()
                except Exception:
                    pass

        # Phase 4: If we have budget left, do one more round of fine-tuning
        if global_x is not None and calls_used < budget and time_ok():
            remaining = budget - calls_used
            if remaining > 20:
                # Very small sigma CMA-ES for final refinement
                try:
                    x = global_x.copy()
                    sigma = 0.1
                    n = dim
                    lam = max(4, int(np.ceil(4 + 3 * np.log(n))))
                    mu = lam // 2
                    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                    weights /= weights.sum()
                    mu_eff = 1.0 / np.sum(weights ** 2)
                    c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
                    d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
                    c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
                    c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
                    c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
                    B = np.eye(n)
                    D = np.ones(n)
                    pc = np.zeros(n)
                    ps = np.zeros(n)
                    chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

                    fx = evaluate(x)
                    best_x = x.copy()
                    best_fx = fx

                    max_iter = remaining // lam
                    for it in range(max_iter):
                        if not time_ok() or calls_used >= budget:
                            break
                        offspring = []
                        for i in range(lam):
                            z = np.random.standard_normal(n)
                            y = np.clip(x + sigma * (B @ (D * z)), -5, 5)
                            fy = evaluate(y)
                            offspring.append((fy, y))
                        offspring.sort(key=lambda t: t[0])
                        if offspring[0][0] < best_fx:
                            best_fx = offspring[0][0]
                            best_x = offspring[0][1].copy()
                        y_w = np.zeros(n)
                        for i in range(mu):
                            y_w += weights[i] * offspring[i][1]
                        x_old = x.copy()
                        x = y_w
                        ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (B @ (D * (y_w - x_old)))
                        if np.linalg.norm(ps) / chi_n < (1.4 + 1 / c_sigma) * c_sigma:
                            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (y_w - x_old) / sigma
                        delta_h = (1 + np.sqrt(c_sigma) * (1 - c_sigma) / c_sigma) * (np.linalg.norm(ps) / chi_n < (1.4 + 1 / c_sigma)) - 1
                        diff = np.zeros((n, n))
                        for i in range(mu):
                            diff += weights[i] * np.outer(offspring[i][1] - x_old, offspring[i][1] - x_old)
                        C = (1 - c_1 - c_mu + delta_h * c_1 * c_c * (2 - c_c)) * np.eye(n) + c_1 * (pc @ pc + np.eye(n) * (1 - c_c) - np.eye(n) * (1 - c_c)) + c_mu * diff
                        eigvals, eigvecs = np.linalg.eigh(C)
                        eigvals = np.clip(eigvals, 1e-20, None)
                        D = np.sqrt(eigvals)
                        B = eigvecs
                        sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                        sigma = np.clip(sigma, 1e-12, 50.0)
                        if sigma < 1e-12:
                            break

                    if best_fx < global_fx:
                        global_fx = best_fx
                        global_x = best_x.copy()
                except Exception:
                    pass

        return global_x, global_fx
# EVOLVE-BLOCK-END
