import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        n = self.dim
        budget = self.budget
        time_limit = 290.0

        # Parameters for CMA-ES
        lambda_ = int(4 + 3 * np.log(n))
        if lambda_ < 4:
            lambda_ = 4
        mu = max(1, lambda_ // 2)

        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Step size adaptation
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma

        # Covariance matrix adaptation
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_cov = (2 + 2 * mu_eff / n) / (n + 2 + 2 * mu_eff / n)

        best_x = None
        best_f = float('inf')
        calls = 0

        # Coarse Exploration: 20 random calls to quickly locate the optimum region
        # This is critical for AOCC - early progress matters a lot
        coarse_budget = 20
        if budget <= 50:
            coarse_budget = budget // 5

        np.random.seed(42)
        for _ in range(coarse_budget):
            if time.time() - start_time > time_limit or calls >= budget:
                break
            x = np.random.uniform(-5, 5, n)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls += 1

        # Initialize CMA-ES state
        mean = best_x.copy()
        sigma = 2.0
        C = np.eye(n)
        C_sqrt = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)

        def matrix_sqrt(mat):
            eigvals, eigvecs = np.linalg.eigh(mat)
            eigvals = np.maximum(eigvals, 1e-20)
            return eigvecs @ (np.sqrt(eigvals)[:, None] * eigvecs.T)

        # CMA-ES main loop - no restarts, let it converge naturally
        while calls < budget:
            if time.time() - start_time > time_limit:
                break

            # Generate population
            pop = np.array([mean + sigma * (C_sqrt @ np.random.randn(n)) for _ in range(lambda_)])
            pop = np.clip(pop, -5, 5)

            # Evaluate population
            f_pop = []
            for x in pop:
                if calls >= budget:
                    break
                fx = func(x)
                calls += 1
                f_pop.append(fx)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

            if calls >= budget or len(f_pop) < lambda_:
                break

            f_pop = np.array(f_pop)
            order = np.argsort(f_pop)
            pop = pop[order]
            f_pop = f_pop[order]

            # Compute new mean
            new_mean = np.sum(weights[:, None] * pop[:mu], axis=0)
            diff = new_mean - mean

            # Convergence check
            if np.linalg.norm(diff) / sigma < 1e-12:
                break

            # Update p_sigma
            norm_C_sqrt = np.sqrt(np.trace(C))
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt.T @ diff) / (sigma * norm_C_sqrt)

            chi_n = np.sqrt(n) * (1 + 1 / (4 * n) + 1 / (20 * n ** 2))
            h_sigma = 1 if np.linalg.norm(p_sigma) / chi_n < (1.4 + 1 / n) else 0

            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * diff / sigma

            # Update covariance matrix
            delta_h = (1 - h_sigma) * c_c * (1 - c_c)
            C = (1 + c_cov * (mu_eff - 2 + delta_h) - c_c) * C

            try:
                C_inv_sqrt = np.linalg.inv(C_sqrt)
            except np.linalg.LinAlgError:
                C_inv_sqrt = np.eye(n)

            for i in range(mu):
                y = C_inv_sqrt @ (pop[i] - mean) / sigma
                C += c_cov * np.outer(y, y)

            C += c_cov * delta_h * np.outer(p_c, p_c)

            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            C_sqrt = matrix_sqrt(C)

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 10)

            # Update mean
            mean = np.clip(new_mean, -5, 5)

            if sigma < 1e-10:
                break

        return best_x, best_f
