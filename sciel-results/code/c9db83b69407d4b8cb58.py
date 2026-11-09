# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts for BBOB-style optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        rng = np.random.default_rng(42)

        # Global best tracking
        best_x = np.zeros(dim)
        best_f = func(best_x)
        calls = 1

        # Try a few diverse initial points to get a good starting region
        init_points = [
            np.ones(dim) * 2.0,
            np.ones(dim) * -2.0,
            rng.uniform(-4, 4, dim),
            rng.uniform(-4, 4, dim),
        ]
        for pt in init_points:
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            pt = np.clip(pt, -5, 5)
            f = func(pt)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = pt.copy()

        # CMA-ES implementation
        def cma_es(x0, sigma0, max_iter, rng):
            nonlocal calls, best_f, best_x

            dim = self.dim
            # CMA-ES parameters
            lambda_ = max(4, int(4 + 3 * np.log(dim)))  # population size
            c_sigma = (lambda_ + 2) / ((dim + 1) ** 2 + lambda_)
            d_sigma = 1 + 2 * max(0, np.sqrt((lambda_ - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + lambda_ / dim) / (dim + 4 + 2 * lambda_ / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + lambda_)
            c_mu = min(1 - c_1, 2 * ((lambda_ / 2 + 1) / (dim + 2) - 1))

            # State variables
            x_mean = x0.copy()
            sigma = sigma0
            B = np.eye(dim)
            D = np.ones(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            # Weights for adaptation
            mu = lambda_ // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1 / (weights ** 2).sum()
            c_mu = min(1 - c_1, 2 * ((mu_eff / 2 + 1) / (dim + 2) - 1))

            for it in range(max_iter):
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break

                # Generate population
                offspring = []
                for i in range(lambda_):
                    z = rng.standard_normal(dim)
                    y = x_mean + sigma * (B @ (D * z))
                    y = np.clip(y, -5, 5)
                    offspring.append(y)

                # Evaluate offspring
                f_values = []
                for y in offspring:
                    if calls >= budget:
                        break
                    if time.time() > deadline:
                        break
                    f_val = func(y)
                    calls += 1
                    f_values.append(f_val)
                    if f_val < best_f:
                        best_f = f_val
                        best_x = y.copy()

                if len(f_values) < lambda_:
                    break

                # Sort by fitness
                idx = np.argsort(f_values)
                offspring = [offspring[i] for i in idx]
                f_values = [f_values[i] for i in idx]

                # Update mean
                x_mean_old = x_mean.copy()
                x_mean = sum(w * offspring[i] for i, w in enumerate(weights))

                # Update step size
                if np.linalg.norm(B @ (D * (x_mean - x_mean_old))) > 0:
                    p_sigma = (1 - c_sigma) * p_sigma + \
                              np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / \
                              np.linalg.norm(B @ (D * (x_mean - x_mean_old))) * \
                              B.T @ ((x_mean - x_mean_old) / D) / D
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                    sigma = max(sigma, 1e-10)
                    sigma = min(sigma, 5.0)

                # Update covariance matrix
                y_norm = (x_mean - x_mean_old) / sigma
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_norm
                delta_h = 1 if np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c) ** (it + 1)) < \
                              (1 + 2 / (dim + 1)) * chi_n else 0
                C = (1 + c_1 * (delta_h - 1) + c_mu) * C + \
                    c_1 * (delta_h - 1 + c_c * (2 - c_c)) * np.outer(p_c, p_c) + \
                    c_mu / sigma ** 2 * sum(w * np.outer(offspring[i] - x_mean_old,
                                                          offspring[i] - x_mean_old)
                                            for i, w in enumerate(weights))

                # Re-derive B and D
                D_sq, B = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D_sq, 1e-20))
                B = B[:, np.argsort(D_sq)]

            return x_mean, sigma

        # Run CMA-ES from the best found point
        x_cma, sigma_cma = cma_es(best_x.copy(), 1.5, budget - calls, rng)

        # If budget remains, do a few more random restarts
        while calls < budget and time.time() < deadline:
            x_rand = rng.uniform(-5, 5, dim)
            x_rand = np.clip(x_rand, -5, 5)
            f_rand = func(x_rand)
            calls += 1
            if f_rand < best_f:
                best_f = f_rand
                best_x = x_rand.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
