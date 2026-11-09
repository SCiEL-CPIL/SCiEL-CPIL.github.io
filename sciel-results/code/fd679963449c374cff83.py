"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style evolutionary strategy for BBOB f8 (Rosenbrock)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0

        dim = self.dim
        budget = self.budget
        calls_used = 0
        best_x = None
        best_f = np.inf

        def call_func(x):
            nonlocal calls_used, best_x, best_f
            if calls_used >= budget:
                return np.inf
            calls_used += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # CMA-ES parameters
        pop_size = max(4, int(np.ceil(4 + 3 * np.log(dim))))  # lambda
        sigma = 1.0  # initial step size
        x_mean = np.zeros(dim)

        # Initialize population around origin with some spread
        rng = np.random.default_rng(12345)

        # Covariance matrix (start with identity)
        C = np.eye(dim)
        # Eigen decomposition for sampling
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        sqrt_eigenvalues = np.sqrt(np.maximum(eigenvalues, 1e-12))

        # Adaptation parameters
        c_sigma = (pop_size + 2) / (dim + 4)
        d_sigma = 1 + 2 * max(0, (dim - 1) / (dim + 3) - 1) + c_sigma
        c_c = (4 + dim / dim) / (4 + dim + 3)
        c_1 = 2 / ((dim + 1.3) ** 2 + pop_size)
        c_mu = min(1 - c_1, 2 * ((pop_size // 2 + 1) - dim - 2) / ((dim + 2) ** 2 + pop_size))

        # Path variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Initial population
        pop = []
        fvals = []
        for i in range(pop_size):
            z = rng.standard_normal(dim)
            x = x_mean + sigma * (eigenvectors @ (sqrt_eigenvalues * z))
            x = np.clip(x, -5, 5)
            pop.append(x)
            fvals.append(call_func(x))

        # Track best
        if best_x is None:
            idx = np.argmin(fvals)
            best_x = pop[idx].copy()
            best_f = fvals[idx]

        max_iter = 200

        for iteration in range(max_iter):
            if time.time() > deadline or calls_used >= budget:
                break

            # Sort by fitness
            order = np.argsort(fvals)
            pop = [pop[i] for i in order]
            fvals = [fvals[i] for i in order]

            # Update best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = pop[0].copy()

            # Weighted mean (mu best individuals)
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights ** 2)

            x_old = x_mean.copy()
            x_mean = np.sum([weights[i] * pop[i] for i in range(mu)], axis=0)

            # Update sigma path
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (
                eigenvectors @ (sqrt_eigenvalues / sigma) @ (x_mean - x_old))

            # Hessian drift control
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (iteration + 1)))) / \
                      (1 - c_sigma) * np.sqrt(dim + 1) / np.sqrt(dim)
            h_sigma = 1 if h_sigma < (1.4 + 2 / (dim + 1)) * np.sqrt(1 - (1 - c_sigma) ** 2) else 0

            # Update covariance matrix
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean - x_old) / sigma

            rank_one = np.outer(x_mean - x_old, x_mean - x_old)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                y_i = (pop[i] - x_old) / sigma
                rank_mu += weights[i] * np.outer(y_i, y_i)

            C = (1 + c_1 * h_sigma * (1 - h_sigma) - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * C) + c_mu * rank_mu

            # Renormalize to keep trace reasonable
            trace = np.trace(C)
            if trace > 0:
                C /= trace / dim

            # Ensure positive semi-definite
            C = (C + C.T) / 2
            eigenvalues, eigenvectors = np.linalg.eigh(C)
            eigenvalues = np.maximum(eigenvalues, 1e-12)
            sqrt_eigenvalues = np.sqrt(eigenvalues)

            # Update sigma
            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-8, 100)

            # Generate new population
            new_pop = []
            new_fvals = []
            for i in range(pop_size):
                z = rng.standard_normal(dim)
                x = x_mean + sigma * (eigenvectors @ (sqrt_eigenvalues * z))
                x = np.clip(x, -5, 5)
                new_pop.append(x)
                new_fvals.append(call_func(x))

            pop = new_pop
            fvals = new_fvals

        # Final polish with coordinate descent if budget remains
        if best_x is not None and time.time() < deadline and calls_used < budget:
            x = best_x.copy()
            fx = best_f
            step = max(sigma, 0.1)
            for _ in range(50):
                if time.time() > deadline or calls_used >= budget:
                    break
                improved = False
                for i in range(dim):
                    if time.time() > deadline or calls_used >= budget:
                        break
                    for direction in [1, -1]:
                        x_new = x.copy()
                        x_new[i] += direction * step
                        x_new = np.clip(x_new, -5, 5)
                        if np.all(x_new == x):
                            continue
                        fx_new = call_func(x_new)
                        if fx_new < fx:
                            x = x_new
                            fx = fx_new
                            improved = True
                        else:
                            step *= 0.5
                if not improved:
                    step *= 0.5
                if step < 1e-7:
                    break
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
