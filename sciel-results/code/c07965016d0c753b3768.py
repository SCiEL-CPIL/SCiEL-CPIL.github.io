"""CMA-ES based optimizer for BBOB f6."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB f6 (Attractive sector)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Initial random sampling to find a good starting point
        n_init = min(30, budget // 10)
        X_init = rng.uniform(-5, 5, (n_init, dim))
        F_init = np.array([func(x) for x in X_init])

        best_idx = np.argmin(F_init)
        x_best = X_init[best_idx].copy()
        f_best = F_init[best_idx]

        # Phase 2: CMA-ES from the best initial point
        # CMA-ES parameters
        sigma = 2.0  # initial step size
        pop_size = 4 + int(3 * np.log(dim))  # lambda = 4 + 3*ln(dim)
        pop_size = max(pop_size, 8)

        # CMA-ES state
        mean = x_best.copy()
        C = np.eye(dim)  # covariance matrix
        C_sqrt = np.eye(dim)  # square root of C

        # Evolution path for covariance adaptation
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Parameters
        c_sigma = (pop_size + 2) / (dim + 4)
        mu = pop_size // 2  # number of best points used for recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2)**2 + mu_eff))

        # Hessian update parameters
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

        evaluations_used = n_init
        step_count = 0
        stagnation_counter = 0
        best_f_cma = f_best

        while evaluations_used < budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate population
            Y = rng.standard_normal((pop_size, dim))
            X = mean + sigma * (Y @ C_sqrt.T)
            X = np.clip(X, -5, 5)

            # Evaluate
            F = np.array([func(x) for x in X])
            evaluations_used += pop_size

            # Sort by fitness
            order = np.argsort(F)
            X_sorted = X[order]
            F_sorted = F[order]

            # Update best
            if F_sorted[0] < f_best:
                f_best = F_sorted[0]
                x_best = X_sorted[0].copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if F_sorted[0] < best_f_cma:
                best_f_cma = F_sorted[0]

            # Weighted recombination
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += weights[i] * (X_sorted[i] - mean) / sigma

            # Update sigma (step-size adaptation)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_w / chi_n
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 10.0)

            # Update covariance matrix
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
            delta_heaviside = 1.0 if np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c)**2) / chi_n > 1.4 + 2 / (dim + 1) else 0.0

            rank_one = np.outer(y_w, y_w)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                y_i = (X_sorted[i] - mean) / sigma
                rank_mu += weights[i] * np.outer(y_i, y_i)

            C = (1 + c_1 * delta_heaviside * (np.outer(p_c, p_c) - C) - c_1 * C + c_mu * (rank_mu - C))

            # Ensure C is symmetric and positive semi-definite
            C = (C + C.T) / 2
            try:
                C_sqrt = np.linalg.cholesky(C + 1e-8 * np.eye(dim))
            except np.linalg.LinAlgError:
                # Fallback: use eigenvalue decomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-8)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))

            # Update mean
            mean = mean + sigma * y_w
            mean = np.clip(mean, -5, 5)

            step_count += 1

            # Adaptive restart: if stagnating, try a different strategy
            if stagnation_counter > 20 and step_count > 50:
                # Try a larger step size or restart from a different point
                if stagnation_counter > 50:
                    # Reduce population and increase exploration
                    sigma *= 1.5
                    sigma = min(sigma, 10.0)
                    stagnation_counter = 0
                elif stagnation_counter > 30:
                    # Slight increase in exploration
                    sigma *= 1.2
                    sigma = min(sigma, 10.0)
                    stagnation_counter = 0

        return x_best, f_best
# EVOLVE-BLOCK-END
