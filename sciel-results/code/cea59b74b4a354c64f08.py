"""Optimizer for BBOB f9 (Rosenbrock rotated) using CMA-ES population-based evolution."""
import numpy as np
import time


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for black-box optimization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        calls_made = 0

        best_x = np.zeros(dim)
        best_f = np.inf

        def safe_func(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            f = float(func(x))
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # CMA-ES parameters
        # Population size: lambda = 4 + floor(3 * log(n))
        lam = 4 + int(3 * np.log(dim)) if dim > 1 else 4
        mu = lam // 2  # number of parents
        # Weights for recombination
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Step size adaptation
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Initial parameters
        x_mean = np.zeros(dim)
        sigma = 2.0  # initial step size for [-5,5]^5
        C = np.eye(dim)  # covariance matrix
        B = np.eye(dim)  # eigenvectors
        D = np.ones(dim)  # eigenvalues (sqrt)
        pc = np.zeros(dim)  # evolution path for covariance
        ps = np.zeros(dim)  # evolution path for step size
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Evaluate initial point
        safe_func(x_mean)

        # Main CMA-ES loop
        while calls_made < budget and time.time() < deadline - 5:
            # Generate population
            population = []
            for i in range(lam):
                if calls_made >= budget or time.time() > deadline - 5:
                    break
                # Sample from N(x_mean, sigma^2 * C)
                z = np.random.randn(dim)
                # Apply transformation: B * D * z
                x_sample = x_mean + sigma * (B @ (D * z))
                population.append((x_sample, i))

            # Evaluate population
            fvals = []
            for x_sample, i in population:
                if calls_made >= budget or time.time() > deadline - 5:
                    break
                fvals.append(safe_func(x_sample))

            if len(fvals) < 2:
                break

            # Sort by fitness
            sorted_idx = np.argsort(fvals)
            population = [population[i] for i in sorted_idx]
            fvals = [fvals[i] for i in sorted_idx]

            # Check for convergence
            if fvals[0] == fvals[-1]:
                break
            if np.std(fvals[:mu]) < 1e-12:
                break

            # Update mean
            x_new_mean = np.sum([weights[i] * population[i][0] for i in range(mu)], axis=0)

            # Update step size (sigma)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (B @ (D @ (x_new_mean - x_mean)))
            if np.linalg.norm(ps) / chi_n < (1.4 + 2 / (dim + 1)) * np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made // lam))):
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

            # Update covariance matrix
            diff = (x_new_mean - x_mean) / sigma
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * diff
            delta_h = 1 if np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c) ** 2) < (1.4 + 2 / (dim + 1)) * chi_n else 0

            # Rank-one and rank-mu updates
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + \
                c_1 * np.outer(pc, pc) + \
                c_mu * np.sum([weights[i] * np.outer(
                    (population[i][0] - x_mean) / sigma,
                    (population[i][0] - x_mean) / sigma
                ) for i in range(mu)], axis=0)

            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            # Regularize if needed
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update B and D
            B, D = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(np.diag(D), 1e-20))

            x_mean = x_new_mean
            sigma = max(sigma, 1e-10)

            # Check if we should stop due to lack of progress
            if calls_made % (lam * 10) == 0 and calls_made > 100:
                if np.std(fvals[:min(10, len(fvals))]) < 1e-10:
                    break

        # Final refinement: local search around best point
        if best_x is not None and calls_made + 50 < budget and time.time() < deadline - 10:
            # Simple coordinate descent refinement
            x = best_x.copy()
            f = best_f
            step = 0.1
            for _ in range(20):
                if calls_made >= budget or time.time() > deadline - 5:
                    break
                improved = False
                for i in range(dim):
                    for direction in [1, -1]:
                        if calls_made >= budget or time.time() > deadline - 5:
                            break
                        x_try = x.copy()
                        x_try[i] += direction * step
                        x_try = np.clip(x_try, -5, 5)
                        f_try = safe_func(x_try)
                        if f_try < f:
                            x = x_try
                            f = f_try
                            improved = True
                if not improved:
                    step *= 0.5
                    if step < 1e-6:
                        break

        if best_x is None:
            best_x = np.zeros(dim)
            safe_func(best_x)

        return best_x, best_f
