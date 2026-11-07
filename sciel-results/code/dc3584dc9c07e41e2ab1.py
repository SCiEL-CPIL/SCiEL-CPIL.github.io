"""Optimizer for BBOB-style f17 (Schaffers F7) with hidden shift/rotation/offset."""
import numpy as np
import time


class Optimizer:
    """CMA-ES with deadline guard and polish phase."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 270.0  # safety margin below the 330 s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Initial point: center of the box
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1

        # CMA-ES parameters
        # Population size
        lambda_ = int(4 + 3 * np.log(dim))
        # Step size
        sigma = 2.0
        # Weighting
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Adaptation parameters
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Covariance matrix
        C = np.eye(dim)
        # Inverse square root of C
        C_sqrt_inv = np.eye(dim)

        # Evolution path for covariance
        p_c = np.zeros(dim)
        # Evolution path for step size
        p_sigma = np.zeros(dim)

        # Initial mean
        mean = x.copy()

        # Precompute eigen decomposition for C_sqrt_inv updates
        def update_C_sqrt_inv(C):
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            return C_sqrt_inv

        # Main CMA-ES loop
        while calls < budget:
            # Check deadline
            if time.time() > deadline - 30.0:
                break

            # Generate population
            pop = np.empty((lambda_, dim))
            pop_f = np.empty(lambda_)

            for i in range(lambda_):
                z = rng.standard_normal(dim)
                xi = mean + sigma * C_sqrt_inv @ z
                xi = np.clip(xi, -5, 5)
                pop[i] = xi
                pop_f[i] = func(xi)
                calls += 1
                if calls >= budget:
                    break

            if calls >= budget:
                break

            # Sort by fitness
            order = np.argsort(pop_f)
            pop = pop[order]
            pop_f = pop_f[order]

            # Update best
            if pop_f[0] < best_f:
                best_f = pop_f[0]
                best_x = pop[0].copy()

            # Weighted mean of best mu solutions
            new_mean = np.sum(weights[:, None] * pop[:mu], axis=0)

            # Update evolution paths
            old_mean = mean.copy()
            mean = new_mean

            # p_sigma update
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * \
                      C_sqrt_inv @ ((mean - old_mean) / sigma)

            # Check h_sigma condition
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            h_sigma = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * (budget - calls))) < \
                       (1.4 + 2 / (dim + 1)) * chi_n else 0.0

            # p_c update
            p_c = (1 - cc) * p_c + h_sigma * np.sqrt(cc * (2 - cc) * mu_eff) * \
                  C_sqrt_inv @ ((mean - old_mean) / sigma)

            # Covariance matrix update
            delta_h = (1 - h_sigma) * cc * (2 - cc)
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.sum(weights[:, None, None] * np.outer(pop[:mu] - old_mean, pop[:mu] - old_mean), axis=0)

            C = (1 + c1 * delta_h - c1 - cmu) * C + c1 * rank_one + cmu * rank_mu

            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            eigvals, _ = np.linalg.eigh(C)
            if np.min(eigvals) < 1e-20:
                C += 1e-20 * np.eye(dim)

            # Update C_sqrt_inv
            C_sqrt_inv = update_C_sqrt_inv(C)

            # Step size adaptation
            sigma *= np.exp(cs / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 100.0)

            # Convergence check
            if np.linalg.norm(mean - old_mean) < 1e-10 * (1 + np.abs(mean)):
                # Try to restart with a smaller step size if stuck
                if sigma > 0.01:
                    sigma *= 0.5
                else:
                    break

        # Polish phase: local refinement using remaining budget
        remaining = budget - calls
        if remaining > 5 and time.time() < deadline - 5.0:
            # Use a simple local search around best_x
            local_sigma = max(sigma * 0.1, 1e-4)
            for _ in range(min(remaining, 50)):
                if time.time() > deadline - 2.0:
                    break
                y = best_x + local_sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    local_sigma *= 1.2
                else:
                    local_sigma *= 0.8

        return best_x, best_f
