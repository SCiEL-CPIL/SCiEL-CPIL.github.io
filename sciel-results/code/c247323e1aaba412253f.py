"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive population and (1+1)-ES polish for sharp ridge."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Phase 1: Initial sampling to find a good starting point
        best_x = None
        best_fx = np.inf
        n_init = min(15, budget)
        for i in range(n_init):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_fx = func(best_x)

        x = best_x.copy()
        fx = best_fx

        # CMA-ES state
        mean = x.copy()
        sigma = 1.5
        C = np.eye(dim)
        
        # CMA-ES parameters
        c_sigma = (1 + 2 * max(0, (np.sqrt(dim) - 1) / (dim + 2))) / (dim + 2 + 2 * max(0, (np.sqrt(dim) - 1) / (dim + 2)))
        c_c = (4 + dim / 4) / (dim + 4 + 2 * c_sigma)
        c_1 = 2 / ((dim + 1.3) ** 2 + dim)
        c_mu = min(1 - c_1, 2 * ((dim + 1) / 2 - 1) / ((dim + 2) ** 2 + dim))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Population size: start with 10, adaptively shrink
        lambda_pop = 10
        mu = lambda_pop // 2
        weights = np.log(lambda_pop + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights ** 2)

        calls_used = n_init
        stall_count = 0
        max_stall = 25
        converged = False

        # Main loop: CMA-ES phase
        while calls_used < budget:
            if time.time() > deadline:
                break

            # Check if we should transition to (1+1)-ES polish
            if sigma < 0.005 or calls_used > budget * 0.6:
                converged = True
                break

            # Adaptive population: shrink as sigma decreases
            current_lambda = max(5, int(lambda_pop * max(0.5, min(1.0, sigma / 0.5))))
            if current_lambda < 3:
                current_lambda = 3

            current_mu = max(2, current_lambda // 2)
            current_weights = np.log(current_lambda + 0.5) - np.log(np.arange(1, current_mu + 1))
            current_weights /= current_weights.sum()
            current_mu_eff = 1 / np.sum(current_weights ** 2)

            # Generate population
            population = []
            for i in range(current_lambda):
                if time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)
                step = sigma * L @ z
                y = np.clip(mean + step, -5, 5)
                fy = func(y)
                calls_used += 1
                population.append((fy, y.copy()))

            if not population:
                break

            # Sort by fitness
            population.sort(key=lambda t: t[0])

            # Update mean
            new_mean = np.zeros(dim)
            for i in range(current_mu):
                new_mean += current_weights[i] * population[i][1]
            mean = new_mean

            # Update best
            if population[0][0] < best_fx:
                best_fx = population[0][0]
                best_x = population[0][1].copy()
                stall_count = 0
            else:
                stall_count += 1

            # Update sigma (CSA)
            if sigma > 1e-15:
                y_old = (mean - x) / (sigma * chi_n)
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * current_mu_eff) * y_old
                sigma *= np.exp(c_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

            # Update C matrix
            if sigma > 1e-15:
                y_c = (mean - x) / (sigma * chi_n)
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * current_mu_eff) * y_c

                # Rank-mu update
                outer = np.zeros((dim, dim))
                for i in range(current_mu):
                    z_i = (population[i][1] - x) / sigma
                    outer += current_weights[i] * np.outer(z_i, z_i)

                C = (1 + c_1 * (2 - c_1) - c_mu) * C + c_1 * np.outer(p_c, p_c) + c_mu * outer
                C = (C + C.T) / 2

                # Ensure positive semi-definite
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            x = mean.copy()
            fx = best_fx

            # Stall detection: restart from a perturbation of best point
            if stall_count >= max_stall:
                stall_count = 0
                sigma = max(sigma * 0.5, 0.05)
                # Try a few local perturbations around best
                for _ in range(3):
                    if time.time() > deadline:
                        break
                    y = np.clip(best_x + sigma * rng.standard_normal(dim), -5, 5)
                    fy = func(y)
                    calls_used += 1
                    if fy < fx:
                        x, fx = y, fy
                        break
                    if fy < best_fx:
                        best_fx = fy
                        best_x = y.copy()

        # Phase 2: (1+1)-ES polish with adaptive step size
        if not converged and calls_used < budget:
            x = best_x.copy()
            fx = best_fx
            sigma = max(sigma * 0.5, 0.01)
            stall_count = 0
            max_stall = 20

            while calls_used < budget:
                if time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    stall_count = 0
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9
                    stall_count += 1

                # Stall detection in polish phase
                if stall_count >= max_stall:
                    stall_count = 0
                    sigma = max(sigma * 0.5, 0.001)
                    for _ in range(2):
                        if time.time() > deadline:
                            break
                        y = np.clip(best_x + sigma * rng.standard_normal(dim), -5, 5)
                        fy = func(y)
                        calls_used += 1
                        if fy < fx:
                            x, fx = y, fy
                            break
                        if fy < best_fx:
                            best_fx = fy
                            best_x = y.copy()

                # Early exit if sigma is very small
                if sigma < 1e-6:
                    break

        return best_x, best_fx
# EVOLVE-BLOCK-END
