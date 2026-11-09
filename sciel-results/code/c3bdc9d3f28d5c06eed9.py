"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with adaptive population size and local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        calls_used = 0
        best_x = np.full(dim, np.nan)
        best_f = np.inf

        def try_update(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def check_time():
            return calls_used < budget and time.time() < deadline - 5.0

        rng = np.random.default_rng(12345)

        # Phase 1: Initial diverse sampling to get a good starting point
        n_init = 2 * dim + 4
        pop = []
        pop.append(np.zeros(dim))
        for axis in range(dim):
            for sign in [-1, 1]:
                p = np.zeros(dim)
                p[axis] = sign * 3.5
                pop.append(p)
        while len(pop) < n_init:
            pop.append(rng.uniform(-4.5, 4.5, dim))

        # Evaluate initial population
        pop_evaluated = []
        for x in pop:
            if not check_time():
                break
            fx = func(x)
            calls_used += 1
            pop_evaluated.append((x, fx))
            try_update(x, fx)

        if len(pop_evaluated) < 4 or np.isnan(best_x).any():
            return best_x, best_f

        # Find the best initial point as CMA-ES starting mean
        pop_sorted = sorted(pop_evaluated, key=lambda t: t[1])
        mean = pop_sorted[0][0].copy()
        sigma = 1.5  # initial step size

        # CMA-ES parameters
        lambda_pop = 10  # population size
        mu = lambda_pop // 2  # number of parents
        # Weights for the best mu individuals
        weights = np.log(np.arange(mu, 0, -1) + 0.5) - np.log(np.arange(mu, 0, -1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Covariance matrix (identity initially)
        C = np.eye(dim)
        # Eigen decomposition for efficient sampling
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 1e-10)
        sigma_C = sigma * np.sqrt(eigvals)

        # Adaptation rates
        c_sigma = (4 + mu_eff / dim) / ((dim + 4) ** 2 + mu_eff)
        c_c = (4 + mu_eff / dim) / (dim + 4)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Step size control
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        rho_sigma = (1 - np.exp(-c_sigma * chi_n)) / (1 + np.exp(-c_sigma * chi_n))
        d_sigma = 1.0
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Track best
        best_x_cma = mean.copy()
        best_f_cma = best_f

        # Main CMA-ES loop
        gen = 0
        while check_time():
            gen += 1

            # Generate lambda_pop samples
            offspring = []
            for i in range(lambda_pop):
                # Sample from multivariate normal using eigen decomposition
                z = rng.standard_normal(dim)
                # Transform: eigvecs @ diag(sigma_C) @ z
                x_new = mean + eigvecs @ (sigma_C * z)
                x_new = np.clip(x_new, -5, 5)
                offspring.append(x_new)

            # Evaluate offspring
            offspring_f = []
            for x in offspring:
                if not check_time():
                    break
                fx = func(x)
                calls_used += 1
                offspring_f.append((x, fx))

            if len(offspring_f) < mu:
                break

            # Sort by fitness
            offspring_f.sort(key=lambda t: t[1])
            selected = offspring_f[:mu]

            # Update best
            if selected[0][1] < best_f_cma:
                best_f_cma = selected[0][1]
                best_x_cma = selected[0][0].copy()
                try_update(best_x_cma, best_f_cma)

            # Compute new mean
            new_mean = np.sum([weights[i] * selected[i][0] for i in range(mu)], axis=0)

            # Update evolution paths
            old_mean = mean.copy()
            mean_diff = new_mean - old_mean

            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (
                (eigvecs.T @ (eigvecs @ (mean_diff / sigma)))
            )

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = max(sigma, 1e-8)
            sigma = min(sigma, 10.0)

            # Update p_c
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))) < (
                1.4 + 2 / (dim + 1)
            ) * chi_n
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (
                eigvecs.T @ (eigvecs @ (mean_diff / sigma))
            )

            # Update covariance matrix
            delta_h = (14 / 10 - 1) * (1 - h_sigma)
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                z_i = eigvecs.T @ (eigvecs @ (selected[i][0] - old_mean) / sigma)
                rank_mu += weights[i] * np.outer(z_i, z_i)
            C = (1 + c_1 * (delta_h - 1) - c_mu) * C + c_1 * rank_one + c_mu * rank_mu

            # Update eigen decomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            # Ensure C is positive semi-definite by re-orthogonalizing if needed
            if np.any(np.abs(eigvals) < 1e-10):
                C = np.eye(dim)
                eigvals = np.ones(dim)
                eigvecs = np.eye(dim)

            sigma_C = sigma * np.sqrt(eigvals)

            # Update mean
            mean = new_mean

            # Adaptive population size: increase if stuck
            if gen > 50 and len(offspring_f) == mu:
                # Check if we're making progress
                recent_best = min(f for _, f in offspring_f[:min(5, len(offspring_f))])
                if recent_best >= best_f_cma * 0.999:
                    lambda_pop = min(lambda_pop + 2, 20)

        # Phase 2: Local refinement around best point
        if check_time() and not np.isnan(best_x).any():
            x = best_x.copy()
            fx = best_f
            sigma_local = 0.3

            no_improve = 0
            max_no_improve = 20

            while check_time() and no_improve < max_no_improve:
                y = np.clip(x + sigma_local * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1

                if fy < fx:
                    x, fx = y, fy
                    try_update(x, fx)
                    sigma_local *= 1.1
                    no_improve = 0
                else:
                    sigma_local *= 0.85
                    no_improve += 1

            # Coordinate-wise refinement
            if check_time() and not np.isnan(best_x).any():
                x = best_x.copy()
                fx = best_f
                step = 0.3
                for coord in range(dim):
                    if not check_time():
                        break
                    for _ in range(8):
                        if not check_time():
                            break
                        y = x.copy()
                        y[coord] += step
                        y = np.clip(y, -5, 5)
                        fy = func(y)
                        calls_used += 1
                        if fy < fx:
                            x, fx = y, fy
                            try_update(x, fx)
                        else:
                            y = x.copy()
                            y[coord] -= step
                            y = np.clip(y, -5, 5)
                            fy = func(y)
                            calls_used += 1
                            if fy < fx:
                                x, fx = y, fy
                                try_update(x, fx)
                            else:
                                break
                    step *= 0.5

        return best_x, best_f
# EVOLVE-BLOCK-END
