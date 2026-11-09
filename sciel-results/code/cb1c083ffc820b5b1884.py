"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with multi-start for Lunacek bi-Rastrigin optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s kill

        n = self.dim
        total_budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_func(x):
            nonlocal best_x, best_f, calls_made
            x = np.clip(x, -5, 5)
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 0: Multi-start initialization to find both basins of Lunacek
        init_points = [
            np.zeros(n),
            rng.uniform(-1, 1, n),
            rng.uniform(-2, 2, n),
            np.ones(n) * 2.5,
            np.ones(n) * -2.5,
            rng.uniform(-4, 4, n),
        ]
        for xp in init_points:
            if calls_made >= total_budget:
                break
            safe_func(xp)

        # Phase 1: CMA-ES optimization
        def run_cma_es(x0, sigma0, max_calls):
            nonlocal calls_made, best_x, best_f
            if calls_made >= total_budget or time.time() >= deadline - 10:
                return

            # CMA-ES parameters
            pop_size = 4 + int(3 * np.log(n))  # lambda = 4 + 3*ln(n)
            if pop_size > total_budget - calls_made:
                pop_size = max(2, total_budget - calls_made)

            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            C_inv = np.eye(n)
            C_sqrt = np.eye(n)

            # Evolution path
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)

            # Adaptation parameters
            mu = pop_size // 2  # number of recombination weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / (weights ** 2).sum()

            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

            # Precompute C_sqrt via Cholesky (will be updated)
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C_sqrt = np.eye(n)

            stagnation = 0
            best_local_f = np.inf

            while calls_made < total_budget and time.time() < deadline - 5:
                if stagnation > 3 * pop_size:
                    break

                # Generate population
                offspring = []
                offspring_f = []

                for i in range(pop_size):
                    if calls_made >= total_budget or time.time() >= deadline - 5:
                        break
                    z = rng.standard_normal(n)
                    y = x_mean + sigma * (C_sqrt @ z)
                    y = np.clip(y, -5, 5)
                    fy = safe_func(y)
                    offspring.append(y)
                    offspring_f.append(fy)

                if len(offspring) < 2:
                    break

                # Sort by fitness
                sorted_idx = np.argsort(offspring_f)
                offspring = [offspring[i] for i in sorted_idx[:mu]]
                offspring_f = [offspring_f[i] for i in sorted_idx[:mu]]

                # Update best
                if offspring_f[0] < best_local_f:
                    if offspring_f[0] < best_f:
                        best_f = offspring_f[0]
                        best_x = offspring[0].copy()
                    best_local_f = offspring_f[0]
                    stagnation = 0
                else:
                    stagnation += 1

                # Weighted mean of selected offspring
                x_new = np.zeros(n)
                for i in range(mu):
                    x_new += weights[i] * offspring[i]

                # Update evolution paths
                y_vec = (x_new - x_mean) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_vec
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_vec

                # Step size adaptation (1/5 rule)
                if np.linalg.norm(p_sigma) > (1.4 + 1 / n) * np.sqrt(1 - (1 - c_sigma) ** (2 * (pop_size + 1))):
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (pop_size + 1))) - 1))

                # Update covariance matrix
                delta = 0 if np.linalg.norm(p_c) ** 2 <= (4 + 2 / n) * (1 - (1 - c_c) ** (2 * (pop_size + 1))) else 1
                C = (1 + c_1 + c_mu - c_c) * C + c_c * np.outer(p_c, p_c)
                for i in range(mu):
                    y_w = (offspring[i] - x_mean) / sigma
                    C += c_mu * weights[i] * (np.outer(y_w, y_w) - C)

                # Ensure positive definiteness
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Regularize
                    C += 0.1 * np.eye(n)
                    try:
                        C_sqrt = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        C = np.eye(n)
                        C_sqrt = np.eye(n)

                x_mean = x_new

                # Reset if sigma becomes too small
                if sigma < 1e-8:
                    break

        # Run CMA-ES from best initial point
        run_cma_es(best_x.copy(), 1.0, total_budget - calls_made)

        # If budget remains, run another CMA-ES from a perturbed point
        if calls_made < total_budget and time.time() < deadline - 10:
            perturb_scale = 1.0
            x_perturbed = best_x + perturb_scale * rng.standard_normal(n)
            x_perturbed = np.clip(x_perturbed, -5, 5)
            run_cma_es(x_perturbed, 0.5, total_budget - calls_made)

        # Phase 2: Fine-grained coordinate descent refinement
        if time.time() < deadline - 5 and calls_made < total_budget:
            x = best_x.copy()
            fx = best_f
            sigma = np.ones(n) * 0.05
            window = []
            window_size = 15
            stagnation_count = 0
            coord_idx = 0

            while calls_made < total_budget and time.time() < deadline - 5:
                d = rng.choice([-1, 1])
                y = x.copy()
                y[coord_idx] += d * sigma[coord_idx]
                y = np.clip(y, -5, 5)
                fy = safe_func(y)

                if fy < fx:
                    x, fx = y, fy
                    window.append(1)
                    stagnation_count = 0
                else:
                    window.append(0)
                    stagnation_count += 1

                if len(window) >= window_size:
                    success_rate = sum(window) / window_size
                    if success_rate > 0.2:
                        sigma[coord_idx] *= 1.1
                    elif success_rate < 0.05:
                        sigma[coord_idx] *= 0.9
                    window = window[1:]

                coord_idx = (coord_idx + 1) % n

                if stagnation_count >= 30:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
