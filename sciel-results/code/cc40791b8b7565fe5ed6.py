"""Rewritten optimizer: CMA-ES with LHS initialization and eigendecomposition for BBOB f1 (Sphere)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with Latin Hypercube initialization and eigendecomposition for BBOB f1 (Sphere)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        dim = self.dim
        budget = self.budget

        call_count = 0
        best_x = None
        best_f = np.inf

        def tracked_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return best_f
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Latin Hypercube Sampling for initial exploration
        n_lhs = min(50, budget // 5)

        def latin_hypercube(n, d, rng):
            points = np.zeros((n, d))
            for i in range(d):
                intervals = rng.random(n)
                order = rng.permutation(n)
                points[order, i] = intervals
            # Map to [-4.5, 4.5]^d (minimizer is in [-4, 4]^5)
            points = points * 9.0 - 4.5
            return points

        lhs_points = latin_hypercube(n_lhs, dim, rng)

        # Evaluate LHS points
        lhs_results = []
        for idx in range(len(lhs_points)):
            if call_count >= budget:
                break
            if time.time() > deadline - 60:
                break
            x = lhs_points[idx]
            f = tracked_func(x)
            lhs_results.append((f, x))

        if not lhs_results:
            x0 = np.zeros(dim)
            f0 = tracked_func(x0)
            lhs_results = [(f0, x0)]

        # Sort and take best point as CMA-ES starting center
        lhs_results.sort(key=lambda t: t[0])
        center = lhs_results[0][1].copy()

        # Phase 2: CMA-ES with eigendecomposition for covariance updates
        remaining_budget = budget - call_count
        if remaining_budget <= 0 or time.time() > deadline - 30:
            return best_x, best_f

        # CMA-ES parameters
        pop_size = 10  # Larger population for better exploration
        sigma = 2.0  # Initial step size to cover the space
        x_mean = center.copy()

        # Covariance matrix initialization
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)

        def update_eig():
            nonlocal C, C_inv_sqrt
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-20, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        update_eig()

        # Strategy parameters
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Learning rates
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Main CMA-ES loop
        max_iters = min(remaining_budget // pop_size, 100)

        for it in range(max_iters):
            if call_count >= budget:
                break
            if time.time() > deadline - 10:
                break

            # Generate population
            population = []
            for i in range(pop_size):
                z = rng.standard_normal(dim)
                y = C_inv_sqrt @ z
                x = x_mean + sigma * y
                x = np.clip(x, -4.5, 4.5)
                population.append(x)

            # Evaluate population
            fitness = []
            for x in population:
                f = tracked_func(x)
                fitness.append(f)

            # Sort by fitness
            sorted_indices = np.argsort(fitness)
            population = [population[i] for i in sorted_indices]
            fitness = [fitness[i] for i in sorted_indices]

            # Update mean
            new_mean = np.zeros(dim)
            for i in range(mu):
                new_mean += weights[i] * population[i]

            # Update evolution paths
            old_sigma = sigma
            old_mean = x_mean.copy()
            x_mean = new_mean

            # Update p_sigma
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * (
                C_inv_sqrt @ (x_mean - old_mean) / old_sigma
            )

            # h_sigma check
            chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * (it + 1))) < (1.4 + 1 / dim) * chi_n:
                h_sigma = 1.0
            else:
                h_sigma = 0.0

            # Update p_c
            p_c = (1 - cs) * p_c + h_sigma * np.sqrt(cs * (2 - cs) * mu_eff) * (
                x_mean - old_mean
            ) / sigma

            # Update covariance matrix
            delta_h = (h_sigma - 1) * cs * (2 - cs)
            C = (1 + c1 * delta_h - c1 - cmu) * C
            for i in range(mu):
                y_i = (population[i] - old_mean) / sigma
                C += cmu * weights[i] * np.outer(y_i, y_i)
            C += c1 * delta_h * np.outer(p_c, p_c)

            # Ensure positive definiteness and update eigendecomposition
            C = (C + C.T) / 2
            try:
                update_eig()
            except:
                C += 1e-10 * np.eye(dim)
                update_eig()

            # Update sigma
            sigma *= np.exp(cs / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))

            # Adaptation limits
            sigma = np.clip(sigma, 1e-12, 10.0)

            # Check convergence
            if sigma < 1e-10:
                break

            # Update best
            if fitness[0] < best_f:
                best_f = fitness[0]
                best_x = population[0].copy()

        # Phase 3: Local refinement with adaptive step size
        remaining = budget - call_count
        if remaining > 0 and best_x is not None and time.time() < deadline - 5:
            local_sigma = max(0.01, sigma * 0.5)
            for _ in range(min(remaining, 50)):
                if call_count >= budget:
                    break
                if time.time() > deadline - 2:
                    break
                y = np.clip(best_x + local_sigma * rng.standard_normal(dim), -4.5, 4.5)
                fy = tracked_func(y)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    local_sigma *= 0.7
                else:
                    local_sigma *= 0.9

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = float(func(best_x))

        return best_x, best_f
# EVOLVE-BLOCK-END
