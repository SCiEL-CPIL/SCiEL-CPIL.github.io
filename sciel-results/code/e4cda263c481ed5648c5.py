"""Rewritten optimizer for BBOB f4 (Büche-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts and time guard for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')

        def time_ok():
            return time.time() < deadline

        # CMA-ES implementation
        def cmaes(x0, sigma0, budget_max):
            n = self.dim
            # CMA-ES parameters
            mu = max(1, n // 2)  # number of parents
            lambda_ = 2 * mu  # population size
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / sum(weights ** 2)
            c_sigma = (mueff + 2) / (n + mueff + 5)
            d_sigma = 1 + 2 * max(0, sqrt(mueff / (n + 1)) - 1) + c_sigma
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))

            x = np.array(x0, dtype=float)
            sigma = sigma0
            pc = np.zeros(n)
            ps = np.zeros(n)
            B = np.eye(n)
            D = np.ones(n)

            # Initial evaluation
            fx = func(x)
            if not np.isfinite(fx):
                fx = 1e10
            best_local = fx
            best_x_local = x.copy()

            for gen in range(budget_max // lambda_ + 1):
                if not time_ok():
                    break

                # Generate population
                offspring = []
                for i in range(lambda_):
                    z = rng.standard_normal(n)
                    y = x + sigma * (B @ (D * z))
                    y = np.clip(y, -5, 5)
                    offspring.append((y, i))

                # Evaluate offspring
                vals = []
                for y, _ in offspring:
                    fy = func(y)
                    if not np.isfinite(fy):
                        fy = 1e10
                    vals.append(fy)
                    if fy < best_local:
                        best_local = fy
                        best_x_local = y.copy()

                # Sort by fitness
                idx = np.argsort(vals)
                selected = [offspring[i][0] for i in idx[:mu]]
                selected_vals = [vals[i] for i in idx[:mu]]

                # Update mean
                x_new = sum(w * xi for w, xi in zip(weights, selected))
                x_new = np.clip(x_new, -5, 5)

                # Update evolution paths
                if sigma > 0:
                    # Normalize step
                    v = (x_new - x) / sigma
                    pc = (1 - c_c) * pc + sqrt(c_c * (2 - c_c) * mueff) * v
                    ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mueff) * (B.T @ (D * v))

                    # Adapt covariance matrix
                    if (pc @ pc) / n < (1 + 4 / n) * (1 - c_c):
                        rank1 = np.outer(pc, pc)
                    else:
                        rank1 = np.zeros((n, n))
                    rankmu = sum(w * np.outer(xi - x, xi - x) for w, xi in zip(weights, selected)) / sigma ** 2

                    C = (1 + c_1 * (n + 1 - 2) + c_mu - c_1) * C + c_1 * (rank1 - C) + c_mu * rankmu
                    # Ensure symmetry
                    C = (C + C.T) / 2

                # Eigen decomposition for step size
                try:
                    D2, B2 = np.linalg.eigh(C)
                    D2 = np.sqrt(np.maximum(D2, 1e-12))
                    B = B2
                    D = D2
                except np.linalg.LinAlgError:
                    break

                # Step size adaptation
                if sigma > 0:
                    chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                    sigma = np.clip(sigma, 1e-12, 10.0)

                x = x_new

            return best_x_local, best_local

        # Strategy: multiple restarts with different initializations
        # For BBOB f4, the function has many local minima due to the Rastrigin component
        # We use a combination of random restarts and local search

        # Phase 1: Random sampling to get a rough idea of the landscape
        n_samples = min(50, self.budget // 10)
        for _ in range(n_samples):
            if not time_ok():
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            if not np.isfinite(fx):
                fx = 1e10
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: CMA-ES from best point and a few other good points
        # Try CMA-ES from the best point found
        if time_ok():
            remaining = self.budget - n_samples
            # Use most of the budget for CMA-ES
            cma_budget = int(remaining * 0.7)
            bx, bf = cmaes(best_x, 1.0, cma_budget)
            if bf < best_f:
                best_f = bf
                best_x = bx

        # Phase 3: If time remains, try another CMA-ES from a different start
        if time_ok():
            # Try from origin or another heuristic point
            alt_start = np.zeros(self.dim)
            remaining2 = self.budget - n_samples - int((self.budget - n_samples) * 0.7)
            if remaining2 > 50:
                bx2, bf2 = cmaes(alt_start, 2.0, remaining2)
                if bf2 < best_f:
                    best_f = bf2
                    best_x = bx2

        # Phase 4: Local refinement with small steps
        if time_ok() and best_x is not None:
            # Simple local search with decreasing step size
            x = best_x.copy()
            fx = best_f
            step = 0.5
            for _ in range(50):
                if not time_ok():
                    break
                improved = False
                for d in range(self.dim):
                    for sign in [1, -1]:
                        y = x.copy()
                        y[d] += sign * step
                        y = np.clip(y, -5, 5)
                        fy = func(y)
                        if fy < fx:
                            x, fx = y, fy
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    step *= 0.5
                    if step < 1e-6:
                        break
            if fx < best_f:
                best_f = fx
                best_x = x

        return best_x, best_f
# EVOLVE-BLOCK-END
