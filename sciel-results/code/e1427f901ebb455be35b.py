"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES with adaptive sigma and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # 300s limit, 20s safety margin

        # Use a deterministic seed for reproducibility
        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = np.inf
        calls = 0

        # Helper to evaluate and track
        def evaluate(x):
            nonlocal calls, best_x, best_f
            x_clipped = np.clip(x, -5.0, 5.0)
            fx = func(x_clipped)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # Generate diverse starting points
        start_points = []
        # Origin
        start_points.append(np.zeros(dim))
        # Random points
        for _ in range(8):
            start_points.append(rng.uniform(-5, 5, dim))
        # Some structured points
        for _ in range(4):
            p = np.zeros(dim)
            for i in range(dim):
                p[i] = rng.choice([-4, -2, 0, 2, 4])
            start_points.append(p)

        # Phase 1: Quick evaluation of diverse starting points (use ~20% of budget)
        phase1_budget = min(100, budget // 5)
        evaluated_starts = []
        for sp in start_points:
            if calls >= phase1_budget or time.time() > deadline:
                break
            f_sp = evaluate(sp)
            evaluated_starts.append((f_sp, sp))

        # Sort by fitness
        evaluated_starts.sort(key=lambda t: t[0])

        # Phase 2: CMA-ES from top starting points
        # Use remaining budget
        remaining_budget = budget - calls
        if remaining_budget <= 0 or time.time() > deadline:
            return best_x, best_f

        # Allocate remaining budget among top starts
        n_restarts = min(5, len(evaluated_starts))
        budget_per_restart = remaining_budget // n_restarts

        for idx in range(n_restarts):
            if calls >= budget or time.time() > deadline:
                break
            if idx >= len(evaluated_starts):
                break

            f_init, x_init = evaluated_starts[idx]
            x = x_init.copy()
            fx = f_init

            # CMA-ES parameters
            sigma = 1.0
            # Population size
            pop_size = max(4, int(4 + 3 * np.log(dim)))
            # Covariance matrix
            C = np.eye(dim)
            # Step sizes for C and sigma
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            # Evolution path
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            iters_in_restart = 0
            max_iters = budget_per_restart

            while iters_in_restart < max_iters:
                if calls >= budget or time.time() > deadline:
                    break

                # Generate population
                archive = []
                for i in range(pop_size):
                    # Sample from multivariate normal
                    z = rng.multivariate_normal(np.zeros(dim), C, 1).flatten()
                    y = x + sigma * z
                    y = np.clip(y, -5.0, 5.0)
                    fy = evaluate(y)
                    archive.append((fy, y))
                    iters_in_restart += 1

                    if calls >= budget or time.time() > deadline:
                        break

                if len(archive) < mu:
                    break

                # Sort by fitness
                archive.sort(key=lambda t: t[0])

                # Update mean
                x_new = np.zeros(dim)
                for i in range(mu):
                    x_new += weights[i] * archive[i][1]
                x = x_new

                # Update sigma via evolution path
                # Compute weighted average of step vectors
                d = np.zeros(dim)
                for i in range(mu):
                    d += weights[i] * (archive[i][1] - x) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * d
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)

                # Update covariance matrix
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x - x_new) / sigma
                # Note: x_new is the new mean, x was the old mean... actually let's be careful
                # Standard CMA-ES:
                # h_sigma = indicator that p_sigma is not too large
                # For simplicity, let's use a simplified covariance update
                C = (1 - c_1 - c_mu) * C + c_1 * (p_c @ p_c + (1 - (1 - c_c) ** 2) * np.eye(dim))
                for i in range(mu):
                    y_i = (archive[i][1] - x_new) / sigma
                    C += c_mu * weights[i] * np.outer(y_i, y_i)

                # Ensure C is symmetric and positive definite
                C = (C + C.T) / 2
                # Add small regularization for stability
                C += 1e-10 * np.eye(dim)

            # After CMA-ES, do a local polish with small steps
            if calls < budget and time.time() < deadline:
                sigma_local = 0.1
                for _ in range(20):
                    if calls >= budget or time.time() > deadline:
                        break
                    y = np.clip(x + sigma_local * rng.standard_normal(dim), -5, 5)
                    fy = evaluate(y)
                    if fy < fx:
                        x = y
                        fx = fy
                        sigma_local *= 0.9
                    else:
                        sigma_local *= 0.7

        return best_x, best_f
# EVOLVE-BLOCK-END
