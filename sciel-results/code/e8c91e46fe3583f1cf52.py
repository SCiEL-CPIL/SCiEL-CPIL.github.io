"""Hybrid optimizer: CMA-ES exploration + Rastrigin-aware exploitation."""
import time
import numpy as np


class Optimizer:
    """Hybrid: CMA-ES for exploration, Rastrigin-snapping for exploitation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Track best
        best_x = None
        best_f = float('inf')

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        # Phase 1: CMA-ES exploration (use ~40% of budget)
        cma_budget = int(budget * 0.4)
        n_offspring = max(6, dim + 2)

        # Initialize CMA-ES
        x0 = rng.uniform(-5, 5, dim)
        fx0 = func(x0)
        update_best(x0, fx0)

        sigma = 2.0
        mean = x0.copy()

        # CMA-ES matrices
        C = np.eye(dim)
        Cs = np.zeros((dim, dim))
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # CMA-ES parameters
        mu = n_offspring // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = (2 + 2 / (dim + 1.7)) / ((dim + 2) ** 2 + mu_eff)
        c_mu = min(c_1 * 1.5, (2 - c_1) * (mu_eff - 2 + 2 / (dim + 2)) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        calls_made = 1

        for _ in range(cma_budget - 1):
            if time.time() > deadline - 50.0:
                break

            # Generate offspring
            ys = []
            fs = []
            for i in range(n_offspring):
                if calls_made >= budget:
                    break
                if time.time() > deadline - 50.0:
                    break
                z = rng.standard_normal(dim)
                y = np.clip(mean + sigma * (C @ z), -5, 5)
                fy = func(y)
                ys.append(y)
                fs.append(fy)
                calls_made += 1
                update_best(y, fy)

            if len(ys) < mu:
                break

            # Sort by fitness
            indices = np.argsort(fs)
            ys_sorted = [ys[i] for i in indices[:mu]]
            fs_sorted = [fs[i] for i in indices[:mu]]

            # Update mean
            y_w = np.sum(weights[:, None] * np.array(ys_sorted), axis=0)
            mean_new = mean + (y_w - mean)

            # Update step size
            y_sigma = np.sqrt(np.sum(weights * np.array([y - mean for y in ys_sorted]) ** 2, axis=1)) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * y_sigma
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2) / chi_n < 1.4 + 2 / (dim + 1))
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update evolution paths
            p_c = (1 - c_c) * p_c + c_c * h_sigma * (mean_new - mean) / sigma
            mean = mean_new

            # Update covariance
            delta_h = (1 + h_sigma) * (2 - c_c) * c_c - 1
            outer = np.outer(mean - mean_new, mean - mean_new)
            C = (1 - c_1 - c_mu) * C + c_1 * (2 - c_c + delta_h) * outer + \
                c_mu * sum(w * np.outer(y - mean, y - mean) for w, y in zip(weights, ys_sorted))
            C = (C + C.T) / 2  # symmetrize

            # Eigen decomposition for C @ z
            eigenvalues, eigenvectors = np.linalg.eigh(C)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            C_inv_sqrt = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
            C = (eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T + C) / 2  # keep C valid
            C_inv_sqrt = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T

            # Ensure C is positive definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-10, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        # Phase 2: Rastrigin-aware exploitation (remaining budget)
        remaining = budget - calls_made
        if remaining <= 0 or time.time() > deadline - 10.0:
            return best_x, best_f

        # Start from best point found
        x = best_x.copy()
        fx = best_f

        # Rastrigin-aware: snap coordinates to nearest integer, then polish
        # For Rastrigin, minima are at integer points (after undoing rotation/shift)
        # Strategy: try snapping each coordinate to nearest integer, then do local search

        # First, try pure integer snapping around current best
        candidates = [x.copy()]
        for i in range(dim):
            nearest_int = np.round(x[i])
            # Try nearest integer and neighbors
            for delta in [-1, 0, 1]:
                c = x.copy()
                c[i] = np.clip(nearest_int + delta, -5, 5)
                candidates.append(c)

        # Evaluate all candidates
        for c in candidates:
            if calls_made >= budget:
                break
            if time.time() > deadline - 10.0:
                break
            fc = func(c)
            calls_made += 1
            if fc < fx:
                x = c
                fx = fc
                update_best(x, fx)

        # Phase 3: Coordinate-wise local search with small steps
        # Use remaining budget for fine-tuning
        remaining = budget - calls_made
        if remaining <= 0 or time.time() > deadline - 5.0:
            return best_x, best_f

        # Adaptive step size for coordinate search
        step_sizes = [0.5, 0.2, 0.1, 0.05, 0.01, 0.001]
        for step in step_sizes:
            if calls_made >= budget:
                break
            if time.time() > deadline - 5.0:
                break
            for _ in range(min(remaining // (dim * 2 + 1), 10)):
                if calls_made >= budget:
                    break
                if time.time() > deadline - 5.0:
                    break
                improved = False
                for i in range(dim):
                    if calls_made >= budget:
                        break
                    if time.time() > deadline - 5.0:
                        break
                    # Try +step
                    y = x.copy()
                    y[i] = np.clip(y[i] + step, -5, 5)
                    fy = func(y)
                    calls_made += 1
                    if fy < fx:
                        x = y
                        fx = fy
                        update_best(x, fx)
                        improved = True
                        continue
                    # Try -step
                    y = x.copy()
                    y[i] = np.clip(y[i] - step, -5, 5)
                    fy = func(y)
                    calls_made += 1
                    if fy < fx:
                        x = y
                        fx = fy
                        update_best(x, fx)
                        improved = True

                if not improved:
                    break

        # Final: if we still have budget, do a few more random restarts from best
        remaining = budget - calls_made
        if remaining > 5 and time.time() < deadline - 10.0:
            # Try a few small perturbations from best
            for _ in range(min(remaining // 3, 5)):
                if calls_made >= budget:
                    break
                if time.time() > deadline - 10.0:
                    break
                for _ in range(3):
                    if calls_made >= budget:
                        break
                    y = np.clip(x + 0.1 * rng.standard_normal(dim), -5, 5)
                    fy = func(y)
                    calls_made += 1
                    if fy < fx:
                        x = y
                        fx = fy
                        update_best(x, fx)

        return best_x, best_f
