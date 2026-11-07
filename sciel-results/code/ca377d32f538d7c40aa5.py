"""Optimizer for BBOB f22 (Gallagher 21 peaks) with CMA-ES + multi-start."""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Track best solution and value
        best_x = None
        best_f = np.inf
        calls_made = 0

        def check_deadline():
            return time.time() > deadline

        # Phase 1: Multi-start scan to find promising basins
        # Use a portion of budget for exploration
        n_scan = min(50, budget // 10)
        scan_starts = []
        # Generate diverse starting points
        for i in range(n_scan):
            if check_deadline():
                break
            # Mix of random and structured starts
            if i < dim + 1:
                # Structured: corners and center
                if i == 0:
                    x0 = np.zeros(dim)
                else:
                    x0 = np.zeros(dim)
                    x0[i - 1] = (-4.0 if (i // dim) % 2 == 0 else 4.0)
            else:
                x0 = rng.uniform(-4, 4, dim)
            scan_starts.append(x0)

        # Evaluate scan starts
        scan_results = []
        for x0 in scan_starts:
            if check_deadline():
                break
            fx = func(x0)
            calls_made += 1
            scan_results.append((fx, x0.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x0.copy()

        # Phase 2: CMA-ES refinement from best scan point
        # Allocate remaining budget to CMA-ES
        remaining_budget = budget - calls_made
        if remaining_budget > 10 and best_x is not None:
            x = best_x.copy()
            fx = best_f
            # Initial sigma: moderate step size
            sigma = 1.5
            # CMA-ES parameters
            lam = max(4, int(np.ceil(4 + 3 * np.log(dim))))  # population size
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            # State
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            step_size = sigma
            count_steps = 0

            # Precompute square root of C using Cholesky
            def _chol(C):
                try:
                    L = np.linalg.cholesky(C)
                    return L
                except np.linalg.LinAlgError:
                    # Fallback: eigenvalue decomposition
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-20)
                    L = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                    return L

            L = _chol(C)

            while calls_made < budget:
                if check_deadline():
                    break

                # Generate population
                offspring = []
                for i in range(lam):
                    if check_deadline():
                        break
                    z = rng.standard_normal(dim)
                    y = x + step_size * (L @ z)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls_made += 1
                    offspring.append((fy, y))
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()

                if len(offspring) < lam:
                    break

                # Sort by fitness
                offspring.sort(key=lambda t: t[0])

                # Update mean
                x_new = np.sum([weights[i] * offspring[i][1] for i in range(mu)], axis=0)

                # Update evolution paths
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * step_size / np.linalg.norm(L) * (L @ (x_new - x))
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (x_new - x) / step_size

                # Update covariance
                delta_sig = (np.linalg.norm(ps) / chi_n - 1) * c_sigma
                C = (1 + c_1 * delta_sig - c_1 - c_mu) * C + \
                    c_1 * np.outer(pc, pc) + \
                    c_mu / step_size ** 2 * np.sum([weights[i] * np.outer(offspring[i][1] - x, offspring[i][1] - x) for i in range(mu)], axis=0)

                # Ensure positive definite
                eigvals, eigvecs = np.linalg.eigh(C)
                if np.min(eigvals) < 1e-10:
                    eigvals = np.maximum(eigvals, 1e-10)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                L = _chol(C)

                # Update step size (1/5 rule-like adaptation)
                h_sigma = (1 + 2 * min(1, (np.linalg.norm(ps) / chi_n - 1)) * (1 - c_sigma))
                if h_sigma > 0.999:
                    step_size *= np.exp(c_sigma * (np.linalg.norm(ps) / chi_n - 1))
                step_size = np.clip(step_size, 1e-10, 10.0)

                # Accept new mean
                x = x_new
                fx = offspring[0][0]

                # Early stopping if convergence
                if np.linalg.norm(x_new - x) < 1e-10:
                    break

        # Phase 3: Local refinement with small steps around best solution
        if best_x is not None and not check_deadline():
            # Use remaining budget for fine-tuning
            local_sigma = 0.1
            for _ in range(budget - calls_made):
                if check_deadline():
                    break
                y = np.clip(best_x + local_sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    local_sigma *= 1.2
                else:
                    local_sigma *= 0.9

        return best_x, best_f
