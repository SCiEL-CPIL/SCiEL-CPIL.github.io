"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with adaptive population and restarts for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit
        rng = np.random.default_rng(123)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf

        def time_check():
            return time.time() < deadline

        # Phase 1: Multi-start with small CMA-ES runs
        # Use lambda=8 for small population, restart every ~40 evals if no improvement
        
        num_starts = 6
        evals_per_start = budget // num_starts  # ~83 evals per start
        remaining = budget

        for s in range(num_starts):
            if not time_check() or remaining <= 0:
                break

            # Initialize center: first start random, subsequent near best or random
            if s == 0:
                center = rng.uniform(-4, 4, dim)
            else:
                if best_x is not None and rng.random() < 0.5:
                    center = best_x + rng.standard_normal(dim) * 0.5
                else:
                    center = rng.uniform(-4, 4, dim)

            # CMA-ES parameters
            lam = 8
            mu = lam // 2
            # Weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mu_eff = 1.0 / (w @ w)
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

            # Initial sigma
            sigma = 2.0
            C = np.eye(dim)
            C_sqrt = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)

            # Evaluate initial center
            fc = func(center)
            if fc < best_f:
                best_f = fc
                best_x = center.copy()
            remaining -= 1

            local_best = fc
            local_best_x = center.copy()
            no_improve_count = 0
            max_no_improve = 25

            # Generate initial population
            for i in range(lam):
                if not time_check() or remaining <= 0:
                    break
                z = rng.standard_normal(dim)
                x_i = center + sigma * (C_sqrt @ z)
                x_i = np.clip(x_i, -5, 5)
                f_i = func(x_i)
                remaining -= 1

                if f_i < best_f:
                    best_f = f_i
                    best_x = x_i.copy()
                if f_i < local_best:
                    local_best = f_i
                    local_best_x = x_i.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1

            # Store for sorting
            pop_x = [x_i]
            pop_f = [f_i]

            # Main CMA-ES loop
            while remaining > 0 and time_check():
                # Sort by fitness
                indices = np.argsort(pop_f)
                pop_x = [pop_x[i] for i in indices]
                pop_f = [pop_f[i] for i in indices]

                # Check for stagnation
                if no_improve_count >= max_no_improve:
                    break

                # Update mean
                y_w = np.zeros(dim)
                for i in range(mu):
                    y_w += w[i] * (pop_x[i] - center)
                center = center + y_w

                # Update step size control
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt.T @ (y_w / sigma))
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (remaining - lam)))) < (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1))
                cs = np.linalg.norm(ps)

                if cs < (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1)):
                    sigma *= np.exp(c_sigma * (cs / np.sqrt((dim + 1) / (dim - 1)) - 1))

                # Update covariance matrix
                pc = (1 - c_c) * pc + hsig * c_c * y_w
                delta_hsig = (1 - hsig) * c_c * (2 - c_c)
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + delta_hsig * C)
                for i in range(mu):
                    y_i = (pop_x[i] - center) / sigma
                    C += c_mu * w[i] * np.outer(y_i, y_i)

                # Ensure PSD and update square root
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))

                # Generate new population
                new_pop_x = []
                new_pop_f = []
                improved = False

                for i in range(lam):
                    if not time_check() or remaining <= 0:
                        break
                    z = rng.standard_normal(dim)
                    x_i = center + sigma * (C_sqrt @ z)
                    x_i = np.clip(x_i, -5, 5)
                    f_i = func(x_i)
                    remaining -= 1
                    new_pop_x.append(x_i)
                    new_pop_f.append(f_i)

                    if f_i < best_f:
                        best_f = f_i
                        best_x = x_i.copy()
                        improved = True
                    if f_i < local_best:
                        local_best = f_i
                        local_best_x = x_i.copy()
                        improved = True

                if improved:
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                pop_x = new_pop_x
                pop_f = new_pop_f

                # Early stop if sigma too small
                if sigma < 1e-6:
                    break

        # Phase 2: Local refinement around best point
        if best_x is not None and remaining > 10 and time_check():
            # Use a simple (1+1)-ES with small sigma for local refinement
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            for _ in range(min(remaining, 30)):
                if not time_check():
                    break
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                remaining -= 1
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.2
                else:
                    sigma *= 0.7

        return best_x, best_f
# EVOLVE-BLOCK-END
