"""CMA-ES based optimizer for the BBOB-style Discus function (f11)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with restarts for the BBOB-style Discus function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')
        call_count = 0

        def try_eval(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def time_ok():
            return time.time() < deadline - 10

        def cma_es(x0, sigma0, budget_cma):
            """Run CMA-ES from x0 with initial step size sigma0 for up to budget_cma evaluations."""
            nonlocal call_count, best_f, best_x

            n = dim
            # CMA-ES parameters
            lambda_ = max(4, int(4 + 3 * np.log(n)))
            mu = lambda_ // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, lambda_ - mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights[:mu] ** 2)
            c_sigma = (mu_eff + 2) / (n + 4 + mu_eff)
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            c_cov_pos = (2 + 2 * min(mu_eff / (n + 2), 1)) / (n + 1) ** 1.8
            c_cov_neg = (2 + 2 * min(mu_eff / (n + 2), 1)) / (n + 1) ** 1.2

            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            C_inv = np.eye(n)
            C_sqrt = np.eye(n)
            C_sqrt_inv = np.eye(n)

            def update_chol():
                nonlocal C, C_inv, C_sqrt, C_sqrt_inv
                # Use eigendecomposition for stability
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                C_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T

            update_chol()

            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            count_success = 0
            evals_in_run = 0

            while call_count < budget and evals_in_run < budget_cma:
                if not time_ok():
                    break

                # Generate lambda_ candidates
                ys = [rng.standard_normal(n) for _ in range(lambda_)]
                xs = [x_mean + sigma * (C_sqrt @ y) for y in ys]

                # Evaluate all candidates
                fvals = []
                for x in xs:
                    fx = try_eval(x)
                    if fx is None:
                        break
                    fvals.append(fx)
                    evals_in_run += 1

                if len(fvals) < lambda_:
                    break

                # Sort by fitness
                indices = np.argsort(fvals)
                xs_sorted = [xs[i] for i in indices]
                fvals_sorted = [fvals[i] for i in indices]
                ys_sorted = [ys[i] for i in indices]

                # Update mean
                x_new = sum(weights[i] * xs_sorted[i] for i in range(mu))

                # Update step size (CSA)
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt_inv @ (x_new - x_mean) / sigma)

                # Update covariance
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x_mean) / sigma
                delta_hsig = (1 - np.exp(-c_sigma * d_sigma)) * np.linalg.norm(p_sigma) < (1.4 + 2 / (n + 1)) * np.sqrt(1 - (1 - c_sigma) ** (2 * (evals_in_run + 1)))
                rank_mu = sum(weights[i] * np.outer(ys_sorted[i], ys_sorted[i]) for i in range(mu))
                rank_one = np.outer(p_c, p_c)
                C = (1 - c_cov_pos * (delta_hsig and c_sigma * d_sigma or 0) - c_cov_neg * c_sigma * d_sigma + c_cov_pos) * C
                C += c_cov_neg * (rank_mu + (delta_hsig and 1 or 0) * rank_one)

                # Ensure C is symmetric
                C = (C + C.T) / 2

                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2) - 1))
                sigma = np.clip(sigma, 1e-12, 100)

                x_mean = x_new

                # Update Cholesky decomposition periodically
                if evals_in_run % max(1, lambda_ // 2) == 0:
                    update_chol()

                # Convergence check
                if np.linalg.norm(x_new - x0) < 1e-10 * sigma or sigma < 1e-12:
                    break

                # Stagnation check
                if evals_in_run > 50 and np.all(np.array(fvals_sorted[:10]) - fvals_sorted[0] < 1e-10):
                    break

            return x_mean, sigma

        # Phase 1: Initial CMA-ES from center
        x0 = np.zeros(dim)
        sigma0 = 2.0
        budget_phase1 = int(budget * 0.5)
        x1, sigma1 = cma_es(x0, sigma0, budget_phase1)

        # Phase 2: Try a few more restarts if budget remains
        restarts = [
            rng.uniform(-3, 3, dim),
            rng.uniform(-4, 4, dim),
            rng.uniform(-2, 2, dim),
        ]

        for rs in restarts:
            if call_count >= budget:
                break
            if not time_ok():
                break
            if best_x is not None and np.linalg.norm(rs - best_x) < 0.3:
                continue
            remaining = budget - call_count
            if remaining < 20:
                break
            budget_restart = min(remaining // 3, 100)
            x_r, sigma_r = cma_es(rs, 1.5, budget_restart)

        # Phase 3: Fine-tuning around best point
        if best_x is not None and call_count < budget:
            fine_budget = min(50, budget - call_count)
            x_fine = best_x.copy()
            sigma_fine = 0.05
            for _ in range(fine_budget):
                if not time_ok():
                    break
                if call_count >= budget:
                    break
                y = np.clip(x_fine + sigma_fine * rng.standard_normal(dim), -5, 5)
                fy = try_eval(y)
                if fy is None:
                    break
                if fy < best_f:
                    x_fine = y.copy()
                    sigma_fine *= 1.1
                else:
                    sigma_fine *= 0.9

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
