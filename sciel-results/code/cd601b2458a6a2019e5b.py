"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with restarts and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        # CMA-ES parameters
        lambda_ = 4 * int(1 + 3 * dim)  # population size
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = (2 + 2 * min(1, (mu_eff - 1) / dim) / dim) / (
            (dim + 1.3) ** 2 + mu_eff)
        c_mu = min(
            1 - c_1,
            2 * (
                (sqrt_mu_eff := np.sqrt(mu_eff)) - 1) /
            ((dim + 2) ** 2 - mu_eff))

        def run_cmaes(x0, f0, sigma0, max_evals):
            x = x0.copy()
            f_best = f0
            sigma = sigma0
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            C = np.eye(dim)
            pc_norm = np.sqrt(dim)
            chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

            # Precompute
            B = np.linalg.eigh(C)[1]
            D = np.sqrt(np.linalg.eigvalsh(C))
            inv_sqrt_D = 1.0 / D

            evals_used = 0
            for gen in range(max_evals // lambda_ + 1):
                if time.time() > deadline:
                    break
                # Generate population
                Y = np.array([x + sigma * (B @ (D * rng.standard_normal(dim)))
                              for _ in range(lambda_)])
                F = np.array([func(y) for y in Y])
                evals_used += lambda_

                # Sort by fitness
                order = np.argsort(F)
                Y_w = Y[order[:mu]]
                F_w = F[order[:mu]]

                # Update best
                if F_w[0] < f_best:
                    f_best = F_w[0]
                    x = Y_w[0].copy()

                # Mean of selected
                m = np.sum(weights[:, None] * Y_w, axis=0)
                # CMA step
                step = m - x
                # Update p_sigma
                p_sigma = ((1 - c_sigma) * p_sigma +
                           (np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / D) *
                           (B.T @ step))
                # Update p_c
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))) \
                    < (1.4 + 2 / (dim + 1)) * chi_n
                h_sigma = float(h_sigma)
                p_c = ((1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * step / sigma)

                # Update C
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                C = ((1 + c_1 * delta_h - c_1 - c_mu) * C +
                     c_1 / dim * np.outer(p_c, p_c) +
                     c_mu / (D * D)[:, None] * (B.T @ np.sum(weights[:, None] * Y_w, axis=0) / sigma).reshape(-1, 1) *
                     np.outer(B.T @ (step / sigma), np.ones(1)) * 0 +  # placeholder
                     c_mu * np.sum(weights[:, None] * np.outer(Y_w - x, Y_w - x) / sigma ** 2, axis=0))

                # Actually let's do the standard update properly
                # Recompute eigenvalues
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                # Update sigma
                sigma = sigma * np.exp(c_sigma / d_sigma *
                                       (np.linalg.norm(p_sigma) / chi_n - 1))

                # Update x
                x = m

                # Recompute B, D for next iteration
                B = eigvecs
                D = np.sqrt(eigvals)

                if evals_used >= max_evals:
                    break

            return x, f_best, evals_used

        # Restart strategy
        best_x = None
        best_f = np.inf
        total_evals = 0

        # Initial random starts
        rng = np.random.default_rng(42)

        # First, do a quick random search to get a good starting point
        n_random = min(20, budget // 10)
        for i in range(n_random):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            total_evals += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Now run CMA-ES from the best point
        remaining = budget - total_evals
        if remaining > lambda_:
            x, f, used = run_cmaes(best_x, best_f, 2.0, remaining)
            total_evals += used
            if f < best_f:
                best_f = f
                best_x = x

        # If we still have budget, try a few more restarts from different points
        restarts = 3
        for r in range(restarts):
            if time.time() > deadline or total_evals >= budget:
                break
            # Start from a perturbation of best or a new random point
            if r == 0:
                x0 = best_x + 1.0 * rng.standard_normal(dim)
                x0 = np.clip(x0, -5, 5)
            else:
                x0 = rng.uniform(-5, 5, dim)
            f0 = func(x0)
            total_evals += 1
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

            remaining = budget - total_evals
            if remaining > lambda_:
                # Use smaller sigma for exploitation
                sigma0 = 1.0 if r == 0 else 2.0
                x, f, used = run_cmaes(x0, f0, sigma0, remaining)
                total_evals += used
                if f < best_f:
                    best_f = f
                    best_x = x

        return best_x, best_f
# EVOLVE-BLOCK-END
