"""CMA-ES optimizer for BBOB f15 (Rastrigin rotated)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES: population-based ES with covariance adaptation for rotation/multimodality."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit
        d = self.dim
        budget = self.budget

        # CMA-ES parameters
        lam = max(4, 2 * d + 1)  # population size
        mu = lam // 2
        # Weights: rank-based, normalized
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2) ** 2 + mu_eff))
        chi_d = np.sqrt(d) * (1 - 1.0 / (4 * d) + 1.0 / (21 * d ** 2))

        rng = np.random.default_rng(42)

        # Initialize mean at origin (center of box), sigma large to cover box
        mean = np.zeros(d)
        sigma = 3.0  # covers most of [-5,5]
        C = np.eye(d)
        C_inv_sqrt = np.eye(d)  # C^{-1/2}
        pc = np.zeros(d)
        ps = np.zeros(d)
        best_x = mean.copy()
        best_f = func(best_x)
        n_calls = 1

        def time_ok():
            return time.time() < deadline

        for gen in range(budget // lam + 1):
            if not time_ok():
                break
            if n_calls + lam > budget:
                # Only evaluate as many as we can
                lam_eff = max(1, budget - n_calls)
                if lam_eff < 2:
                    break
            else:
                lam_eff = lam

            # Sample population
            pop = np.zeros((lam_eff, d))
            fvals = np.zeros(lam_eff)
            for i in range(lam_eff):
                if not time_ok():
                    break
                z = rng.standard_normal(d)
                x = mean + sigma * (C_inv_sqrt @ z)
                x = np.clip(x, -5, 5)
                pop[i] = x
                fvals[i] = func(x)
                n_calls += 1
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = x.copy()

            if lam_eff < 2:
                break

            # Sort by fitness
            order = np.argsort(fvals)
            pop_sel = pop[order]
            fvals_sel = fvals[order]

            # Use up to mu best
            mu_eff_gen = min(mu, lam_eff)
            w = weights[:mu_eff_gen]
            w = w / w.sum()

            # Weighted mean
            y_w = (w[:, None] * (pop_sel[:mu_eff_gen] - mean)).sum(axis=0)

            # Update evolution paths
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ y_w)
            norm_ps = np.linalg.norm(ps)
            h_sigma = (norm_ps / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) / chi_d < (1.4 + 2 / (d + 1)))
            pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w

            # Update mean
            new_mean = mean + sigma * y_w
            new_mean = np.clip(new_mean, -5, 5)

            # Update covariance
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = ((1 - c_1 - c_mu + delta_h) * C
                 + c_1 * np.outer(pc, pc)
                 + c_mu * np.sum(w[:, None, None] * np.outer(pop_sel[:mu_eff_gen] - mean, pop_sel[:mu_eff_gen] - mean), axis=0))

            # Ensure C is symmetric positive definite
            C = 0.5 * (C + C.T)
            try:
                C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C + 1e-10 * np.eye(d)))
            except np.linalg.LinAlgError:
                # Fallback: reset covariance
                C = np.eye(d)
                C_inv_sqrt = np.eye(d)

            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (norm_ps / chi_d - 1))
            sigma = np.clip(sigma, 1e-6, 10.0)

            mean = new_mean

        return best_x, best_f
# EVOLVE-BLOCK-END
