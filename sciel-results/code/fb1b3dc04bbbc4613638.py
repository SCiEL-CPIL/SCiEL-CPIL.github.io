"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB f4 (Büche-Rastrigin)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        deadline = time.time() + 290.0

        # CMA-ES parameters
        lam = 4 + int(np.floor(3 * np.log(n)))
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = (2 + 2 * (mu_eff - 1) / (n + 1)) / ((n + 2) ** 2 + mu_eff)
        c_mu = min(1 - c_1, (2 * (weights[0] - 0.5) + mu_eff / (n + 2)) *
                   ((n + 2) / (n + 1) - 1) * (1 - c_1))

        # initialization
        x_mean = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)  # covariance matrix
        B = np.eye(n)  # eigenvectors
        D = np.ones(n)  # eigenvalues
        ps = np.zeros(n)
        pc = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

        # evaluate initial point
        x_eval = np.clip(x_mean, -5, 5)
        fx = func(x_eval)
        best_x = x_eval.copy()
        best_fx = fx
        calls = 1

        for gen in range(budget):
            if time.time() > deadline:
                break

            # generate offspring
            zs = np.random.standard_normal((lam, n))
            ys = np.zeros((lam, n))
            fs = np.zeros(lam)

            for i in range(lam):
                y = x_mean + sigma * (B @ (D * zs[i]))
                y_clipped = np.clip(y, -5, 5)
                ys[i] = y_clipped
                fs[i] = func(y_clipped)
                calls += 1
                if fs[i] < best_fx:
                    best_fx = fs[i]
                    best_x = y_clipped.copy()

            if time.time() > deadline:
                break

            # sort by fitness
            idx = np.argsort(fs)
            ys_sorted = ys[idx]
            fs_sorted = fs[idx]

            # update best
            if fs_sorted[0] < best_fx:
                best_fx = fs_sorted[0]
                best_x = ys_sorted[0].copy()

            # recombination
            y_recomb = np.zeros(n)
            for i in range(mu):
                y_recomb += weights[i] * ys_sorted[i]

            # update path evolution strategies
            x_old = x_mean.copy()
            x_mean = y_recomb

            # update ps
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (B @ (D * ((y_recomb - x_old) / sigma)))

            # update pc
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) < (1.4 + 2 / (n + 1)) * chi_n:
                pc = (1 - c_c) * pc + c_c * ((y_recomb - x_old) / sigma)

            # update covariance matrix
            delta_sig = 1.0 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) < (1.4 + 2 / (n + 1)) * chi_n else 0.0
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                diff = (ys_sorted[i] - x_old) / sigma
                rank_mu += weights[i] * np.outer(diff, diff)

            C = (1 + c_1 * (delta_sig - 1) - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (delta_sig - 1) * np.eye(n)) + c_mu * rank_mu

            # ensure positive semi-definite
            C = (C + C.T) / 2

            # update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 1e6)

            # update B, D from C
            try:
                D_sq, B = np.linalg.eigh(C)
                D_sq = np.clip(D_sq, 1e-20, None)
                D = np.sqrt(D_sq)
            except np.linalg.LinAlgError:
                C = np.eye(n)
                B = np.eye(n)
                D = np.ones(n)

        return best_x, best_fx
# EVOLVE-BLOCK-END
