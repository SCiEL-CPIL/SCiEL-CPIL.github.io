"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)

        # Population size
        lam = 4 + int(3 * np.log(n))
        # Initial step size
        sigma = 2.0
        # Initial mean
        m = np.zeros(n)
        # Initial covariance
        C = np.eye(n)
        C_inv = np.eye(n)
        C_sqrt = np.eye(n)

        # Evolution path parameters
        c_sigma = (lam // 2 + 1) / (n + 4)
        c_c = (4 + lam / n) / (n + 4 + 2 * lam / n)
        c_1 = 2 / ((n + 1.3) ** 2 + lam)
        c_mu = min(1 - c_1, 2 * ((lam // 2 + 1) - 2) / ((n + 2) ** 2 + lam))

        # Initial weights
        w = np.log(lam / 2 + 0.5) - np.log(np.arange(1, lam + 1))
        w = w / w.sum()
        mueff = 1 / np.sum(w ** 2)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # Evolution paths
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)

        # Initial evaluation
        x = m
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        # Generation loop
        while calls < budget:
            # Generate population
            offspring = []
            offspring_f = []
            for i in range(lam):
                z = rng.standard_normal(n)
                y = m + sigma * (C_sqrt @ z)
                # Reflect if outside bounds
                for d in range(n):
                    if y[d] < -5:
                        y[d] = -5 + (-5 - y[d])
                    elif y[d] > 5:
                        y[d] = 5 - (y[d] - 5)
                y = np.clip(y, -5, 5)
                offspring.append(y)
                fy = func(y)
                offspring_f.append(fy)
                calls += 1
                if calls >= budget:
                    break

            if calls >= budget:
                break

            offspring = np.array(offspring)
            offspring_f = np.array(offspring_f)

            # Sort by fitness
            order = np.argsort(offspring_f)
            offspring = offspring[order]
            offspring_f = offspring_f[order]

            # Update best
            if offspring_f[0] < best_fx:
                best_fx = offspring_f[0]
                best_x = offspring[0].copy()

            # Weighted mean of selected points
            mu_sel = min(mueff, lam // 2)
            w_sel = w[:mu_sel]
            w_sel = w_sel / w_sel.sum()
            m_new = np.sum(w_sel[:, None] * offspring[:mu_sel], axis=0)

            # Update evolution path for sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_inv @ (m_new - m) / sigma)
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (calls - 1)) < (1.4 + 2 / (n + 1)) * chi_n)
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 1e12)

            # Update evolution path for C
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * (m_new - m) / sigma

            # Rank-1 and rank-mu updates
            C = (1 + c_1 * (2 - c_1) - c_1) * C + c_1 * np.outer(p_c, p_c) + (1 - h_sigma) * c_1 * np.outer(p_sigma, p_sigma)
            # Actually, the standard update is:
            # C = (1 - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c) + c_mu * sum(w_i * np.outer(y_i, y_i))
            # Let me correct this
            C = (1 - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c)
            for i in range(mu_sel):
                y_i = (offspring[i] - m) / sigma
                C += c_mu * w_sel[i] * np.outer(y_i, y_i)

            # Ensure positive semi-definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update inverse and square root
            C_inv = np.linalg.inv(C)
            C_sqrt = np.linalg.cholesky(C)

            m = m_new

        return best_x, best_fx
# EVOLVE-BLOCK-END
