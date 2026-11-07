"""Rewritten optimizer for BBOB-style f14 (Different powers)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    CMA-ES optimizer with adaptive step size, boundary handling,
    and a wall-clock deadline guard.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 275.0  # safety margin within 300s limit

        rng = np.random.default_rng(42)

        # Initial point: center of the box
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        call_count = 1

        # CMA-ES parameters
        lam = 4 + int(3 * np.log(dim))  # population size
        sigma = 2.0  # initial step size
        mu = lam // 2  # number of parents
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # State variables
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        chi_n = np.sqrt(dim) * (1 + 1 / (2 * dim) + 1 / (8 * dim ** 2))

        while call_count < budget:
            if time.time() > deadline:
                break

            # Generate offspring
            offspring = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = x + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                fy = func(y)
                call_count += 1
                if call_count >= budget:
                    break
                offspring.append((fy, y))

            if len(offspring) < mu:
                break

            # Sort offspring by fitness
            offspring.sort(key=lambda t: t[0])

            # Update best
            if offspring[0][0] < best_fx:
                best_fx = offspring[0][0]
                best_x = offspring[0][1].copy()

            # Select parents
            parents = [y for _, y in offspring[:mu]]

            # Weighted mean of parents
            m_new = sum(weights[i] * parents[i] for i in range(mu))

            # Update evolution paths
            pc = (1 - c_sigma) * pc + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (B @ (D * (m_new - x)))
            norm_pc = np.linalg.norm(pc)
            h_sigma = (norm_pc / np.sqrt(1 - (1 - c_sigma) ** (call_count + 1))) < (1 + c_sigma) * chi_n
            ps = (1 - c_sigma) * ps + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (m_new - x) / sigma

            # Update covariance matrix
            delta_h = (1 - h_sigma) * c_c * (1 - c_c)
            diff = (m_new - x)
            C = B @ (D ** 2) @ B.T
            C = (1 + c_1 + c_mu * (h_sigma - 1)) * C
            C += c_1 * (np.outer(pc, pc) - np.eye(dim))
            C += c_mu * (np.outer(ps, ps) - (1 - h_sigma) * C)

            # Eigen-decomposition for C
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-20, None)
            D = np.sqrt(eigvals)
            B = eigvecs

            # Update step size
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 50.0)

            # Update mean
            x = m_new

        return best_x, best_fx
# EVOLVE-BLOCK-END
