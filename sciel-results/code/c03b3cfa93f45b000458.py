"""CMA-ES based black-box optimizer for BBOB f4 (Büche-Rastrigin)."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np


class Optimizer:
    """CMA-ES with deadline guard for BBOB f4 on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        # --- Population size ---
        lam = max(4, int(4 + 2 * math.log(dim)))
        if lam > 20:
            lam = 20

        # --- Initial state ---
        # Start at origin; minimizer lies in [-4,4]^5
        mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        D = np.ones(dim)
        B = np.eye(dim)

        # Step size control
        cs = (lam / 2 + 1) / (dim + lam / 2 + 3)
        ds = 1 + 2 * max(0, math.sqrt((lam / dim - 1) / (dim + 1)) - 1) + cs
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        count_eval = 0

        # Evaluate initial mean
        x0 = np.clip(mean, -5, 5)
        f0 = func(x0)
        count_eval += 1
        best_x = x0.copy()
        best_f = f0

        # Weights
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1.0 / w.sum()
        c1 = (2 * (mueff - 2 + 2 / dim)) / ((dim + 2) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / dim)) / ((dim + 2) ** 2 + mueff)

        # --- Main loop ---
        while count_eval < budget:
            if time.time() > deadline - 2.0:
                break

            # Generate population
            offspring = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                step = B @ (D * z)
                x = mean + sigma * step
                x = np.clip(x, -5, 5)
                offspring.append(x)

            # Evaluate offspring
            values = []
            for x in offspring:
                if count_eval >= budget:
                    break
                fx = func(x)
                values.append(fx)
                count_eval += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

            if len(values) < mu:
                break

            # Sort
            idx = np.argsort(values)
            x_sorted = [offspring[i] for i in idx]
            f_sorted = [values[i] for i in idx]

            # Update mean
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += w[i] * (x_sorted[i] - mean) / sigma
            mean = mean + sigma * y_w
            mean = np.clip(mean, -5, 5)

            # Update step size (CSA)
            y_sigma = B @ (D * (B.T @ (D * (B.T @ y_w))))
            ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * y_sigma
            hsig = (np.linalg.norm(ps) / math.sqrt(1 - (1 - cs) ** (count_eval / (cs * (dim + 1))))) < \
                   (1.4 + 2 / (dim + 1)) * math.sqrt((dim + 1) / (dim - 1))
            sigma = sigma * math.exp((cs / ds) * (np.linalg.norm(ps) / math.sqrt(dim) - 1))
            sigma = np.clip(sigma, 1e-12, 50.0)

            # Update covariance
            delta_hsig = (1 - hsig) * cs * (2 - cs)
            pc = (1 - cmu) * pc + delta_hsig * (y_w + delta_hsig * ps)
            rank1 = np.outer(pc, pc)
            rankmu = np.zeros((dim, dim))
            for i in range(mu):
                zi = (x_sorted[i] - (mean - sigma * y_w)) / sigma
                rankmu += w[i] * np.outer(zi, zi)
            C = (1 + c1 * (2 - c1 - delta_hsig) + cmu) * C - c1 * rank1 - cmu * rankmu
            C = _ensure_pd(C, dim)
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            D = np.sqrt(eigvals)
            B = eigvecs

            if sigma < 1e-10:
                break

        return best_x, best_f


def _ensure_pd(C, dim):
    """Ensure C is positive definite by adding small regularization if needed."""
    eigvals, eigvecs = np.linalg.eigh(C)
    min_eig = np.min(eigvals)
    if min_eig < 1e-12:
        C = C + (1e-12 - min_eig) * np.eye(dim)
    return C
# EVOLVE-BLOCK-END
