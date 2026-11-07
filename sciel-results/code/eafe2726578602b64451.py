"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with full covariance matrix adaptation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill

        rng = np.random.default_rng(0)
        n = self.dim
        # Population size: 4 + floor(3 * ln(n * ln(n)))
        # For n=5: 4 + floor(3 * ln(5 * ln(5))) = 4 + floor(3 * ln(5 * 1.609)) = 4 + floor(3 * 2.079) = 4 + 6 = 10
        # Let's use a slightly larger population for faster convergence in small budget
        # Standard CMA-ES suggests lambda = 4 + floor(3 * ln(n)). For n=5, lambda = 4 + floor(3*1.609) = 8.
        # Let's use lambda = 12 to balance exploration and exploitation.
        lam = 12

        # Initial mean: center of the box
        x = np.zeros(n)
        # Initial sigma: 1.0 is reasonable for [-5, 5]
        sigma = 1.0
        C = np.eye(n)
        ps = np.zeros(n)
        pc = np.zeros(n)

        # CMA-ES parameters
        cmu = 2.0 / ((n + 1.3) ** 2 + lam)
        cs = (lam / 2 + 1) / (n + lam / 2 + 5)
        damps = 1 + 2 * max(0, np.sqrt((lam / 2 - 1) / (n + 1)) - 1) + cs
        cc = (4 + lam / n) / (n + 4 + 2 * lam / n)
        chi_n = np.sqrt(n) * (1 - 1. / (4 * n) + 1. / (21 * n ** 2))

        # Generate initial population
        def sample():
            # Sample from N(x, sigma^2 * C)
            # Use Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, regularize
                C_reg = C + 1e-6 * np.eye(n)
                L = np.linalg.cholesky(C_reg)
            z = rng.standard_normal((lam, n))
            return x + sigma * np.dot(z, L.T)

        # Initial evaluation
        pop = sample()
        fvals = np.array([func(np.clip(p, -5, 5)) for p in pop])
        calls = lam

        # Sort indices by fitness
        idx = np.argsort(fvals)
        best_x = np.clip(pop[idx[0]], -5, 5)
        best_f = fvals[idx[0]]

        while calls < self.budget:
            if time.time() > deadline:
                break

            # Update mean
            w = np.log(lam / 2 + 1) - np.log(np.arange(1, lam + 1))
            w /= w.sum()
            mu = int(lam / 2)
            w = w[:mu]
            wmu = w.sum()
            mu_eff = 1.0 / np.sum(w ** 2)

            x_old = x.copy()
            x = x + np.dot(w, pop[idx[:mu]])

            # Update step size control vector ps
            # C^{-1/2} is needed. We can approximate or compute via eigendecomposition
            # For efficiency in small dim, we can use the Cholesky factor L
            # C^{-1/2} = L^{-T}
            try:
                L = np.linalg.cholesky(C)
                L_inv = np.linalg.inv(L)
            except np.linalg.LinAlgError:
                L_inv = np.linalg.inv(np.linalg.cholesky(C + 1e-6 * np.eye(n)))
            
            # ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mu_eff) * C^{-1/2} * (x - x_old) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * np.dot(L_inv.T, (x - x_old) / sigma)

            # Update covariance matrix control vector pc
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls + 1) / lam))) < (1.4 + 2 / (n + 1)) * chi_n
            pc = (1 - cc) * pc + hsig * cc * (x - x_old) / sigma

            # Update covariance matrix C
            v = np.dot(L_inv.T, (x - x_old) / sigma)
            C = (1 + cmu * (2 - hsig) - cmu) * C
            # Rank-mu update
            for i in range(mu):
                vi = np.dot(L_inv.T, (pop[idx[i]] - x_old) / sigma)
                C += cmu * w[i] * np.outer(vi, vi)
            
            # Ensure C is symmetric
            C = (C + C.T) / 2

            # Update sigma
            sigma = sigma * np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
            # Clip sigma to reasonable range
            sigma = np.clip(sigma, 1e-5, 10.0)

            # Generate new population
            pop = sample()
            fvals = np.array([func(np.clip(p, -5, 5)) for p in pop])
            calls += lam

            # Update best
            idx = np.argsort(fvals)
            if fvals[idx[0]] < best_f:
                best_f = fvals[idx[0]]
                best_x = np.clip(pop[idx[0]], -5, 5)

        return best_x, best_f
# EVOLVE-BLOCK-END
