"""Rewritten optimizer: CMA-ES style population-based search with deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Population-based CMA-ES style continuous optimizer on [-5,5]^dim."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        # CMA-ES parameters
        N = max(4, int(4 + 3 * np.log(dim)))  # population size
        sigma = 2.5  # initial step size
        cc = (4 + dim / (dim + 4)) / ((dim + 1.7) + (4 + dim / (dim + 4)))
        cmu = min(0.05, (2 + 2.0 / (dim + 1.2)) / ((dim + 1.7) + (4 + dim / (dim + 4)) + 0.01))
        c1 = (2 + mu_eff / (dim * dim)) / ((dim + 1.3) ** 2 + mu_eff if False else (dim + 1.3) ** 2)
        # mu_eff for CMA-ES weights
        mus = N // 2
        w = np.log(mus + 0.5) - np.log(np.arange(1, mus + 1))
        w /= w.sum()
        mu_eff = 1.0 / (w @ w)
        c1 = (2 + mu_eff / (dim * dim)) / ((dim + 1.3) ** 2)
        cmu = min(0.05, (2 + 2.0 / (dim + 1.2)) / ((dim + 1.7) ** 2 + mu_eff))

        # Initial mean: center of box
        m = np.zeros(dim)
        C = np.eye(dim)
        Cinv = np.eye(dim)
        Csqr = np.eye(dim)
        # Precompute sqrt of C
        eigval, eigvec = np.linalg.eigh(C)
        Csqrt = eigvec @ np.diag(np.sqrt(np.maximum(eigval, 1e-12))) @ eigvec.T

        ps = np.zeros(dim)
        cs = 1.0 / (N + 4 + 2 * dim / (dim - 1))
        hs = (1 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1))

        # Evaluate initial mean
        m_clip = np.clip(m, -5, 5)
        fm = func(m_clip)
        best_x = m_clip.copy()
        best_f = fm
        calls = 1

        # Adaptive sigma with success rule
        sigma_target = 2.5
        sigma_min = 1e-8
        sigma_max = 10.0

        while calls < budget:
            # Time check
            if time.time() > deadline:
                break

            # Generate population
            Y = np.zeros((N, dim))
            for i in range(N):
                Y[i] = rng.standard_normal(dim)

            # Update evolution path
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (Cinv @ (w @ Y.T).T).T
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls + 1))) / np.sqrt(mu_eff)
                    < (1 + 2 / (dim + 1)) * hs)

            # Update covariance
            delta_h = (hsig - 1) * cs * (2 - cs)
            for i in range(N):
                yi = Y[i]
                C = (1 - c1 - cmu + delta_h) * C + c1 * np.outer(yi, yi)
                C += cmu * (w @ Y.T)[:, i][:, None] * Y[i, None, :]

            # Ensure positive definiteness
            eigval, eigvec = np.linalg.eigh(C)
            eigval = np.maximum(eigval, 1e-12)
            C = eigvec @ np.diag(eigval) @ eigvec.T
            Csqrt = eigvec @ np.diag(np.sqrt(eigval)) @ eigvec.T

            # Sample candidates
            X = np.zeros((N, dim))
            for i in range(N):
                X[i] = m + sigma * Csqrt @ Y[i]
                X[i] = np.clip(X[i], -5, 5)

            # Evaluate
            F = np.zeros(N)
            for i in range(N):
                if calls >= budget:
                    break
                F[i] = func(X[i])
                calls += 1
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = X[i].copy()

            if calls >= budget:
                break

            # Sort by fitness
            order = np.argsort(F)
            Y = Y[order]
            F = F[order]

            # Update mean
            m_old = m.copy()
            m = m + sigma * (w @ Y[:mus, :].T).T

            # Adapt sigma
            sigma_ratio = np.linalg.norm(m - m_old) / (sigma * np.linalg.norm(Csqrt @ np.ones(dim)) + 1e-32)
            sigma *= np.exp(cs * (sigma_ratio - 1) / (dim * 0.5 + 1))
            sigma = np.clip(sigma, sigma_min, sigma_max)

            # Keep mean in box
            m = np.clip(m, -5, 5)

        return best_x, best_f
# EVOLVE-BLOCK-END
