"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with adaptive covariance matrix and step size."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin before 330s kill

        # CMA-ES parameters
        lambda_ = 4 * int(1 + 3 * n)  # population size
        mu = lambda_ // 2             # number of parents
        # weights (logarithmic)
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        # step size adaptation
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        # covariance adaptation
        cmu = min(1 - 1 / (n * n), 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        c1 = (5 - n + mueff) / ((n + 2) ** 3 + mueff)
        c1 = max(c1, 0.1 * cmu)
        cov_update = c1 + 2 * cmu * (1 - c1 - 0.5 * c1)
        # sigma init
        sigma = 2.0
        hs = (1 + 0.35) ** (1 / (budget * damps)) - 1
        # mean
        m = np.zeros(n)
        C = np.eye(n)
        Cinv_sqrt = np.eye(n)
        # eigen decomposition cache
        eigval = np.ones(n)
        eigvec = np.eye(n)
        # path variables
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        # best tracking
        best_x = m.copy()
        best_f = func(m)
        n_calls = 1

        def time_left():
            return time.time() < deadline

        # Precompute inverse sqrt of C via eigendecomposition
        def update_eig():
            nonlocal eigval, eigvec, Cinv_sqrt
            eigval, eigvec = np.linalg.eigh(C)
            eigval = np.maximum(eigval, 1e-20)
            Cinv_sqrt = eigvec @ np.diag(1.0 / np.sqrt(eigval)) @ eigvec.T

        update_eig()

        # Main loop: generate population, evaluate, update
        while n_calls < budget and time_left():
            # Generate population
            ys = np.zeros((lambda_, n))
            for i in range(lambda_):
                z = np.random.randn(n)
                y = m + sigma * (Cinv_sqrt @ z)
                ys[i] = np.clip(y, -5, 5)
                n_calls += 1
                if n_calls >= budget:
                    break
            if n_calls >= budget:
                break

            # Evaluate population
            fs = np.empty(lambda_)
            for i in range(lambda_):
                fs[i] = func(ys[i])
                n_calls += 1
                if not time_left():
                    # fill rest with worst
                    fs[i:] = np.inf
                    break

            # Track best
            fmin = np.min(fs)
            if fmin < best_f:
                best_f = fmin
                idx_best = np.argmin(fs)
                best_x = ys[idx_best].copy()

            # Sort by fitness
            order = np.argsort(fs)
            ys_sorted = ys[order]
            fs_sorted = fs[order]

            # Update mean
            m_new = m + sigma * np.sum(w[:, None] * (ys_sorted[:mu] - m)[:, :mu].T, axis=1)
            m_new = np.clip(m_new, -5, 5)

            # Update sigma (path)
            y_sigma = Cinv_sqrt.T @ (m_new - m) / sigma
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * y_sigma
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) - 1 / (24 * n ** 2))
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * (budget - n_calls))) < (1.4 + 2 / (n + 1)) * chi_n:
                sigma *= np.exp(cs / damps * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 1e6)

            # Update covariance
            y_c = Cinv_sqrt.T @ (m_new - m) / sigma
            p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * mueff) * y_c
            delta_hs = (np.linalg.norm(p_c) / np.sqrt(1 - (1 - cc) ** (budget - n_calls))) ** 2 - (1 + 2 / (n + 1)) * chi_n
            C = (1 + cov_update * (cmu * delta_hs - 1)) * C \
                + c1 * (np.outer(p_c, p_c) + (1 - cc) * (1 - cc) * C) \
                + 2 * cmu * np.sum(w[:, None] * np.outer(ys_sorted[:mu] - m, (ys_sorted[:mu] - m).T), axis=0) / sigma ** 2

            # Symmetrize
            C = 0.5 * (C + C.T)

            # Update eigendecomposition
            update_eig()

            m = m_new

        return best_x, best_f
# EVOLVE-BLOCK-END
