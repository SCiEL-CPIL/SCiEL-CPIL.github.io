"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        rng = np.random.default_rng(12345)

        # Initial population size
        lam = max(4, int(4 + 3 * np.log(dim)))
        mu = lam // 2

        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))

        # Initial point: try center first, then use best of a few random starts
        x0 = np.zeros(dim)
        f0 = func(x0)
        calls = 1

        # Try a few random points to get a better starting point
        for _ in range(min(5, budget - 1)):
            if time.time() > deadline:
                break
            x_try = rng.uniform(-5, 5, dim)
            f_try = func(x_try)
            calls += 1
            if f_try < f0:
                x0, f0 = x_try, f_try

        x = x0.copy()
        fx = f0
        best_x = x.copy()
        best_fx = fx

        # Initial step size
        sigma = 2.0

        # CMA-ES state
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (20 * dim ** 2))

        while calls < budget:
            if time.time() > deadline:
                break

            # Generate population
            B = rng.standard_normal((lam, dim))
            # Whitening transformation: C^{1/2}
            # Use Cholesky decomposition
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback: use identity if Cholesky fails
                L = np.eye(dim)
                C = np.eye(dim)

            pops = np.array([x + sigma * (B[i] @ L.T) for i in range(lam)])
            pops = np.clip(pops, -5, 5)

            # Evaluate population
            fvals = np.zeros(lam)
            for i in range(lam):
                if time.time() > deadline:
                    fvals[i] = fvals[i-1] if i > 0 else fx
                    break
                fvals[i] = func(pops[i])
                calls += 1

            # Sort by fitness
            order = np.argsort(fvals)
            pops = pops[order]
            fvals = fvals[order]

            # Update best
            if fvals[0] < best_fx:
                best_fx = fvals[0]
                best_x = pops[0].copy()

            # Mean of best mu solutions
            m_new = np.sum(w[:, None] * pops[:mu], axis=0)

            # Update evolution path for covariance
            C_inv_sqrt = np.linalg.inv(L)  # C^{-1/2}
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * C_inv_sqrt @ (m_new - x)
            norm_ps = np.linalg.norm(ps)

            # Update covariance matrix
            delta_hs = (norm_ps / chi_n) > (1.4 + 2 / (dim + 1))
            pc = (1 - c_c) * pc + delta_hs * c_c * (m_new - x) / sigma

            # Rank-one and rank-mu updates
            C = (1 + c_1 * delta_hs * (2 - c_c) + c_mu) * C \
                - (c_1 + c_mu) * C \
                + (c_1 + c_mu * delta_hs) * np.outer(pc, pc)
            # Rank-mu update
            for i in range(mu):
                zi = C_inv_sqrt @ (pops[i] - x) / sigma
                C += c_mu * w[i] * np.outer(zi, zi)

            # Ensure C is positive semi-definite
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                if np.any(eigvals < 0):
                    eigvals = np.maximum(eigvals, 1e-10)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            except np.linalg.LinAlgError:
                C = np.eye(dim)

            # Update step size (CSA)
            sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)

            x = m_new

            # Update current best
            if fx > fvals[0]:
                fx = fvals[0]
                x = pops[0].copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
