"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer for BBOB f1 (Sphere)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit

        # Count function calls
        calls = [0]

        def wrapped(x):
            calls[0] += 1
            x = np.asarray(x, dtype=np.float64)
            # Clip to box
            x = np.clip(x, -5.0, 5.0)
            return float(func(x))

        # Initial point
        rng = np.random.default_rng(42)
        x0 = rng.uniform(-4, 4, dim)
        f0 = wrapped(x0)
        best_x = x0.copy()
        best_f = f0

        # CMA-ES parameters
        # Population size
        lam = max(4, int(4 + 3 * np.log(dim)))
        # Step size
        sigma = 2.0
        # Mean
        m = x0.copy()

        # Covariance matrix
        C = np.eye(dim)
        # Eigen decomposition for C
        # We'll maintain C and its square root via eigen decomposition periodically
        # For simplicity in dim=5, we can do full eigen each step or use a simplified update

        # Learning rates
        cmu = (mu + 2) / ((lam / 2 + mu) ** 2 + 2)
        cphi = (5 + cmu) / (dim + 5 + 2 * cmu)
        c1 = (2 + 2 * min(mu, (lam - mu) / 2) / (dim + 2)) / ((dim + 1.3) ** 2 + mu)
        cmu = (2 + 2 * min(mu, (lam - mu) / 2) / (dim + 2)) / ((dim + 1.3) ** 2 + mu)
        # Actually, let me use standard CMA-ES learning rates
        mu = lam // 2  # number of best points used for update
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        # State
        ps = np.zeros(dim)  # evolution path for sigma
        pc = np.zeros(dim)  # evolution path for C
        # B and D for C = B B^T, D^2
        B = np.eye(dim)
        D = np.ones(dim)

        def sample():
            """Generate lam samples."""
            # Standard normals
            z = rng.standard_normal((lam, dim))
            # Transform: B * D * z
            # y = sigma * (z @ D) @ B.T + m
            ys = m + sigma * (z * D) @ B.T
            return ys

        while calls[0] < budget and time.time() < deadline:
            # Generate population
            ys = sample()
            # Evaluate
            fs = np.array([wrapped(y) for y in ys])

            # Update best
            idx_min = np.argmin(fs)
            if fs[idx_min] < best_f:
                best_f = fs[idx_min]
                best_x = ys[idx_min].copy()

            # Sort by fitness
            order = np.argsort(fs)
            ys_sorted = ys[order]
            fs_sorted = fs[order]

            # Weighted mean of mu best
            m_new = m + sigma * np.sum(weights[:, None] * ys_sorted[:mu], axis=0)

            # Update evolution path for sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (m_new - m)
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (budget - calls[0]))) < (1 + 2 / (dim + 1)) * chi_n
            if hsig:
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

            # Update evolution path for C
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mueff) / sigma * (m_new - m)

            # Rank-mu update
            delta_h = (1 - hsig) * c_c * (2 - c_c)
            C_update = (1 + c_1 * delta_h - c_1 - c_mu) * C
            C_update += c_1 * (np.outer(pc, pc) - (1 - c_c) * (2 - c_c) * C)
            for i in range(mu):
                yi = (ys_sorted[i] - m) / sigma
                C_update += c_mu * weights[i] * np.outer(yi, yi)

            # Update C
            C = C_update

            # Eigen decomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip eigenvalues
            eigvals = np.clip(eigvals, 1e-20, None)
            B = eigvecs
            D = np.sqrt(eigvals)

            # Update mean
            m = m_new

            # Clip sigma
            sigma = np.clip(sigma, 1e-8, 1e8)

        return best_x, best_f
# EVOLVE-BLOCK-END
