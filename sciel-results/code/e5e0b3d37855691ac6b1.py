"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB f24 (Lunacek bi-Rastrigin)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        # Population size
        lam = int(4 + 3 * np.log(dim))
        # Initial sigma
        sigma = 2.0
        # Initial mean
        rng = np.random.default_rng(42)
        m = rng.uniform(-4, 4, dim)

        # Weights for recombination
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        # Learning rates
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))

        # Damping
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Memory
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)
        step_size = np.ones(dim)

        # Initial evaluation
        x = np.clip(m, -5, 5)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1

        # Main loop
        while calls < budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate population
            pop = np.zeros((lam, dim))
            for i in range(lam):
                z = rng.standard_normal(dim)
                # Transform z using C matrix (Cholesky decomposition for efficiency)
                # Use Cholesky of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # If C is not positive definite, reset
                    C = np.eye(dim)
                    L = np.eye(dim)
                y = L @ z
                pop[i] = m + sigma * y
                pop[i] = np.clip(pop[i], -5, 5)

            # Evaluate population
            fvals = np.zeros(lam)
            for i in range(lam):
                fvals[i] = func(pop[i])
                calls += 1
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = pop[i].copy()

            # Sort by fitness
            order = np.argsort(fvals)
            pop = pop[order]
            fvals = fvals[order]

            # Update mean
            m_old = m.copy()
            m = np.sum(weights[:, None] * pop[:mu], axis=0)

            # Update covariance matrix adaptation
            # Compute the difference
            diff = (m - m_old) / sigma
            # Update step-size control vector
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / chi_n * np.linalg.norm(diff) * \
                 (diff / np.linalg.norm(diff) if np.linalg.norm(diff) > 1e-16 else np.zeros(dim))
            # Update evolution path
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) * diff

            # Rank-1 update
            C = (1 + c_1 * (mueff - 1) * np.dot(ps, ps) / chi_n ** 2 - c_1) * C + \
                c_1 * np.outer(pc, pc)

            # Rank-mu update
            C += c_mu * np.sum(weights[:mu] * np.outer(pop[:mu] - m_old, pop[:mu] - m_old) / sigma ** 2, axis=0)

            # Ensure C is symmetric and positive semi-definite
            C = (C + C.T) / 2
            # Regularize if needed
            eigvals, eigvecs = np.linalg.eigh(C)
            if np.min(eigvals) < 1e-12:
                C = eigvecs @ np.diag(np.maximum(eigvals, 1e-12)) @ eigvecs.T
                C = (C + C.T) / 2

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))

            # Clip sigma
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Ensure mean stays in bounds
            m = np.clip(m, -5, 5)

        return best_x, best_f
# EVOLVE-BLOCK-END
