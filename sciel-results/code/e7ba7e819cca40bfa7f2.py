"""Optimizer for BBOB-style f17 (Schaffers F7) with hidden shift/rotation/offset."""
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive step size and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 280.0  # safety margin below the 330 s kill

        dim = self.dim
        budget = self.budget

        # Use a fixed seed for reproducibility within a single call
        rng = np.random.default_rng(42)

        # Initial point: center of the box
        x = np.zeros(dim)
        fx = func(x)

        # CMA-ES parameters
        sigma = 2.0
        # Initial covariance identity
        C = np.eye(dim)
        # Eigen decomposition for step sampling
        eigvals, eigvecs = np.linalg.eigh(C)
        # Ensure positive eigenvalues
        eigvals = np.maximum(eigvals, 1e-10)

        # Adaptation rates
        mu = 4  # number of parents
        lam = 8  # population size
        weights = np.zeros(lam)
        weights[:mu] = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        ps = np.zeros(dim)
        pc = np.zeros(dim)
        hsig = 0.0
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        dampc = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cc

        calls_made = 1  # initial evaluation

        for gen in range(budget - 1):
            # Check deadline
            if time.time() > deadline - 5.0:
                break

            # Generate population
            ys = np.zeros((lam, dim))
            fs = np.zeros(lam)
            for i in range(lam):
                # Sample from N(0, C)
                z = rng.standard_normal(dim)
                # Transform to C: eigvecs @ diag(sqrt(eigvals)) @ z
                y = x + sigma * (eigvecs @ (np.sqrt(eigvals) * z))
                y = np.clip(y, -5, 5)
                ys[i] = y
                fs[i] = func(y)
                calls_made += 1

            # Sort by fitness
            order = np.argsort(fs)
            ys = ys[order]
            fs = fs[order]

            # Update best solution
            if fs[0] < fx:
                x = ys[0].copy()
                fx = fs[0]

            # Weighted mean of parents
            m_new = np.sum(weights[:mu, None] * ys[:mu], axis=0)

            # Update evolution paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (eigvecs.T @ (m_new - x))
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) <= (1.4 + 2 / (dim + 1)) * chi_n
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * dampc) / sigma * (m_new - x)

            # Update covariance matrix
            delta_hsig = (1 - hsig) * cc * (2 - cc)
            C = (1 + c1 * delta_hsig - c1 - cmu) * C + \
                c1 * np.outer(pc, pc) + \
                cmu / sigma ** 2 * np.sum(weights[:mu, None, None] * np.outer(ys[:mu] - m_new, ys[:mu] - m_new), axis=0)

            # Update mean
            x = m_new

            # Update sigma
            sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))

            # Update eigen decomposition (every few generations to save time)
            if gen % 5 == 0 or gen < 10:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)

            # Adaptive step size: if we're making progress, keep sigma; if stuck, reduce
            if gen > 20 and fs[0] == fs[-1]:
                sigma *= 0.9

        # Final polish with (1+1)-ES if we have time
        remaining_time = deadline - time.time()
        if remaining_time > 10.0:
            sigma_polish = max(sigma * 0.1, 0.01)
            for _ in range(20):
                if time.time() > deadline - 2.0:
                    break
                y = np.clip(x + sigma_polish * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma_polish *= 1.2
                else:
                    sigma_polish *= 0.85

        return x, fx
