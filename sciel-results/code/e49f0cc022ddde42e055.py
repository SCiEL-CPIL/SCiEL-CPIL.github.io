"""Rewritten optimizer using CMA-ES for BBOB-style f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for 5-dimensional box-constrained optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit

        rng = np.random.default_rng(12345)

        # CMA-ES parameters
        lam = 15  # population size (standard for d=5: 4+3*ln(5)≈9, but 15 is robust)
        # Adjust lam to fit budget: each generation uses lam evaluations
        # Initial point + lam per generation
        # Total = 1 + (budget-1) evaluations available
        max_gens = (budget - 1) // lam
        if max_gens < 5:
            lam = max(4, (budget - 1) // 5)
            max_gens = (budget - 1) // lam

        mu = lam // 2  # number of parents
        # Weights for recombination
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        # Step size control parameters
        cc = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        cs = (mueff + 2) / (d + mueff + 5)
        c1 = (2 - cc + 2 * mueff / (d - 1)) / ((d + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (d + 1)) - 1) + cs

        # Initial state
        x = rng.uniform(-5, 5, d)
        sigma = 2.0
        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)

        # Evaluate initial point
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls_made = 1

        for gen in range(max_gens):
            # Check time
            if time.time() > deadline:
                break

            # Generate population
            # Cholesky factor of C
            L = np.linalg.cholesky(C)

            Y = rng.standard_normal((lam, d))
            pop = np.array([x + sigma * (Y[i] @ L.T) for i in range(lam)])
            pop = np.clip(pop, -5, 5)

            # Evaluate population
            F = np.empty(lam)
            for i in range(lam):
                if time.time() > deadline:
                    break
                F[i] = func(pop[i])
                calls_made += 1

            # Sort by fitness
            idx = np.argsort(F)
            pop_sorted = pop[idx]
            F_sorted = F[idx]

            # Update best
            if F_sorted[0] < best_fx:
                best_fx = F_sorted[0]
                best_x = pop_sorted[0].copy()

            # Mean of selected parents
            m_new = np.sum(weights[:, None] * pop_sorted[:mu], axis=0)

            # Update path covariance
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) / sigma * (m_new - x)

            # Update path sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (L.T @ (m_new - x) / sigma)

            # Hessian update for C
            delta_sigma = np.sqrt((np.linalg.norm(ps) ** 2 / d - 1) / (1 - (1 - cs) ** (2 * (gen + 1))))
            C = ((1 - c1 - cmu) * C
                 + c1 * (np.outer(pc, pc) + (cc * (2 - cc)) * np.eye(d))
                 + cmu * np.sum(weights[:mu] * np.outer(pop_sorted[:mu] - x, pop_sorted[:mu] - x) / sigma ** 2, axis=0)
                 + c1 * (np.outer(pc, pc) + (cc * (2 - cc)) * np.eye(d) - C) * 0  # no-op, keep structure
                 )
            # Simplified: standard CMA-ES update
            C = ((1 - c1 - cmu) * C
                 + c1 * (np.outer(pc, pc) + (cc * (2 - cc)) * np.eye(d))
                 + cmu / sigma ** 2 * np.sum(weights[:mu] * np.outer(pop_sorted[:mu] - x, pop_sorted[:mu] - x), axis=0))

            # Ensure positive semi-definite
            C = (C + C.T) / 2
            # Regularize if needed
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-10:
                C += 1e-10 * np.eye(d)

            # Update sigma
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(d) - 1))
            sigma = np.clip(sigma, 1e-12, 1e6)

            # Update mean
            x = m_new

            # Check convergence
            if sigma < 1e-10 or np.all(np.abs(pop_sorted[:mu] - x) < 1e-10):
                break

        return best_x, best_fx
# EVOLVE-BLOCK-END
