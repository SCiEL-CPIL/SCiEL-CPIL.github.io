"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with deadline guard for ill-conditioned Schaffers F7."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0

        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # Initialize at origin (center of box)
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        # CMA-ES parameters
        sigma = 1.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        cs = (dim + 2) / (dim + 8)
        cmu = 2 / ((dim + 4) ** 2 + 2)
        cc = (4 + dim / dim) / (dim + 4 + 2 / dim)
        c1 = (2 / ((dim + 1.3) ** 2 + dim)) * (1 - cmu)
        csig = (1 + 2 / (dim + dim / 3)) * (1 - c1 - cmu / 2)
        chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Population size
        lam = 4 + int(np.floor(3 * np.log(dim)))

        # Budget allocation: use most of budget for CMA-ES
        max_iters = min(budget - 1, 499)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            if calls >= budget:
                break

            # Sample population
            B = np.array([C @ (sigma * rng.standard_normal(dim)) for _ in range(lam)])
            Y = x[None, :] + B
            Y = np.clip(Y, -5, 5)

            # Evaluate population
            F = np.array([func(y) for y in Y])
            calls += lam

            # Track best
            idx_min = np.argmin(F)
            if F[idx_min] < best_fx:
                best_fx = F[idx_min]
                best_x = Y[idx_min].copy()

            # Sort by fitness
            order = np.argsort(F)
            Y_sorted = Y[order]
            F_sorted = F[order]

            # Weighted recombination
            weights = np.log(lam / 2 + 0.5) - np.log(np.arange(1, lam + 1))
            weights /= weights.sum()
            mueff = int(np.sum(weights[:1]) ** 2 / np.sum(weights ** 2))

            # Mean update
            m_new = np.sum(weights[:, None] * Y_sorted, axis=0)

            # Covariance matrix adaptation
            y_sigma = (m_new - x) / sigma
            pc = (1 - cs) * pc + cs * y_sigma
            hsig = (np.linalg.norm(pc) / np.sqrt(1 - (1 - cs) ** (it + 1)) / chi) < 1.4 + 2 / (dim + 1)
            C = (1 - c1 - cmu + c1 * (1 - hsig) + cmu * hsig) * C \
                + c1 * np.outer(pc, pc) \
                + cmu / hsig * np.sum(weights[:mueff, None] * np.outer(Y_sorted[:mueff] - x, Y_sorted[:mueff] - x) / sigma ** 2, axis=0)

            # Step size adaptation
            sigma *= np.exp((cs / chi) * (np.linalg.norm(y_sigma) / chi - 1))
            sigma = np.clip(sigma, 1e-10, 100.0)

            x = m_new
            fx = F_sorted[0]

        return best_x, best_fx
# EVOLVE-BLOCK-END
