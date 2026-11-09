"""Black-box optimizer for BBOB-style f6 (Attractive sector) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """CMA-ES (self-contained) tuned for rotated/shifted black-box landscapes."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        deadline = time.time() + 295.0  # safety margin before the 330 s kill

        # --- Initialisation: spread points across the box (cheap, informative) ---
        x0 = np.zeros(n)
        # Deterministic-ish spread start: corners of the box + center, then best.
        starts = []
        for c in ([-5.0] * n, [5.0] * n):
            starts.append(np.array(c, dtype=float))
        # Add a few deterministic spread points to seed the population.
        rng = np.random.default_rng(12345)
        for _ in range(4):
            starts.append(rng.uniform(-5.0, 5.0, n))
        starts.append(np.zeros(n))

        fx = []
        for s in starts:
            fx.append(func(np.clip(s, -5.0, 5.0)))
        fx = np.array(fx, dtype=float)
        best_idx = int(np.argmin(fx))
        xbest = starts[best_idx].copy()
        fbest = fx[best_idx]
        calls = len(starts)

        # Seed population around the best start point.
        lam = max(n + int(1 + 3 * np.log(n)), 4)  # population size
        m = lam // 2

        # Initial step size: large to cover the box quickly, then adapt.
        sigma = 3.0
        x = xbest.copy()

        # CMA-ES state
        C = np.eye(n)
        B = np.eye(n)
        D = np.ones(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        mu = m
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1.0 + 2 * max(0.0, (mueff - 1) / (n + 1) - 1) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = (2 - mueff / (n + 1) ** 2 + c_sigma) / ((n + 2) + mueff)
        c_mu = min(1.0 - c_1,
                   2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        def sample_population():
            # Draw lam candidates from N(x, sigma^2 C).
            cand = np.empty((lam, n))
            vals = np.empty(lam)
            for i in range(lam):
                z = B @ (D * rng.standard_normal(n))
                cand[i] = x + sigma * z
                vals[i] = func(cand[i])
            return cand, vals

        while calls < budget:
            if time.time() > deadline:
                break

            cand, vals = sample_population()
            calls += lam

            # Track best
            local_best = int(np.argmin(vals))
            if vals[local_best] < fbest:
                fbest = vals[local_best]
                xbest = cand[local_best].copy()

            # Sort by fitness
            order = np.argsort(vals)
            cand_s = cand[order]
            vals_s = vals[order]

            # Update mean (weighted recombination of the best mu)
            y = (weights[:, None] * cand_s[:mu]).sum(axis=0)
            # Update evolution paths
            pc = (1 - c_c) * pc + c_c * (y / (sigma * D) - 1 + c_1 * pc)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (y / sigma)

            # Update covariance
            delta = np.sqrt(c_sigma * (2 - c_sigma) * mueff) / (sigma * D)
            ps_sq = np.dot(ps, ps)
            rank_c = np.outer(pc, pc)
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                yi = cand_s[i] - x
                rank_mu += weights[i] * np.outer(yi / sigma, yi / sigma)
            C = ((1 + c_1 * (ps_sq / n - 1) - c_1) * C
                 + c_1 * rank_c + c_mu * rank_mu)
            # Symmetrize
            C = 0.5 * (C + C.T)

            # Re-orthonormalize C (periodic, but cheap for n=5 each step)
            evals, evecs = np.linalg.eigh(C)
            evals = np.clip(evals, 1e-20, None)
            B = evecs
            D = np.sqrt(evals)
            C = (B * D) @ B.T

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (ps_sq / chi_n ** 2 - 1))
            sigma = float(np.clip(sigma, 1e-12, 1e3))

            # New mean
            x = x + y

        # Final safety: return best found
        return xbest, fbest
# EVOLVE-BLOCK-END
