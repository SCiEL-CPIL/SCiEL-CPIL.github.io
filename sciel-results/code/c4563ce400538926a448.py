"""Combined CMA-ES optimizer for BBOB-style Bent Cigar (f12)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive covariance for ill-conditioned functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)
        start_time = time.time()
        deadline = start_time + 285.0

        # CMA-ES parameters
        lam = max(4, int(4 + 3 * np.log(dim)))  # ~9 for dim=5
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        sigma = 2.0
        C = np.eye(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        dsigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initial point: try origin first, then a few random points to get a good start
        candidates = [np.zeros(dim)]
        for _ in range(4):
            candidates.append(rng.uniform(-5, 5, dim))
        
        best_x = None
        best_fx = np.inf
        for cand in candidates:
            fc = func(cand)
            if fc < best_fx:
                best_fx = fc
                best_x = cand.copy()
        
        x = best_x.copy()
        calls_made = len(candidates)
        remaining = budget - calls_made

        # Pre-compute Cholesky once; recompute if C changes significantly
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            L = np.eye(dim)

        while remaining > 0 and time.time() < deadline:
            # Generate population
            offspring = []
            for i in range(lam):
                if time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                y = x + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                remaining -= 1
                offspring.append((y, fy))
                if remaining <= 0:
                    break

            if not offspring:
                break

            # Sort by fitness
            offspring.sort(key=lambda t: t[1])

            # Update best
            if offspring[0][1] < best_fx:
                best_fx = offspring[0][1]
                best_x = offspring[0][0].copy()

            # Select parents
            parents = offspring[:mu]

            # Update mean
            new_x = np.zeros(dim)
            for w, p in zip(weights, parents):
                new_x += w * p[0]
            new_x = np.clip(new_x, -5, 5)

            # Update evolution paths
            y_mean = (new_x - x) / max(sigma, 1e-10)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * y_mean
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * calls_made))) < (1.4 + 2 / (dim + 1)) * chi_n
            pc = (1 - cs) * pc + hsig * np.sqrt(cs * (2 - cs) * mueff) * y_mean

            # Update covariance
            delta = 0 if hsig else 1
            C = (1 + c1 * delta + cmu - c1 - cmu) * C
            C += c1 * (np.outer(pc, pc) - C)
            for w, p in zip(weights, parents):
                yk = (p[0] - x) / max(sigma, 1e-10)
                C += cmu * w * np.outer(yk, yk)

            # Update sigma
            sigma *= np.exp((1 / dsigma) * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 100)

            # Ensure C is positive semi-definite and update Cholesky
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(dim)

            x = new_x

            # Early termination if sigma is very small
            if sigma < 1e-8:
                break

        return best_x, best_fx
# EVOLVE-BLOCK-END
