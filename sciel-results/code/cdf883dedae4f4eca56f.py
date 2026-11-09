"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with time guard for BBOB-style f24 (Lunacek bi-Rastrigin)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cmaes(self, func, x0, sigma0, budget, rng, deadline):
        """Run CMA-ES from x0 for at most `budget` evaluations, respecting deadline."""
        n = self.dim
        # Population size: 4 + floor(3 * log(n))
        lam = max(4, 4 + int(3 * np.log(n)))
        # Ensure we don't exceed budget
        lam = min(lam, max(2, budget // 2))
        if lam < 2:
            lam = 2

        # Strategy parameters
        mu = lam // 2  # number of recombination weights
        # Logarithmic weights
        w = np.log(np.arange(mu, 0, -1)) - np.log(np.arange(lam, mu, -1))
        w = w / w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        # Learning rates
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        # Step-size control
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        # Initial values
        mean = np.array(x0, dtype=float)
        sigma = sigma0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        # Precompute inverse sqrt of C
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 1e-20)
        Csqr_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        best_x = mean.copy()
        best_f = func(mean)
        calls = 1
        if calls > budget:
            return best_x, best_f, calls

        for gen in range(budget // lam + 2):
            # Check deadline
            if time.time() > deadline - 5:
                break
            # Check budget
            if calls >= budget:
                break

            # Sample population
            Y = rng.standard_normal((lam, n))
            # Transform to X
            X = mean[None, :] + sigma * (Y @ Csqr_inv.T)
            # Evaluate
            F = np.empty(lam)
            for i in range(lam):
                xi = np.clip(X[i], -5, 5)
                F[i] = func(xi)
                calls += 1
                if calls > budget:
                    break
                if time.time() > deadline - 3:
                    break
            # If we ran out of budget mid-generation, break
            if calls > budget or time.time() > deadline - 3:
                # Update best
                for i in range(lam):
                    xi = np.clip(X[i], -5, 5)
                    if F[i] < best_f:
                        best_f = F[i]
                        best_x = xi.copy()
                break

            # Sort by fitness
            order = np.argsort(F)
            x_sorted = X[order]
            f_sorted = F[order]

            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = np.clip(x_sorted[0], -5, 5).copy()

            # Recombination
            mean_old = mean.copy()
            mean = np.sum(w[:, None] * x_sorted[:mu], axis=0)

            # Update covariance matrix
            y = (mean - mean_old) / sigma
            pc = (1 - cc) * pc + cc * y
            delta_sig = np.linalg.norm(pc) / np.sqrt(1 - (1 - cc) ** (gen + 1)) - 1
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * cc) * y
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (gen + 1))) < (1.4 + 2 / (n + 1)) * chi_n
            C = ((1 - c1 - cmu) * C
                 + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C - cc * C)
                 + cmu * np.sum(w[:, None, None] * np.outer(x_sorted[:mu] - mean_old, x_sorted[:mu] - mean_old), axis=0) / sigma ** 2)
            # Symmetrize
            C = (C + C.T) / 2

            # Step-size update
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))

            # Re-derive Csqr_inv periodically (every 10 generations)
            if (gen + 1) % 10 == 0:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                Csqr_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            # Clip mean to box
            mean = np.clip(mean, -5, 5)

            # Early stopping: if sigma is tiny or no improvement in many generations
            if sigma < 1e-10:
                break

        return best_x, best_f, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        total_calls = 0

        # Multi-start: try several initial points with decreasing sigma
        # Lunacek bi-Rastrigin has a deceptive global min far from local min
        # Use diverse starting points
        n_starts = 5
        # Reserve budget per start
        per_start_budget = max(20, budget // n_starts)

        for s in range(n_starts):
            if total_calls >= budget:
                break
            if time.time() > deadline - 10:
                break

            # Choose starting point
            if s == 0:
                x0 = rng.uniform(-2, 2, n)
                sigma0 = 1.5
            elif s == 1:
                x0 = rng.uniform(-4, 4, n)
                sigma0 = 2.0
            elif s == 2:
                x0 = rng.uniform(-5, 5, n)
                sigma0 = 2.5
            elif s == 3:
                # Try near origin (Lunacek's local min is near origin)
                x0 = rng.uniform(-1, 1, n)
                sigma0 = 0.5
            else:
                # Random
                x0 = rng.uniform(-4, 4, n)
                sigma0 = 1.0

            remaining = budget - total_calls
            if remaining < 10:
                break

            x, f, calls = self._cmaes(func, x0, sigma0, remaining, rng, deadline)
            total_calls += calls

            if f < best_f:
                best_f = f
                best_x = x.copy()

            # If we found a very good solution, polish with smaller budget
            if best_f < 1.0 and total_calls < budget and time.time() < deadline - 20:
                # Polish phase
                remaining = budget - total_calls
                x, f, calls = self._cmaes(func, best_x, 0.1, remaining, rng, deadline)
                total_calls += calls
                if f < best_f:
                    best_f = f
                    best_x = x.copy()

        if best_x is None:
            # Fallback: single evaluation
            x = np.zeros(n)
            best_f = func(x)
            best_x = x
            total_calls = 1

        return best_x, best_f
# EVOLVE-BLOCK-END
