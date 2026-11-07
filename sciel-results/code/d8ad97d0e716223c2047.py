"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive population, restarts, and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, sigma, budget, rng, deadline):
        """Run CMA-ES from x0 with initial sigma for up to budget evaluations."""
        dim = self.dim
        n = dim
        lam = max(4, int(4 + 3 * np.log(n)))
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        x = np.array(x0, dtype=float)
        ps = np.zeros(n)
        pc = np.zeros(n)
        B = np.eye(n)
        D = np.ones(n)
        sigma = sigma
        calls = 0
        best_x = x.copy()
        best_f = func(x)
        calls += 1

        while calls < budget:
            if time.time() > deadline:
                break

            # Generate population
            Y = np.zeros((lam, n))
            F = np.zeros(lam)
            for i in range(lam):
                zi = rng.standard_normal(n)
                Y[i] = x + sigma * (B @ (D * zi))
                F[i] = func(Y[i])
                calls += 1
                if time.time() > deadline:
                    break

            # Sort by fitness
            order = np.argsort(F)
            Y = Y[order]
            F = F[order]

            # Update best
            if F[0] < best_f:
                best_f = F[0]
                best_x = Y[0].copy()

            # Weighted mean
            m_new = np.sum(weights[:, None] * Y[:mu], axis=0)

            # Update evolution paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / (sigma) * (B @ (D * (m_new - x)))
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls // lam + 1)))) < (1.4 + 2 / (n + 1)) * chi_n
            pc = (1 - cc) * pc + (float(hsig) * cs * cc / damps) * (m_new - x)

            # Covariance update
            delta_d = float(hsig) * (1 - cc) / (1 - cc ** (2 * (calls // lam + 1)))
            rank_one = np.outer(pc, pc)
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                vi = B @ (D * (Y[i] - x)) / sigma
                rank_mu += weights[i] * np.outer(vi, vi)

            C = (1 + c1 * delta_d - c1 - cmu) * C + c1 * (rank_one + (1 - delta_d) * np.eye(n)) + cmu * rank_mu

            # Eigen decomposition for B and D
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            D = np.sqrt(eigvals)
            B = eigvecs

            # Sigma update
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-15, 1e15)

            x = m_new

        return best_x, best_f, calls

    def __call__(self, func):
        rng = np.random.default_rng(42)
        deadline = time.time() + 280  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Phase 1: Initial exploration with random restarts
        # Try a few random starting points to find a good region
        best_x = None
        best_f = np.inf
        used_calls = 0

        # First, do a quick random search to get an initial point
        for _ in range(5):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            used_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            if time.time() > deadline:
                break

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            used_calls += 1

        # Phase 2: CMA-ES with adaptive restarts
        remaining = budget - used_calls
        sigma_init = 2.0

        # Run CMA-ES with restarts
        n_restarts = 3
        calls_per_restart = remaining // n_restarts

        for r in range(n_restarts):
            if used_calls >= budget:
                break
            if time.time() > deadline:
                break

            if r == 0:
                x0 = best_x
                sigma = sigma_init
            else:
                # Restart from a perturbation of the best found so far
                x0 = best_x + 1.5 * rng.standard_normal(dim)
                x0 = np.clip(x0, -5, 5)
                sigma = max(0.5, sigma_init / (r + 1))

            cur_budget = min(calls_per_restart, budget - used_calls)
            if cur_budget < 10:
                break

            x_new, f_new, n_calls = self._cma_es(func, x0, sigma, cur_budget, rng, deadline)
            used_calls += n_calls

            if f_new < best_f:
                best_f = f_new
                best_x = x_new

        # Phase 3: Final local refinement if time and budget remain
        if used_calls < budget and time.time() < deadline:
            # Use a small (1+1)-ES for fine-tuning
            sigma = 0.1
            x = best_x.copy()
            fx = best_f
            while used_calls < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                used_calls += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                else:
                    sigma *= 0.85
                if sigma < 1e-10:
                    break
            if fx < best_f:
                best_f = fx
                best_x = x

        return best_x, best_f
# EVOLVE-BLOCK-END
