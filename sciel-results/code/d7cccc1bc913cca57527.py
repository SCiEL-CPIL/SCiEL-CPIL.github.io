"""Optimized for BBOB f3 (Rastrigin) with a population-based CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts and time guard for BBOB f3."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_single(self, func, x0, sigma0, remaining_budget, rng, deadline):
        """Run CMA-ES from x0 with sigma0, using up to remaining_budget evaluations."""
        n = self.dim
        # CMA-ES parameters
        lambda_ = max(4, 4 + int(3 * np.log(n)))  # population size
        mu = lambda_ // 2
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, lambda_ + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w[:mu] ** 2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

        x = x0.copy()
        sigma = sigma0
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        C = np.eye(n)

        evals_used = 0
        best_x = x0.copy()
        best_f = func(x0)
        evals_used += 1

        while evals_used < remaining_budget:
            if time.time() > deadline:
                break

            # Generate population
            B = rng.standard_normal((lambda_, n))
            # Whitening transformation using Cholesky of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(n)

            Y = B @ L.T  # (lambda_, n)
            # Shift by x
            pops = x[None, :] + sigma * Y
            # Clip to bounds
            pops = np.clip(pops, -5, 5)

            # Evaluate
            fs = np.array([func(pops[i]) for i in range(lambda_)])
            evals_used += lambda_

            # Update best
            idx_best = np.argmin(fs)
            if fs[idx_best] < best_f:
                best_f = fs[idx_best]
                best_x = pops[idx_best].copy()

            # Sort
            order = np.argsort(fs)
            ys = Y[order]
            fs_sorted = fs[order]

            # Mean
            m_new = x + sigma * np.sum(w[:mu, None] * ys[:mu], axis=0)
            m_new = np.clip(m_new, -5, 5)

            # Update step-size control
            y_sigma = (m_new - x) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)

            # Update covariance matrix
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * evals_used / lambda_)) < (1.4 + 2 / (n + 1)) * chi_n:
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_sigma
            else:
                p_c = (1 - c_c) * p_c

            C = ((1 - c_1 - c_mu) * C +
                 c_1 * (np.outer(p_c, p_c) + c_c * (2 - c_c) * np.eye(n)) +
                 c_mu * np.sum(w[:mu, None, None] * np.outer(ys[:mu], ys[:mu]), axis=0))
            # Symmetrize
            C = (C + C.T) / 2

            x = m_new

        return best_x, best_f, evals_used

    def __call__(self, func):
        n = self.dim
        deadline = time.time() + 290  # safety margin before 300s limit
        total_budget = self.budget

        # Initial random points to get a sense of the landscape
        rng = np.random.default_rng(42)

        # Start with a few random initializations
        n_starts = min(5, total_budget // 50)
        if n_starts < 1:
            n_starts = 1

        best_x = None
        best_f = np.inf
        used = 0

        for i in range(n_starts):
            x0 = rng.uniform(-5, 5, n)
            f0 = func(x0)
            used += 1
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

            if used >= total_budget:
                break

        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            used += 1

        # Now run CMA-ES from the best point
        remaining = total_budget - used
        if remaining <= 0:
            return best_x, best_f

        # Adaptive sigma based on initial spread
        sigma0 = 2.0

        best_x, best_f, evals_used = self._cma_single(
            func, best_x, sigma0, remaining, rng, deadline
        )
        used += evals_used

        # If we still have budget and time, do a second restart from a different point
        if used < total_budget and time.time() < deadline:
            x0_2 = rng.uniform(-5, 5, n)
            f0_2 = func(x0_2)
            used += 1
            if f0_2 < best_f:
                best_f = f0_2
                best_x = x0_2.copy()

            remaining2 = total_budget - used
            if remaining2 > 20 and time.time() < deadline:
                best_x2, best_f2, evals2 = self._cma_single(
                    func, best_x, sigma0, remaining2, rng, deadline
                )
                used += evals2
                if best_f2 < best_f:
                    best_f = best_f2
                    best_x = best_x2

        return best_x, best_f
# EVOLVE-BLOCK-END
