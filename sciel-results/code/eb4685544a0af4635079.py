"""Fresh design: multi-start CMA-ES with a random seeding phase and a deadline guard."""
import numpy as np
import time

class Optimizer:
    """Population-based CMA-ES with random seeding and a wall-clock deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.deadline = time.time() + 320.0  # safety margin before 330s

    def _cmaes(self, func, x0, fx0, sigma, budget_left, rng):
        """Run CMA-ES starting from x0 with step size sigma, using up to budget_left calls."""
        dim = self.dim
        # Population size
        lam = max(4, int(4 + 3 * np.log(dim)))
        mu = lam // 2
        # Weights for recombination
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # State
        x_best = x0.copy()
        f_best = fx0
        mean = x0.copy()
        C = np.eye(dim)
        C_inv = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        steps = 0

        # Cholesky helper
        def chol_with_fallback(mat, rng):
            try:
                L = np.linalg.cholesky(mat)
                return L
            except np.linalg.LinAlgError:
                # Symmetrize and add jitter
                mat = 0.5 * (mat + mat.T) + 1e-9 * np.eye(dim)
                try:
                    L = np.linalg.cholesky(mat)
                    return L
                except np.linalg.LinAlgError:
                    return np.eye(dim)

        while steps < budget_left:
            if time.time() > self.deadline:
                break
            L = chol_with_fallback(C, rng)
            # Sample population
            samples = np.array([mean + sigma * (L @ rng.standard_normal(dim)) for _ in range(lam)])
            samples = np.clip(samples, -5, 5)
            # Evaluate
            fvals = np.array([func(s) for s in samples])
            steps += lam
            # Sort
            order = np.argsort(fvals)
            y_sorted = samples[order]
            f_sorted = fvals[order]

            # Update best
            if f_sorted[0] < f_best:
                f_best = f_sorted[0]
                x_best = y_sorted[0].copy()

            # Weighted mean
            new_mean = np.sum(w[:, None] * y_sorted[:mu], axis=0)
            # Success rule for sigma
            d_sigma = c_sigma * (1 + (1 - c_sigma) / 2)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (L.T @ (new_mean - mean) / sigma)
            if np.linalg.norm(p_sigma) / chi_n < (1.4 + 2 / (dim + 1)) * c_sigma:
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))

            # Update covariance
            yk = (new_mean - mean) / sigma
            C = ((1 - c_c * (2 - c_c) - c_1 - c_mu) * C
                 + c_c * (2 - c_c) * np.outer(p_c, p_c)
                 + c_1 * (C - np.outer(yk, yk))
                 + c_mu * np.sum(w[:, None, None] * np.outer(y_sorted[:mu] - mean, y_sorted[:mu] - mean) / sigma ** 2, axis=0))
            C = 0.5 * (C + C.T)
            # Invert C (with jitter)
            try:
                C_inv = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                C_inv = np.linalg.inv(C + 1e-9 * np.eye(dim))

            # Evolution path p_c
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * yk
            mean = new_mean
            mean = np.clip(mean, -5, 5)

            # Stop if sigma too small or budget exhausted
            if sigma < 1e-12:
                break

        return x_best, f_best

    def __call__(self, func):
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget
        start_time = time.time()

        # Phase 1: Random seeding to find a good starting point
        n_seed = min(50, budget // 10)
        x_best = None
        f_best = np.inf
        for _ in range(n_seed):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < f_best:
                f_best = fx
                x_best = x.copy()
        budget_left = budget - n_seed

        if budget_left <= 0:
            return x_best, f_best

        # Phase 2: Multi-start CMA-ES
        n_restarts = 3
        budget_per_restart = max(10, budget_left // n_restarts)
        for r in range(n_restarts):
            if budget_per_restart <= 0:
                break
            if time.time() > self.deadline:
                break
            # Start from current best with a fresh sigma
            x0 = x_best.copy()
            sigma = 2.0
            xb, fb = self._cmaes(func, x0, f_best, sigma, budget_per_restart, rng)
            if fb < f_best:
                f_best = fb
                x_best = xb.copy()
            budget_left -= budget_per_restart

        # Phase 3: Fine-tuning with a smaller sigma from the best point
        if budget_left > 5 and time.time() < self.deadline:
            sigma = 0.3
            xb, fb = self._cmaes(func, x_best.copy(), f_best, sigma, budget_left, rng)
            if fb < f_best:
                f_best = fb
                x_best = xb.copy()

        return x_best, f_best
