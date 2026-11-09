# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES for Rosenbrock f8 on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cmaes(self, func, x0, sigma0, budget, rng, time_start, time_limit):
        """Run CMA-ES with a given start, sigma, and call budget.
        Returns (best_x, best_f, calls_used)."""
        d = self.dim
        x_best = np.array(x0, dtype=float)
        f_best = func(x_best)
        calls = 1

        # CMA-ES parameters
        lam = max(4, int(4 + 3 * np.log(d)))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = (2 / ((d + 1.3) ** 2 + mu_eff)) * (2 - c_c)
        c_mu = (2 / ((d + 2) ** 2 + mu_eff)) * (
            min(1 - c_1, 2 / ((d + 1.3) ** 2 + mu_eff))
        )
        chi_n = np.sqrt(d) * (1 - 1.0 / (4 * d) + 1.0 / (21 * d ** 2))

        mean = np.array(x0, dtype=float)
        sigma = sigma0
        C = np.eye(d)
        C_inv = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        # Precompute inverse sqrt of C via eigendecomposition each step is costly;
        # use Cholesky-based sampling and inverse via solve.
        # We'll maintain C and compute sqrt_inv via eig each generation (d=5, cheap).

        for gen in range(budget):
            if time.time() - time_start > time_limit - 2.0:
                break
            # Sample lambda points
            Y = np.zeros((lam, d))
            for i in range(lam):
                # Sample via eigendecomposition of C
                w, V = np.linalg.eigh(C)
                w = np.maximum(w, 1e-12)
                z = rng.standard_normal(d)
                Y[i] = V @ (np.sqrt(w) * z)
            X = mean[None, :] + sigma * Y
            # Clip to box
            X = np.clip(X, -5.0, 5.0)
            # Evaluate
            F = np.empty(lam)
            for i in range(lam):
                if time.time() - time_start > time_limit - 2.0:
                    break
                F[i] = func(X[i])
                calls += 1
                if F[i] < f_best:
                    f_best = F[i]
                    x_best = X[i].copy()
            # If we ran out of time mid-eval, break
            if time.time() - time_start > time_limit - 2.0:
                break

            # Sort by fitness
            order = np.argsort(F)
            X_sorted = X[order]
            F_sorted = F[order]
            X_w = X_sorted[:mu]

            # Update mean
            mean_old = mean.copy()
            mean = mean + sigma * np.sum(weights[:, None] * X_w, axis=0)

            # Update step-size control
            C_w, V_w = np.linalg.eigh(C)
            C_w = np.maximum(C_w, 1e-12)
            C_sqrt_inv = V_w @ np.diag(1.0 / np.sqrt(C_w)) @ V_w.T
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * \
                 C_sqrt_inv @ ((mean - mean_old) / sigma)
            norm_ps = np.linalg.norm(ps)
            sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
            sigma = min(sigma, 1e6)
            sigma = max(sigma, 1e-12)

            # Update evolution path for covariance
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * \
                 (mean - mean_old) / sigma

            # Update covariance
            delta_hs = (1.7 + 2 / (mu_eff + 1)) if (norm_ps ** 2 / (1 - (1 - c_sigma) ** (2 * (gen + 1)))) > \
                (1.1 + d / lam) * chi_n ** 2 else 0.0
            C = (1 + c_c * (2 - c_c) - c_1 - c_mu) * C \
                + c_c * (np.outer(pc, pc) - delta_hs * C)
            C = C + c_1 * (C - np.eye(d))
            C = C + c_mu * (np.outer(weights @ X_w, weights @ X_w) - C)
            # Symmetrize
            C = 0.5 * (C + C.T)

        return x_best, f_best, calls

    def __call__(self, func):
        time_start = time.time()
        time_limit = 300.0  # safety margin under 330s kill
        budget = self.budget
        d = self.dim
        rng = np.random.default_rng(42)

        # Total call budget
        calls_used = 0
        best_x = None
        best_f = np.inf

        # Multi-restart: allocate sub-budgets
        # First restart gets a larger share, subsequent get smaller
        n_restarts = 5
        # Sub-budgets: decreasing
        sub_budgets = []
        remaining = budget
        for i in range(n_restarts):
            if i == n_restarts - 1:
                sub = remaining
            else:
                frac = 0.4 if i == 0 else 0.2
                sub = int(remaining * frac)
            sub = max(sub, 10)
            sub_budgets.append(sub)
            remaining -= sub
        # Ensure sum <= budget
        total_alloc = sum(sub_budgets)
        if total_alloc > budget:
            sub_budgets[-1] = budget - sum(sub_budgets[:-1])

        for r, sub in enumerate(sub_budgets):
            if calls_used >= budget:
                break
            if time.time() - time_start > time_limit - 3.0:
                break
            # Start point: for first restart, use origin-ish; for others, random
            if r == 0:
                x0 = np.zeros(d)
                sigma0 = 2.0
            else:
                x0 = rng.uniform(-4, 4, d)
                sigma0 = 1.5

            # Cap sub-budget to remaining
            sub = min(sub, budget - calls_used)
            if sub <= 0:
                break

            x_b, f_b, calls = self._cmaes(
                func, x0, sigma0, sub, rng, time_start, time_limit
            )
            calls_used += calls
            if f_b < best_f:
                best_f = f_b
                best_x = x_b.copy()

        if best_x is None:
            # Fallback: at least one call
            x0 = np.zeros(d)
            best_f = func(x0)
            best_x = x0
            calls_used += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
