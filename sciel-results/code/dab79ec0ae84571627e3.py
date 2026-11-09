"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with deadline guard for BBOB f7 (Step ellipsoidal)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cmaes(self, func, x0, sigma, max_iter, deadline, rng):
        n = self.dim
        x = np.array(x0, dtype=float)
        f = func(x)
        calls = 1

        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(n))
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
        cs = (mueff + 2.0) / (n + mueff + 5.0)
        c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))
        chi = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))

        ps = np.zeros(n)
        pc = np.zeros(n)
        B = np.eye(n)
        D = 1.0
        # Use diagonal covariance initially for simplicity, but we need full CMA-ES
        C = np.eye(n)
        sqrtC = np.eye(n)
        invsqrtC = np.eye(n)

        def update_sqrtC(C):
            nonlocal B, D, sqrtC, invsqrtC
            B, D = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-20))
            sqrtC = B @ np.diag(D)
            invsqrtC = B @ np.diag(1.0 / D)

        update_sqrtC(C)

        for it in range(max_iter):
            if time.time() > deadline:
                break
            if calls >= self.budget:
                break

            # Generate lambda samples
            y = rng.standard_normal((lambda_, n))
            z = invsqrtC @ y.T
            x_trials = np.array([x + sigma * z[:, i] for i in range(lambda_)])
            x_trials = np.clip(x_trials, -5, 5)

            # Evaluate
            f_trials = np.array([func(xi) for xi in x_trials])
            calls += lambda_

            # Sort
            idx = np.argsort(f_trials)
            x_trials = x_trials[idx]
            f_trials = f_trials[idx]

            # Update best
            if f_trials[0] < f:
                f = f_trials[0]
                x = x_trials[0]

            # Weighted mean
            m_new = np.sum(weights[:, None] * x_trials[:mu], axis=0)

            # Update evolution paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ (m_new - x) / sigma)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (it + 1)))) < (1.4 + 2 / (n + 1)) * chi
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (m_new - x) / sigma

            # Rank-one and rank-mu update
            delta_hsig = (1 - hsig) * cc * (1 - cc)
            C = (1 + c1 * delta_hsig - c1 - cmu) * C \
                + c1 * np.outer(pc, pc) \
                + cmu * np.sum(weights[:, None, None] * np.outer(x_trials[:mu] - x, x_trials[:mu] - x) / sigma ** 2, axis=0)

            # Ensure symmetry
            C = 0.5 * (C + C.T)

            # Update sigma
            sigma *= np.exp(cs / chi * (np.linalg.norm(ps) / chi - 1))
            sigma = np.clip(sigma, 1e-10, 1e10)

            # Update x
            x = m_new
            x = np.clip(x, -5, 5)

            # Update sqrt(C) every 10 iterations
            if it % 10 == 0:
                update_sqrtC(C)

        return x, f, calls

    def __call__(self, func):
        deadline = time.time() + 280  # Safety margin before 300s limit
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        total_calls = 0

        # Multi-start CMA-ES
        n_starts = 3
        calls_per_start = self.budget // n_starts

        for start in range(n_starts):
            if total_calls >= self.budget:
                break
            if time.time() > deadline:
                break

            # Initialize
            if start == 0:
                x0 = np.zeros(self.dim)
            else:
                x0 = rng.uniform(-5, 5, self.dim)

            sigma0 = 2.0 if start == 0 else 3.0

            remaining = self.budget - total_calls
            x, f, calls = self._cmaes(func, x0, sigma0, remaining, deadline, rng)
            total_calls += calls

            if f < best_f:
                best_f = f
                best_x = x

        # Final polish: (1+1)-ES with small step
        if best_x is not None and total_calls < self.budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            remaining = self.budget - total_calls
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                total_calls += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.8
                if fx < best_f:
                    best_f = fx
                    best_x = x

        if best_x is None:
            # Fallback
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
            total_calls += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
