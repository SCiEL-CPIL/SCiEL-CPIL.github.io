"""Black-box continuous optimizer on BBOB-style f3 (Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with Rastrigin-aware restarts and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_step(self, x, sigma, C, diag, rng, func, deadline):
        """One step of simplified CMA-ES. Returns (x, fx, sigma, C, diag, calls_used)."""
        n = self.dim
        calls_used = 0

        # Sample lambda points (use small lambda for dim=5)
        lam = max(4, n + 1)  # 6 for dim=5
        mu = lam // 2

        # Generate candidates
        Y = np.zeros((lam, n))
        for i in range(lam):
            if time.time() > deadline:
                return x, None, sigma, C, diag, calls_used
            Y[i] = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
            calls_used += 1

        # Evaluate
        F = np.full(lam, np.inf)
        for i in range(lam):
            if time.time() > deadline:
                break
            F[i] = func(Y[i])
            calls_used += 1

        # Sort by fitness
        idx = np.argsort(F)
        Ys = Y[idx]
        Fs = F[idx]

        # Update mean
        weights = np.ones(mu) / mu
        x_new = np.sum(weights[:, None] * Ys[:mu], axis=0)

        # Update covariance
        diff = (Ys[:mu] - x) / sigma
        C = (1 - 2 * (1 / mu) * (1 / n)) * C + (1 / mu) * np.outer(diff.T @ diff, 1)
        # Simplified: just use rank-1 update
        C = 0.8 * C + 0.2 * np.outer(diff.T @ diff, 1) / mu

        # Update sigma (1/5 rule style)
        if Fs[0] < Fs[-1]:
            sigma *= 1.05
        else:
            sigma *= 0.95

        sigma = np.clip(sigma, 0.01, 5.0)

        return x_new, Fs[0], sigma, C, diag, calls_used

    def __call__(self, func):
        n = self.dim
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        total_calls = 0

        # Multi-start strategy: try several different initializations
        # For Rastrigin, we want to explore the landscape broadly first
        n_restarts = 10
        budget_per_restart = max(1, (self.budget - 1) // n_restarts)

        for r in range(n_restarts):
            if total_calls >= self.budget:
                break
            if time.time() > deadline:
                break

            # Different initialization strategies
            if r == 0:
                x = rng.uniform(-5, 5, n)
            elif r == 1:
                x = np.zeros(n)
            elif r == 2:
                x = rng.uniform(-3, 3, n)
            elif r == 3:
                x = np.ones(n) * 3.0
            elif r == 4:
                x = -np.ones(n) * 3.0
            elif r == 5:
                x = rng.uniform(-4, 4, n)
            elif r == 6:
                x = np.array([4.0, -4.0, 4.0, -4.0, 4.0])
            elif r == 7:
                x = np.array([-4.0, 4.0, -4.0, 4.0, -4.0])
            elif r == 8:
                x = rng.uniform(-5, 5, n) * 0.5
            else:
                x = rng.uniform(-2, 2, n)

            fx = func(x)
            total_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # CMA-ES from this start
            sigma = 2.0
            C = np.eye(n)

            remaining = self.budget - total_calls
            for step in range(min(budget_per_restart, remaining)):
                if time.time() > deadline:
                    break
                if total_calls >= self.budget:
                    break

                x, fx, sigma, C, _, calls_used = self._cma_step(
                    x, sigma, C, None, rng, func, deadline
                )
                total_calls += calls_used

                if fx is not None and fx < best_f:
                    best_f = fx
                    best_x = x.copy()

            # Early stopping if we're doing well
            if best_f < 0.5:
                break

        # Final local refinement with small step size
        if best_x is not None and total_calls < self.budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            C = np.eye(n)
            remaining = self.budget - total_calls
            for step in range(min(remaining, 50)):
                if time.time() > deadline:
                    break
                if total_calls >= self.budget:
                    break
                x, fx, sigma, C, _, calls_used = self._cma_step(
                    x, sigma, C, None, rng, func, deadline
                )
                total_calls += calls_used
                if fx is not None and fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
