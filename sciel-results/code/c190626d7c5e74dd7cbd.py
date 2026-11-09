"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with time-based budget management and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, budget, deadline, rng):
        """Run CMA-ES from x0 with given budget, stopping at deadline."""
        dim = self.dim
        n = dim
        # Population size
        lambda_ = 4 + int(3 * np.log(n))
        # Weights
        mu = lambda_ // 2
        weights = np.log(np.arange(lambda_, 0, -1))
        weights /= weights.sum()
        nu_eff = 1 / np.sum(weights ** 2)
        c_sigma = (nu_eff + 2) / (n + nu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((nu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + nu_eff / n) / (n + 4 + 2 * nu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + nu_eff)
        c_mu = min(1 - c_1, 2 * ((nu_eff + 2) / (n + 2) - 1))

        # State
        x_mean = x0.copy()
        sigma = 2.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        best_x = x_mean.copy()
        best_f = func(best_x)
        calls = 1

        # Generate initial population
        for i in range(lambda_):
            if time.time() > deadline:
                break
            y = x_mean + sigma * (C @ rng.standard_normal(n))
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls += 1
            if fy < best_f:
                best_f = fy
                best_x = y.copy()

        # Sort by fitness
        # We need to store all evaluations
        # Let me restructure: generate, evaluate, sort, update
        # Actually, let's do it properly with a loop

        # Reset and do proper CMA-ES
        x_mean = x0.copy()
        sigma = 2.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        best_x = x_mean.copy()
        best_f = func(best_x)
        calls = 1

        while calls < budget:
            if time.time() > deadline:
                break

            # Generate population
            ys = np.zeros((lambda_, n))
            fys = np.zeros(lambda_)
            for i in range(lambda_):
                if time.time() > deadline:
                    break
                if calls >= budget:
                    break
                z = rng.standard_normal(n)
                ys[i] = x_mean + sigma * (C @ z)
                ys[i] = np.clip(ys[i], -5, 5)
                fys[i] = func(ys[i])
                calls += 1
                if fys[i] < best_f:
                    best_f = fys[i]
                    best_x = ys[i].copy()

            if calls >= budget:
                break

            # Sort
            idx = np.argsort(fys)
            ys_sorted = ys[idx]
            fys_sorted = fys[idx]

            # Weighted mean of top mu
            x_new = np.zeros(n)
            for i in range(mu):
                x_new += weights[i] * ys_sorted[i]

            # Update step size
            y_old = (x_new - x_mean) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * nu_eff) * (C @ y_old) / chi_n
            if np.linalg.norm(ps) / chi_n < (1.4 + 1 / n) * np.sqrt(1 - (1 - c_sigma) ** (2 * (calls // lambda_ + 1))):
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

            # Update mean
            x_mean = x_new

            # Update covariance
            delta_sigma = (np.linalg.norm(ps) / chi_n > (1.4 + 1 / n) * np.sqrt(1 - (1 - c_sigma) ** 2)) * 1
            rank_one = np.outer(y_old, y_old)
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                yi = (ys_sorted[i] - x_new) / sigma
                rank_mu += weights[i] * np.outer(yi, yi)
            C = (1 - c_1 - c_mu) * C + c_1 * (rank_one + delta_sigma * np.outer(ps, ps)) + c_mu * rank_mu

            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2

        return best_x, best_f, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 50s safety margin before 330s limit

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        total_calls = 0

        # Try multiple restarts
        n_restarts = 5
        budget_per_restart = budget // n_restarts

        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            remaining_budget = budget - total_calls
            if remaining_budget <= 0:
                break

            # Initialize
            if restart == 0:
                x0 = rng.uniform(-5, 5, dim)
            else:
                # Perturb the best solution found so far
                x0 = best_x + 0.5 * rng.standard_normal(dim)
                x0 = np.clip(x0, -5, 5)

            x, f, calls_used = self._cma_es(func, x0, remaining_budget, deadline, rng)
            total_calls += calls_used

            if f < best_f:
                best_f = f
                best_x = x

            # Check if we've converged (very small improvement)
            if best_f < 1e-10:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
