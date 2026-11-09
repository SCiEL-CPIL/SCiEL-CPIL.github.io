"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer for sharp ridge (f13) with restarts and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cmaes(self, func, x0, sigma0, budget, rng, deadline):
        """Run CMA-ES from x0 with initial step sigma0 for up to budget evaluations."""
        n = self.dim
        x = np.array(x0, dtype=float)
        f_best = func(x)
        calls = 1
        if calls >= budget:
            return x, f_best, calls

        # CMA-ES parameters
        lam = max(4, 2 * n + 2)
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w**2)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, (mu_eff - 2 + 2 / mu_eff) / ((n + 2)**2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_mu
        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n**2))

        sigma = sigma0
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        C = np.eye(n)
        C_sqrt = np.eye(n)
        C_sqrt_inv = np.eye(n)
        # Precompute eigendecomposition
        evals, evecs = np.linalg.eigh(C)
        eigvals = np.sqrt(np.maximum(evals, 1e-12))
        eigvals_inv = 1.0 / np.maximum(evals, 1e-12)
        # Track for periodic recompute
        step_since_eig = 0

        # History for stall detection
        best_f = f_best
        no_improve = 0
        stall_threshold = 100

        for _ in range(budget - calls):
            if time.time() > deadline:
                break

            # Generate lambda candidates
            ys = rng.standard_normal((lam, n))
            # Apply C^{1/2}
            # Use eigendecomposition: C^{1/2} = V diag(sqrt(lambda)) V^T
            # y_transformed = V diag(sqrt(lambda)) V^T y
            ys_transformed = (ys @ evecs.T * eigvals) @ evecs
            ps = [np.clip(x + sigma * y, -5, 5) for y in ys_transformed]

            # Evaluate
            fs = [func(p) for p in ps]
            calls += lam

            # Sort by fitness
            idx = np.argsort(fs)
            ps_sorted = [ps[i] for i in idx]
            fs_sorted = [fs[i] for i in idx]

            # Update best
            if fs_sorted[0] < best_f:
                best_f = fs_sorted[0]
                x = ps_sorted[0].copy()
                no_improve = 0
            else:
                no_improve += 1

            # Update mean
            x_old = x.copy()
            x = sum(w[i] * ps_sorted[i] for i in range(mu))

            # Update step size (CSA)
            y_sigma = (x - x_old) / (sigma * chi_n)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            sigma *= np.exp((d_sigma / chi_n) * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 50.0)

            # Update evolution paths
            p_c = (1 - c_c) * p_c + c_c * ((x - x_old) / sigma)

            # Rank-mu update
            H = np.zeros((n, n))
            for i in range(mu):
                H += w[i] * np.outer(ys_transformed[i], ys_transformed[i])
            # Update C
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + c_c * (2 - c_c) * np.eye(n)) + c_mu * H

            # Periodic eigendecomposition
            step_since_eig += 1
            if step_since_eig >= 10:
                evals, evecs = np.linalg.eigh(C)
                eigvals = np.sqrt(np.maximum(evals, 1e-12))
                eigvals_inv = 1.0 / np.maximum(evals, 1e-12)
                step_since_eig = 0

            if calls >= budget:
                break

            # Stall detection: restart if no improvement for a while
            if no_improve > stall_threshold:
                break

        return x, best_f, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit
        n = self.dim
        budget = self.budget

        # Multiple restarts with different initializations
        # For sharp ridge, try several strategies
        best_x = None
        best_f = np.inf
        total_calls = 0

        # Strategy 1: CMA-ES from center
        rng = np.random.default_rng(42)
        x0 = np.zeros(n)
        x, f, calls = self._cmaes(func, x0, 2.0, budget, rng, deadline)
        total_calls += calls
        if f < best_f:
            best_f, best_x = f, x.copy()

        if time.time() > deadline or total_calls >= budget:
            return best_x, best_f

        # Strategy 2: CMA-ES from a random point
        remaining = budget - total_calls
        if remaining > 50:
            rng2 = np.random.default_rng(123)
            x0 = rng2.uniform(-3, 3, n)
            x, f, calls = self._cmaes(func, x0, 1.0, remaining, rng2, deadline)
            total_calls += calls
            if f < best_f:
                best_f, best_x = f, x.copy()

        if time.time() > deadline or total_calls >= budget:
            return best_x, best_f

        # Strategy 3: Local refinement around best point with small step
        remaining = budget - total_calls
        if remaining > 20 and best_x is not None:
            rng3 = np.random.default_rng(777)
            x, f, calls = self._cmaes(func, best_x, 0.5, remaining, rng3, deadline)
            total_calls += calls
            if f < best_f:
                best_f, best_x = f, x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
