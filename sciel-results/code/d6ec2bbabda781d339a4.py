"""CMA-ES-like covariance-adapting multi-start optimizer for BBOB f18 (Schaffers F7)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES-like single-point search with covariance adaptation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    # ---- CMA-ES-like single-point search with rank-1/rank-0 covariance updates ----
    def _cma_local(self, func, rng, x0, sigma0, max_calls, deadline):
        dim = self.dim
        calls = 0

        def _eval(x):
            nonlocal calls
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            calls += 1
            return x, f

        x, fx = _eval(np.asarray(x0, dtype=float))
        sigma = max(sigma0, 1e-6)
        C = np.eye(dim)
        Cinv = np.eye(dim)
        Csqr = np.eye(dim)

        # path / adaptation parameters
        c_sigma = (4 + dim / 3.0) / (dim + 4.0) ** 2
        d_sigma = 1.0 + 2.0 * max(0.0, (dim - 2.0) / (dim + 2.0)) + c_sigma
        c_c = (4 + dim / 1.8) / (dim + 4.2)
        c_cov = (2 + 2.0 / (dim + 1.0)) / (dim + 3.5)
        mu = 0.5
        chi_n = np.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim))

        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        best_x = x.copy()
        best_f = fx

        while calls < max_calls and time.time() < deadline:
            z = rng.standard_normal(dim)
            y = x + sigma * (Csqr @ z)
            y, fy = _eval(y)

            if fy < fx:
                # accepted step
                x_new = y
                fx_new = fy
                s = (x_new - x) / sigma
                C = C + mu * np.outer(s, s) * (1.0 + (2.0 - mu) / (dim + 1.0))
                Cinv = Cinv - mu * (np.outer(s, s) @ Cinv + Cinv @ np.outer(s, s)) / (dim + 1.0) \
                    + mu * (2.0 - mu) * Cinv @ np.outer(s, s) @ Cinv
                Csqr = np.linalg.cholesky(C + 1e-9 * np.eye(dim))
                p_sigma = (1.0 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu) * (Cinv @ s)
                p_c = (1.0 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu) * s
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1.0))
                x, fx = x_new, fx_new
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                # rejected step: only update paths (no covariance change)
                s = (y - x) / sigma
                p_c = (1.0 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu) * s
                hsig = (np.linalg.norm(p_sigma) / np.sqrt(1.0 - (1.0 - c_sigma) ** (2 * (calls + 1)))) \
                    < (1.4 + 2.0 / (dim + 1.0)) * chi_n
                if hsig:
                    p_sigma = (1.0 - c_sigma) * p_sigma
                C = C + (1.0 - c_cov) * (2.0 + mu * (2.0 - mu)) * (np.outer(p_c, p_c) - C) \
                    * (1.0 / (dim + 1.0))
                # keep C positive definite
                try:
                    Csqr = np.linalg.cholesky(C + 1e-9 * np.eye(dim))
                    Cinv = np.linalg.inv(C + 1e-9 * np.eye(dim))
                except np.linalg.LinAlgError:
                    Csqr = np.eye(dim)
                    Cinv = np.eye(dim)
                    C = np.eye(dim)

            sigma = float(np.clip(sigma, 1e-10, 5.0))

        return best_x, best_f, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(20240517)

        best_x = np.zeros(dim)
        best_f = np.inf
        total_calls = 0

        # ---- Phase 1: random basin sampling (ill-conditioned => many basins) ----
        n_samples = min(80, budget // 6)
        for _ in range(n_samples):
            if total_calls >= budget or time.time() > deadline - 5.0:
                break
            x = rng.uniform(-4.0, 4.0, size=dim)
            fx = func(x)
            total_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # ---- Phase 2: multi-start CMA-ES-like local search with restarts ----
        # Build a candidate list: best sample first, then progressively worse,
        # plus fresh random points used as restarts on stagnation.
        remaining = budget - total_calls
        per_start = max(15, remaining // 8)

        # Stagnation detection: if no improvement over a window, restart.
        stagnation_window = 30
        last_improve_call = total_calls

        start_idx = 0
        while total_calls < budget and time.time() < deadline - 5.0:
            # Choose starting point: cycle through samples + fresh random on restart
            if start_idx == 0:
                x0 = best_x.copy()
                sigma0 = 1.0
            else:
                # fresh random point, biased toward the best region
                x0 = best_x + rng.normal(0.0, 1.5, size=dim)
                x0 = np.clip(x0, -4.0, 4.0)
                sigma0 = 0.8

            calls_before = total_calls
            x_best, f_best, calls_used = self._cma_local(
                func, rng, x0, sigma0, per_start, deadline
            )
            total_calls += calls_used

            if f_best < best_f:
                best_f = f_best
                best_x = x_best.copy()
                last_improve_call = total_calls

            # Stagnation: if this whole start made no progress, do a fresh restart
            # (structural restart, not a fixed count).
            if total_calls - last_improve_call >= stagnation_window:
                # try a fresh random basin before continuing
                if total_calls < budget - per_start and time.time() < deadline - 5.0:
                    x = rng.uniform(-4.0, 4.0, size=dim)
                    fx = func(x)
                    total_calls += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                        last_improve_call = total_calls

            start_idx += 1

        # ---- Phase 3: final fine polish from the global best ----
        remaining = budget - total_calls
        if remaining > 5 and time.time() < deadline - 5.0:
            x_local = best_x.copy()
            fx_local = best_f
            sigma_local = 0.05
            local_calls = 0
            while local_calls < remaining and time.time() < deadline - 5.0:
                z = rng.standard_normal(dim)
                y = x_local + sigma_local * z
                y = np.clip(y, -5.0, 5.0)
                fy = func(y)
                local_calls += 1
                total_calls += 1
                if fy < fx_local:
                    x_local = y
                    fx_local = fy
                    sigma_local *= 1.05
                    if fx_local < best_f:
                        best_f = fx_local
                        best_x = x_local.copy()
                else:
                    sigma_local *= 0.92
                sigma_local = float(np.clip(sigma_local, 1e-8, 5.0))

        return best_x, best_f
# EVOLVE-BLOCK-END
