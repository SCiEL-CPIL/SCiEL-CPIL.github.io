"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart 1+1-CMA-ES with deadline guard for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Global best tracking
        global_best_x = None
        global_best_f = np.inf

        def track(x, f):
            nonlocal global_best_x, global_best_f
            if f < global_best_f:
                global_best_f = f
                global_best_x = x.copy()

        def safe_func(x):
            nonlocal total_calls
            fx = func(x)
            total_calls += 1
            track(x, fx)
            return fx

        total_calls = 0

        def cma_11(x0, sigma0, max_calls, rng_local):
            """1+1 CMA-ES with covariance adaptation."""
            nonlocal total_calls
            x = x0.copy()
            sigma = sigma0
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            C = np.eye(n)

            # CMA-ES parameters for 1+1
            cs = 1.0 / (n + 3)
            cc = (4 + 1 / n) / (n + 4 + 2 / n)
            c1 = 2 / ((n + 1.3) ** 2 + 1)
            cmu = 0.0  # No rank-mu for 1+1
            mueff = 1.0
            hsig = (1 - cc) / np.sqrt(1 - cc ** 2)

            def compute_c_inv_sqrt(C_mat):
                eigvals, eigvecs = np.linalg.eigh(C_mat)
                eigvals = np.maximum(eigvals, 1e-15)
                c_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                return c_inv_sqrt

            C_inv_sqrt = compute_c_inv_sqrt(C)

            calls_used = 0
            while calls_used < max_calls:
                if time.time() > deadline - 2:
                    break

                # Generate trial point
                z = rng_local.standard_normal(n)
                y = x + sigma * C_inv_sqrt @ z
                y = np.clip(y, -5, 5)

                fy = func(y)
                calls_used += 1
                total_calls += 1
                track(y, fy)

                # Update mean if successful
                if fy < func(x):
                    x_old = x.copy()
                    x = y.copy()
                    d = x - x_old
                    C_inv_sqrt_d = C_inv_sqrt @ d
                    norm_d = np.linalg.norm(C_inv_sqrt_d)

                    if norm_d > 1e-12:
                        # Update sigma
                        p_sigma = (1 - cs) * p_sigma + cs * (norm_d / np.sqrt(2) - 1) * C_inv_sqrt_d / norm_d
                        sigma *= np.exp(cs / (n + 2) * (norm_d / np.sqrt(2) - 1))
                        sigma = np.clip(sigma, 1e-12, 100.0)

                        # Update C: rank-1 update
                        C = (1 + c1 * cc) * C + c1 * np.outer(d, d)

                        # Update p_c
                        p_c = (1 - cc) * p_c + cc * hsig * (norm_d / np.sqrt(2)) * C_inv_sqrt_d / (norm_d + 1e-12)

                        # Update C with p_c (path covariance adaptation)
                        C = (1 - c1 * cc * (2 - cc)) * C + c1 * (1 + cc * (2 - cc)) * np.outer(p_c, p_c)

                        # Ensure C is positive definite
                        eigvals, eigvecs = np.linalg.eigh(C)
                        if np.min(eigvals) < 1e-12:
                            C = eigvecs @ np.diag(np.maximum(eigvals, 1e-12)) @ eigvecs.T

                        # Recompute C^{-1/2}
                        C_inv_sqrt = compute_c_inv_sqrt(C)
                else:
                    # Failed step: still update p_sigma and p_c for exploration
                    # But don't move x
                    pass

            return x, global_best_f, calls_used

        # Phase 1: Random initialization
        n_random = min(30, budget // 10)
        for i in range(n_random):
            if total_calls >= budget:
                break
            if time.time() > deadline - 10:
                break
            x_try = rng.uniform(-5, 5, n)
            f_try = func(x_try)
            total_calls += 1
            track(x_try, f_try)

        if global_best_x is None:
            # Fallback: return origin
            global_best_x = np.zeros(n)
            global_best_f = func(global_best_x)
            total_calls += 1

        # Phase 2: Run 1+1 CMA-ES from the best starting point
        remaining = budget - total_calls
        if remaining > 20:
            x_opt, f_opt, calls_used = cma_11(global_best_x.copy(), 2.0, remaining, rng)
            if f_opt < global_best_f:
                global_best_f = f_opt
                global_best_x = x_opt.copy()

        # Phase 3: If budget remains, do a second CMA-ES run with a smaller sigma
        if total_calls < budget and time.time() < deadline - 20:
            remaining = budget - total_calls
            if remaining > 20:
                x_opt, f_opt, calls_used = cma_11(global_best_x.copy(), 0.5, remaining, rng)
                if f_opt < global_best_f:
                    global_best_f = f_opt
                    global_best_x = x_opt.copy()

        # Phase 4: If still budget remains, try a third restart from a different random point
        if total_calls < budget and time.time() < deadline - 15:
            x_try = rng.uniform(-5, 5, n)
            f_try = func(x_try)
            total_calls += 1
            track(x_try, f_try)

            remaining = budget - total_calls
            if remaining > 20:
                x_opt, f_opt, calls_used = cma_11(global_best_x.copy(), 1.5, remaining, rng)
                if f_opt < global_best_f:
                    global_best_f = f_opt
                    global_best_x = x_opt.copy()

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
