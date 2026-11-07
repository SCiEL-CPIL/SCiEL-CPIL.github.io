"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with Rank-1 Covariance Adaptation and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        best_fx = fx
        best_x = x.copy()

        sigma = 2.0
        C = np.eye(self.dim)
        n = self.dim

        calls = 1
        # Reserve last 60 calls for polish
        polish_budget = min(60, self.budget // 8)
        main_budget = self.budget - polish_budget

        # Window for success rate tracking
        window_size = 15
        successes = []  # list of 1s and 0s

        while calls < main_budget:
            if time.time() > deadline:
                break

            try:
                L = np.linalg.cholesky(C + 1e-10 * np.eye(n))
            except np.linalg.LinAlgError:
                C = np.eye(n)
                L = np.eye(n)

            z = rng.standard_normal(n)
            step = sigma * np.dot(L, z)
            y = np.clip(x + step, -5, 5)
            fy = func(y)
            calls += 1

            success = fy < fx
            successes.append(1 if success else 0)
            if len(successes) > window_size:
                successes.pop(0)

            if success:
                x = y
                fx = fy
                if fy < best_fx:
                    best_fx = fy
                    best_x = x.copy()

                # Rank-1 covariance update
                mu_c = 0.25
                C = (1 - mu_c) * C + mu_c * np.outer(z, z)
                trace_C = np.trace(C)
                if trace_C > 1e-10:
                    C = C * (n / trace_C)
            else:
                pass

            # Sigma adaptation based on success rate in window
            if len(successes) >= 5:
                success_rate = np.mean(successes)
                # Target ~1/3 success rate for (1+1)-ES
                if success_rate > 0.4:
                    sigma *= 1.05
                elif success_rate < 0.2:
                    sigma *= 0.95
                else:
                    sigma *= 1.01
            sigma = np.clip(sigma, 1e-5, 10.0)

            # Periodic re-centering to prevent covariance drift
            if calls % 30 == 0:
                C = 0.9 * C + 0.1 * np.eye(n)
                trace_C = np.trace(C)
                if trace_C > 1e-10:
                    C = C * (n / trace_C)

            # Periodic eigenvalue clipping for numerical stability
            if calls % 100 == 0:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 0.01, 50.0)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                trace_C = np.trace(C)
                if trace_C > 1e-10:
                    C = C * (n / trace_C)

        # Polish phase 1: Coordinate descent for the anisotropic bent cigar
        if calls < self.budget and time.time() < deadline:
            coord_budget = min(20, self.budget - calls)
            x_cd = best_x.copy()
            fx_cd = best_fx
            step_sizes = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
            
            for _ in range(coord_budget):
                if time.time() > deadline:
                    break
                improved = False
                for i in range(n):
                    for ss in step_sizes:
                        # Try positive direction
                        y_pos = x_cd.copy()
                        y_pos[i] = np.clip(x_cd[i] + ss, -5, 5)
                        fy_pos = func(y_pos)
                        calls += 1
                        if fy_pos < fx_cd:
                            x_cd = y_pos
                            fx_cd = fy_pos
                            improved = True
                            break
                        
                        # Try negative direction
                        y_neg = x_cd.copy()
                        y_neg[i] = np.clip(x_cd[i] - ss, -5, 5)
                        fy_neg = func(y_neg)
                        calls += 1
                        if fy_neg < fx_cd:
                            x_cd = y_neg
                            fx_cd = fy_neg
                            improved = True
                            break
                        if calls >= self.budget:
                            break
                    if calls >= self.budget:
                        break
                if not improved:
                    break
                
            if fx_cd < best_fx:
                best_fx = fx_cd
                best_x = x_cd.copy()

        # Polish phase 2: Nelder-Mead with tighter tolerances
        if calls < self.budget and time.time() < deadline:
            try:
                from scipy.optimize import minimize
                remaining = self.budget - calls
                if remaining > 10:
                    result = minimize(func, best_x, method='Nelder-Mead',
                                      options={'maxiter': remaining, 'xatol': 1e-12, 'fatol': 1e-14,
                                               'adaptive': True})
                    if result.fun < best_fx:
                        best_fx = result.fun
                        best_x = result.x.copy()
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
