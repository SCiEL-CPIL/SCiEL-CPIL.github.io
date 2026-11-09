"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-strategy optimizer: CMA-ES with small population + Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        def time_left():
            return time.time() < deadline

        # Phase 1: Random sampling to find a good starting point
        best_x = None
        best_f = float('inf')
        n_init = 20
        for _ in range(n_init):
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            if not time_left():
                break

        # Phase 2: CMA-ES with small population (lambda=8)
        # This gives more generations within budget
        x = best_x.copy()
        sigma = 1.0
        best_f_cma = best_f
        best_x_cma = best_x.copy()

        lambda_ = 8
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights @ weights)
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (self.dim + 3)) + c_sigma

        B = np.eye(self.dim)
        D = np.ones(self.dim)
        p_sigma = np.zeros(self.dim)
        p_c = np.zeros(self.dim)
        calls_used = n_init
        generations = 0

        while calls_used < self.budget and time_left():
            # Generate lambda candidates
            candidates = []
            fvals = []
            for i in range(lambda_):
                z = rng.standard_normal(self.dim)
                y = x + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_used += 1
                candidates.append(y)
                fvals.append(fy)
                if fy < best_f_cma:
                    best_f_cma = fy
                    best_x_cma = y.copy()
                if not time_left():
                    break

            if calls_used >= self.budget or not time_left():
                break

            # Sort by fitness
            idx = np.argsort(fvals)
            candidates = [candidates[i] for i in idx]
            fvals = [fvals[i] for i in idx]

            # Update mean
            x_new = sum(weights[i] * candidates[i] for i in range(mu))

            # Update step-size control
            z_old = np.linalg.solve(B.T, (x_new - x) / sigma)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_old
            hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (generations + 1))) / np.sqrt(mu_eff) < 1.4 + 2 / (self.dim + 1)
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 50.0)

            # Update covariance matrix
            p_c = (1 - c_c) * p_c + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x) / sigma
            hup = (np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c) ** (generations + 1))) / np.sqrt(mu_eff) < (1.7 + 2 / (self.dim + 1))
            rank_mu = np.zeros((self.dim, self.dim))
            for i in range(mu):
                y_i = (candidates[i] - x) / sigma
                rank_mu += weights[i] * np.outer(y_i, y_i)
            rank_one = np.outer(p_c, p_c)
            delta_h = (1 - hup) * c_c * (2 - c_c)
            C = (1 + c_c * (2 - c_c - mu_eff) + delta_h) * C if hasattr(C, '__len__') else np.eye(self.dim)
            # Simpler: just use identity if we don't maintain full C properly
            # Let's use a simpler update
            C = np.eye(self.dim) if generations == 0 else C
            C = (1 - c_c * (2 - c_c - mu_eff)) * C + c_c * rank_one + delta_h * (np.outer(p_c, p_c) - C) if generations > 0 else np.eye(self.dim)
            # Ensure PSD
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            B, D = np.linalg.eigh(C)
            D = np.sqrt(D)

            x = x_new
            generations += 1

            # If progress stalled, try a restart with larger sigma
            if generations % 10 == 0 and sigma < 0.01:
                sigma = 0.5

        best_x = best_x_cma
        best_f = best_f_cma

        # Phase 3: Nelder-Mead polish on the best point
        if calls_used < self.budget - 50 and time_left():
            try:
                from scipy.optimize import minimize
                # Create a wrapper that counts calls
                nm_calls = [0]
                nm_budget = self.budget - calls_used - 10

                def nm_func(x):
                    if nm_calls[0] >= nm_budget:
                        return best_f
                    nm_calls[0] += 1
                    fx = func(x)
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    return fx

                # Initial simplex
                simplex = [best_x.copy()]
                for i in range(self.dim):
                    xi = best_x.copy()
                    step = 0.1 if abs(best_x[i]) > 1e-8 else 0.1
                    xi[i] += step
                    xi = np.clip(xi, -5, 5)
                    simplex.append(xi)

                res = minimize(nm_func, best_x, method='Nelder-Mead',
                               options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 100})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # Final: ensure we have a valid best
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
