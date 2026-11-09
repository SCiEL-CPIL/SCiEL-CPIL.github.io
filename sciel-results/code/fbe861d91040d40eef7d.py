"""Rewritten optimizer for BBOB f24 (Lunacek bi-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based CMA-ES with adaptive restarts and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # 20s safety margin before 300s limit
        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget

        # Global best tracking
        best_x = None
        best_f = np.inf
        calls_made = 0

        # CMA-ES parameters
        lambda_ = max(4, int(4 + 3 * np.log(d)))
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = (2 + 2.0 / (d + 1.3)) / ((d + 1.8) * (d + mu_eff / 2))
        c_mu = min(1 - c_1, 2 * (sqrt_mu_eff - 2) / ((d + 2) * (mu_eff - 2 + sqrt_mu_eff)))
        chi_d = np.sqrt(d) * (1 - 1.0 / (4 * d) + 1.0 / (21 * d ** 2))
        sqrt_mu_eff = np.sqrt(mu_eff) if (mu_eff := (weights @ weights) ** -0.5) else 1.0

        def run_cmaes(x0, sigma0, max_iter):
            nonlocal best_x, best_f, calls_made
            x = x0.copy()
            sigma = sigma0
            ps = np.zeros(d)
            pc = np.zeros(d)
            B = np.eye(d)
            D = np.ones(d)
            C = np.eye(d)
            for _ in range(max_iter):
                if time.time() > deadline:
                    return
                # Generate candidates
                ys = []
                fs = []
                for k in range(lambda_):
                    if time.time() > deadline:
                        return
                    z = rng.standard_normal(d)
                    y = x + sigma * (B @ (D * z))
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls_made += 1
                    ys.append(y)
                    fs.append(fy)
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                # Sort by fitness
                order = np.argsort(fs)
                ys_sorted = np.array(ys)[order]
                fs_sorted = np.array(fs)[order]
                # Update mean
                y_w = sum(weights[i] * ys_sorted[i] for i in range(mu))
                # Update step-size control
                d_hat = B @ (D * (B.T @ (ps / sigma)))
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * d_hat
                # Update covariance matrix adaptation
                c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
                c_1 = (2 + 2.0 / (d + 1.3)) / ((d + 1.8) * (d + mu_eff / 2))
                c_mu = min(1 - c_1, 2 * (np.sqrt(mu_eff) - 2) / ((d + 2) * (mu_eff - 2 + np.sqrt(mu_eff))))
                h_sigma = np.linalg.norm(ps) < (1.4 + 1 / d) * np.sqrt((1 - (1 - c_sigma) ** (2 * max_iter)))
                pc = (1 - c_c) * pc + h_sigma * c_c / sigma * (y_w - x)
                delta_h = (1 - (1 - c_c) ** (2 * max_iter)) * c_c * (2 - c_c)
                C = (1 - c_1 - c_mu + delta_h) * C + c_1 * (np.outer(pc, pc) - C) + \
                    c_mu / sigma ** 2 * sum(weights[i] * np.outer(ys_sorted[i] - x, ys_sorted[i] - x) for i in range(mu))
                # Ensure symmetry and positive semi-definite
                C = 0.5 * (C + C.T)
                # Eigen decomposition
                try:
                    D2, B2 = np.linalg.eigh(C)
                    D2 = np.sqrt(np.maximum(D2, 1e-20))
                    B = B2
                    D = D2
                except np.linalg.LinAlgError:
                    break
                # Update sigma
                sigma *= np.exp(c_sigma / chi_d * (np.linalg.norm(ps) / chi_d - 1))
                sigma = np.clip(sigma, 1e-10, 50.0)
                # Update mean
                x = y_w
                x = np.clip(x, -5, 5)
            return x, sigma

        # Restart strategy: multiple short CMA-ES runs with different initializations
        n_restarts = 5
        calls_per_restart = max(10, budget // n_restarts)
        
        for r in range(n_restarts):
            if calls_made >= budget or time.time() > deadline:
                break
            if r == 0:
                x0 = np.zeros(d)
                sigma0 = 2.0
            else:
                # Diversify restarts
                x0 = rng.uniform(-4, 4, d)
                sigma0 = 1.5 + r * 0.3
            remaining = budget - calls_made
            max_iter = min(calls_per_restart, remaining)
            if max_iter < lambda_:
                max_iter = max(2, max_iter // lambda_)
            if max_iter < 1:
                break
            run_cmaes(x0, sigma0, max_iter)

        # Final local refinement if time and budget remain
        if best_x is not None and calls_made < budget and time.time() < deadline:
            # Simple local search around best point
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            for _ in range(min(50, budget - calls_made)):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy
                    if fy < best_f:
                        best_f = fy
                        best_x = x.copy()
                    sigma *= 1.2
                else:
                    sigma *= 0.8
                if sigma < 1e-6:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
