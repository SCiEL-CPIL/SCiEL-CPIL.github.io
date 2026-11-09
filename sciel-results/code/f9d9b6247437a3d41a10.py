"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with 1/5 success rule and restarts for BBOB f24 (Lunacek bi-Rastrigin)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin before 300s limit

        n = self.dim
        total_budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        # CMA-ES with 1/5 success rule: structural change from (1+1)-ES
        # Uses covariance matrix adaptation and proper step-size control
        def cma_es(x0, budget, current_calls, deadline):
            nonlocal best_x, best_f, calls_made
            
            x = x0.copy()
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 1.0
            C = np.eye(n)
            C_inv = np.eye(n)
            
            # Evolution paths
            ps = np.zeros(n)
            pc = np.zeros(n)
            
            # Time constants
            mu = 1  # (1+1) strategy
            cmu = 1.0 / (4 * n + 4)
            cs = (mu + 2) / (n + mu + 5)
            c1 = 2 / ((n + 2)**2 + 10)
            cmu = 2 / ((n + 2)**2 + 2 * mu)
            damps = 1 + 2 * max(0, sqrt((mu - 1) / (n + 1) - 1)) + cs
            dampc = 1 + 2 * max(0, sqrt(mu / (n + 1) - 1)) - cs

            # 1/5 success rule state
            successes = 0
            trials = 0

            # Precompute sqrt of C for sampling
            def _chol_update(C):
                try:
                    L = np.linalg.cholesky(C)
                    return L
                except np.linalg.LinAlgError:
                    # Add jitter
                    C_jit = C + 1e-10 * np.eye(n)
                    try:
                        L = np.linalg.cholesky(C_jit)
                        return L
                    except:
                        return np.eye(n)

            L = _chol_update(C)

            while calls_made < total_budget:
                if time.time() > deadline:
                    break

                # Sample new point
                z = rng.standard_normal(n)
                y = x + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1

                # 1/5 success rule: adapt sigma
                trials += 1
                if fy < fx:
                    successes += 1
                    x = y
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()

                if trials >= 15:
                    success_rate = successes / trials
                    if success_rate > 0.2:
                        sigma *= 1.5
                    elif success_rate < 0.1:
                        sigma *= 0.7
                    successes = 0
                    trials = 0
                    sigma = np.clip(sigma, 1e-8, 1e6)

                # Update evolution paths
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * cmu) * (z / sigma)
                ps_norm = np.linalg.norm(ps)
                hsig = ps_norm / np.sqrt(1 - (1 - cs)**(2 * (calls_made - total_budget + budget))) < (1.4 + 2 / (n + 1)) * np.sqrt(1 - (1 - cs)**2)
                pc = (1 - cs) * pc + hsig * np.sqrt(cs * (2 - cs) * cmu) * (y - x) / sigma
                pc_norm = np.linalg.norm(pc)

                # Update covariance matrix
                delta_hsig = (1 - hsig) * cs * (2 - cs)
                C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + delta_hsig * C) + cmu * np.outer(y - x, y - x) / sigma**2

                # Ensure symmetry
                C = 0.5 * (C + C.T)

                # Update Cholesky
                L = _chol_update(C)

            return x, fx

        # Multiple restarts with different initializations
        num_restarts = 3
        budgets = [int(total_budget * 0.45), int(total_budget * 0.30), total_budget - int(total_budget * 0.45) - int(total_budget * 0.30)]

        for restart_idx in range(num_restarts):
            if calls_made >= total_budget:
                break
            if time.time() > deadline:
                break

            budget_this_restart = min(budgets[restart_idx], total_budget - calls_made)
            if budget_this_restart < 5:
                break

            # Different initialization strategies
            if restart_idx == 0:
                x0 = np.zeros(n)
            elif restart_idx == 1:
                x0 = rng.uniform(-3, 3, n)
            else:
                x0 = rng.uniform(-5, 5, n)

            cma_es(x0, budget_this_restart, calls_made, deadline)

        # Ensure we return valid values
        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls_made += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
