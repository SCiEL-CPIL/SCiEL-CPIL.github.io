"""Optimizer for BBOB-style f17 (Schaffers F7) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280  # safety margin before 300s limit
        d = self.dim
        total_budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        # We'll do a CMA-ES with restarts
        # Each restart gets a portion of the budget
        # Use adaptive budget allocation: first run gets more, restarts get less

        # CMA-ES parameters
        # For dim=5, lambda is typically 4 + floor(3 * log(d)) = 4 + 3*1.6 = 8 -> use 8 or 10
        lam = 10
        mu = lam // 2  # selection mass

        # Standard CMA-ES parameters
        cc = (4 + d / (d + 4)) / ((d + 1.3) + 2 / (d + mu / 2))
        cs = (mu / 2 + 1) / (d + mu / 2 + 2)
        c1 = 2 / ((d + 1.3) ** 2 + mu)
        cmu = min(1 - c1, 2 * (mu / 2 - 1) / ((d + 2) ** 2 + mu))
        damps = 1 + 2 * max(0, np.sqrt((mu / 2 - 1) / (d + 1)) - 1) + cs

        def cmaes_run(x0, sigma0, budget_for_this_run, rng):
            nonlocal best_x, best_f, calls_made

            x = x0.copy()
            sigma = sigma0
            C = np.eye(d)
            pc = np.zeros(d)
            ps = np.zeros(d)
            chiN = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

            f_best = np.inf
            x_best = x.copy()

            for gen in range(budget_for_this_run // lam + 1):
                if time.time() > deadline:
                    break
                if calls_made >= total_budget:
                    break

                # Generate lambda offspring
                B = np.linalg.cholesky(C)
                # Actually, C should be PSD; use eigh for stability
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                sqrt_eigvals = np.sqrt(eigvals)
                B = eigvecs @ np.diag(sqrt_eigvals) @ eigvecs.T

                offspring = []
                for i in range(lam):
                    zi = rng.standard_normal(d)
                    yi = x + sigma * (B @ zi)
                    yi = np.clip(yi, -5, 5)
                    fi = func(yi)
                    calls_made += 1
                    offspring.append((fi, yi))

                # Sort offspring
                offspring.sort(key=lambda t: t[0])

                # Update best
                if offspring[0][0] < f_best:
                    f_best = offspring[0][0]
                    x_best = offspring[0][1].copy()
                if offspring[0][0] < best_f:
                    best_f = offspring[0][0]
                    best_x = offspring[0][1].copy()

                # Selection
                w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                w = w / np.sum(w)
                mueff = 1 / np.sum(w ** 2)

                # Mean update
                x_new = sum(w[i] * offspring[i][1] for i in range(mu))

                # Check for stagnation
                if np.linalg.norm(x_new - x) < 1e-14:
                    break

                # Update evolution paths
                z_mean = np.linalg.solve(B.T, (x_new - x) / sigma) if sigma > 1e-14 else np.zeros(d)
                pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * z_mean
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * z_mean

                # Update covariance matrix
                delta_hc = (1 - cc) if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** 2) < (1.4 + 2 / (d + 1)) * chiN else 0
                C = (1 + c1 * delta_hc * cc * (2 - cc) - c1) * C + c1 * np.outer(pc, pc)
                delta_hc = (1 - cc) if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** 2) < (1.4 + 2 / (d + 1)) * chiN else 0
                rank_mu = sum(w[i] ** 2 * np.outer(offspring[i][1] - x, offspring[i][1] - x) for i in range(mu))
                C = C + cmu * rank_mu

                # Symmetrize
                C = (C + C.T) / 2

                # Ensure positive definiteness
                eigvals, eigvecs = np.linalg.eigh(C)
                if np.min(eigvals) < 1e-14:
                    eigvals = np.maximum(eigvals, 1e-14)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                # Update step size
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chiN - 1))

                x = x_new

                # Convergence check
                if sigma < 1e-12:
                    break

                # Check if we're stuck
                if gen > 5:
                    # Simple stagnation check: if best hasn't improved in last few generations
                    pass

            return x_best, f_best

        # Multi-restart CMA-ES
        # First run: center, larger sigma
        # Subsequent runs: random points, smaller sigma

        rng = np.random.default_rng(123)

        # Allocate budget: first run gets 60%, rest split among restarts
        first_budget = int(total_budget * 0.6)
        remaining = total_budget - first_budget
        num_restarts = 3
        restart_budget = remaining // num_restarts if num_restarts > 0 else 0

        # First run from center
        x0 = np.zeros(d)
        sigma0 = 2.0
        cmaes_run(x0, sigma0, first_budget, rng)

        # Restart runs from random points
        for r in range(num_restarts):
            if calls_made >= total_budget or time.time() > deadline:
                break
            x0 = rng.uniform(-4, 4, d)
            sigma0 = 1.0
            cmaes_run(x0, sigma0, restart_budget, rng)

        # Final: use remaining budget for local search around best
        if best_x is not None and calls_made < total_budget:
            remaining_calls = total_budget - calls_made
            sigma_local = 0.5
            x = best_x.copy()
            fx = best_f
            for _ in range(remaining_calls):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma_local * rng.standard_normal(d), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma_local *= 1.2
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma_local *= 0.85

        return best_x, best_f
# EVOLVE-BLOCK-END
