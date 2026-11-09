"""
Multi-Start CMA-ES optimizer for BBOB f18 (Schaffers F7 ill-conditioned).

Strategy:
  - Use full-covariance CMA-ES to model the anisotropic search space.
  - Multi-start: run multiple independent CMA-ES instances, each from a different
    random starting point, to explore different regions of the search space.
  - This addresses the stagnation issue by allowing the optimizer to escape local
    minima or flat regions that a single run might get stuck in.
  - Each run uses a fraction of the budget; the global best is tracked across runs.
  - A small population (lambda=10) keeps evaluations per generation low.
  - Strict call counting and a time guard ensure validity and timely completion.
"""

import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es_run(self, func, start_x, budget, dim, deadline, n_evals, best_f, best_x):
        """Run a single CMA-ES instance from start_x, updating global best."""
        lambda_ = 10
        mu = lambda_ // 2

        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))

        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

        mean = start_x.copy()
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        no_improve = 0
        max_stagnation = 12
        local_best_f = best_f
        local_best_x = best_x.copy()

        while n_evals[0] < budget and time.time() < deadline:
            # Generate population
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                try:
                    w, v = np.linalg.eigh(C)
                    w = np.maximum(w, 1e-10)
                    C = v @ np.diag(w) @ v.T
                    C = (C + C.T) / 2.0
                    B = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    B = np.eye(dim)

            pop = np.empty((lambda_, dim))
            fit = np.empty(lambda_)
            valid_count = 0

            for i in range(lambda_):
                if n_evals[0] >= budget or time.time() > deadline:
                    break
                z = np.random.randn(dim)
                x = mean + sigma * (B @ z)
                x = np.clip(x, -5, 5)
                fx = func(x)
                n_evals[0] += 1
                pop[i] = x
                fit[i] = fx
                valid_count += 1

            if valid_count < mu:
                break

            order = np.argsort(fit[:valid_count])
            pop_sorted = pop[order, :]
            fit_sorted = fit[order]

            if fit_sorted[0] < local_best_f:
                local_best_f = fit_sorted[0]
                local_best_x = pop_sorted[0].copy()
                no_improve = 0
            else:
                no_improve += 1

            if local_best_f < best_f:
                best_f = local_best_f
                best_x = local_best_x.copy()

            mean_new = np.sum(weights[:, None] * pop_sorted[:mu], axis=0)

            diff = (pop_sorted[:mu] - mean) / sigma
            C_new = ((1 - c_1 - c_mu) * C
                     + c_1 * ((1 + (1 - c_c) * c_c) * np.outer(pc, pc) - C)
                     + c_mu * np.sum(weights[:, None, None] * np.transpose(diff) * diff, axis=0))
            C_new = (C_new + C_new.T) / 2.0

            pc = (1 - c_c) * pc + c_c * np.sqrt(mu_eff) * (mean_new - mean) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (mean_new - mean) / sigma

            sigma *= np.exp(c_sigma / chiN * (np.linalg.norm(ps) / chiN - 1))
            sigma = np.clip(sigma, 1e-10, 50.0)

            mean = mean_new
            C = C_new

            if no_improve >= max_stagnation:
                break

            if sigma < 1e-8:
                break

        return best_f, best_x

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # 45s safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        n_evals = [0]  # Use list for mutability in nested function

        # Global best tracking
        best_f = np.inf
        best_x = np.zeros(dim)

        # Multi-start: divide budget among multiple independent runs
        # First run gets more budget for initial exploration
        n_starts = 4
        # Budget allocation: first run gets 50%, remaining split equally
        budget_alloc = [int(budget * 0.5)] + [int(budget * 0.5 / (n_starts - 1))] * (n_starts - 1)
        # Ensure total doesn't exceed budget
        total_alloc = sum(budget_alloc)
        if total_alloc > budget:
            budget_alloc[-1] -= (total_alloc - budget)

        for run_idx in range(n_starts):
            if n_evals[0] >= budget or time.time() > deadline:
                break

            # Initialize start point
            if run_idx == 0:
                # First run: random start in the box
                start_x = np.random.uniform(-5, 5, dim)
            else:
                # Subsequent runs: start near current best with some randomization
                if np.isfinite(best_f):
                    start_x = best_x + np.random.uniform(-1, 1, dim)
                    start_x = np.clip(start_x, -5, 5)
                else:
                    start_x = np.random.uniform(-5, 5, dim)

            run_budget = min(budget_alloc[run_idx], budget - n_evals[0])
            if run_budget < 20:  # Need at least one generation
                break

            best_f, best_x = self._cma_es_run(
                func, start_x, budget, dim, deadline, n_evals, best_f, best_x
            )

        return best_x, best_f
