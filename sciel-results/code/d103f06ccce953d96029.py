"""Two-phase optimizer: CMA-ES exploration + L-BFGS-B gradient refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES for global exploration, then L-BFGS-B for local gradient-based refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget

        # Phase 1: CMA-ES (simplified, budget-limited)
        # Allocate 70% of budget to CMA-ES
        cma_budget = int(budget * 0.7)
        cma_calls = 0

        # CMA-ES parameters
        pop_size = max(4, int(4 + 3 * np.log(d)))  # ~11 for d=5
        sigma = 2.0
        x_mean = np.zeros(d)
        
        # Initialize population
        pop = [x_mean + sigma * rng.standard_normal(d) for _ in range(pop_size)]
        pop = [np.clip(x, -5, 5) for x in pop]
        fvals = [func(x) for x in pop]
        cma_calls = pop_size

        best_x = pop[np.argmin(fvals)].copy()
        best_fx = min(fvals)

        # CMA-ES main loop
        lambda_ = pop_size
        mu = lambda_ // 2
        # Weights (proportional)
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2)**2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
        chi_d = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d**2))

        # State variables
        pc = np.zeros(d)
        ps = np.zeros(d)
        C = np.eye(d)
        # Use Cholesky for sampling: C = L L^T
        try:
            L = np.linalg.cholesky(C)
        except:
            L = np.eye(d)

        for gen in range(1, 1000):
            if time.time() > deadline:
                break
            if cma_calls >= cma_budget:
                break

            # Sort population by fitness
            order = np.argsort(fvals)
            sorted_pop = [pop[i] for i in order]
            sorted_fvals = [fvals[i] for i in order]

            # Update mean
            x_mean_new = np.sum([weights[i] * sorted_pop[i] for i in range(mu)], axis=0)

            # Update step size control
            old_L_inv = np.linalg.inv(L)
            y = (x_mean_new - x_mean) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * old_L_inv @ y
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_d - 1))
            sigma = max(sigma, 1e-10)

            # Update covariance
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * gen)) < (1.4 + 2 / (d + 1)) * chi_d:
                rank_one = np.outer(ps, ps)
            else:
                rank_one = None

            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(y, y) + (rank_one is None and 0 or (np.eye(d) - C) if rank_one is None else 0))
            # Simplified: just update with rank-one
            if rank_one is not None:
                C = (1 - c_1 - c_mu) * C + c_1 * rank_one
            # Rank-mu update
            for i in range(mu):
                zi = old_L_inv @ (sorted_pop[i] - x_mean) / sigma
                C += c_mu * np.outer(zi, zi)
            # Ensure positive definite
            C = (C + C.T) / 2
            # Add small regularization
            C += 1e-10 * np.eye(d)

            x_mean = x_mean_new

            # Sample new population
            try:
                L = np.linalg.cholesky(C)
            except:
                L = np.eye(d)

            pop = []
            for i in range(lambda_):
                z = rng.standard_normal(d)
                x = x_mean + sigma * (L @ z)
                x = np.clip(x, -5, 5)
                pop.append(x)

            fvals = [func(x) for x in pop]
            cma_calls += lambda_

            # Track best
            cur_best_idx = np.argmin(fvals)
            if fvals[cur_best_idx] < best_fx:
                best_fx = fvals[cur_best_idx]
                best_x = pop[cur_best_idx].copy()

            # Convergence check
            if np.max(fvals) - np.min(fvals) < 1e-10:
                break

        # Phase 2: L-BFGS-B local refinement
        lbfgs_budget = budget - cma_calls
        if lbfgs_budget > 10 and time.time() < deadline:
            try:
                res = minimize(
                    func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * d,
                    options={'maxiter': lbfgs_budget, 'maxfun': lbfgs_budget}
                )
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
