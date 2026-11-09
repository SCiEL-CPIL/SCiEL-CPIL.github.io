"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Portfolio of CMA-ES and Nelder-Mead runs with time and budget management."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill
        dim = self.dim
        total_budget = self.budget
        calls_used = 0

        best_x = None
        best_f = np.inf

        def track_call(x, f):
            nonlocal calls_used, best_x, best_f
            calls_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def time_left():
            return deadline - time.time()

        def cma_es_run(rng, x0, sigma0, budget_left):
            """Simple CMA-ES implementation."""
            nonlocal calls_used, best_x, best_f, deadline
            n = dim
            lam = max(4, int(4 + 3 * np.log(n)))  # population size
            mu = lam // 2  # number of selected individuals
            
            # CMA-ES parameters
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
            
            # State
            x = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            sigma_history = []
            
            def sample_population():
                nonlocal x, sigma, C
                B = np.linalg.cholesky(C + 1e-10 * np.eye(n))
                z = rng.standard_normal((lam, n))
                y = z @ B.T
                return x + sigma * y, z, y
            
            for gen in range(budget_left):
                if time.time() > deadline:
                    return
                if calls_used >= total_budget:
                    return
                
                # Sample population
                pop, z, y = sample_population()
                pop = np.clip(pop, -5, 5)
                
                # Evaluate population
                fvals = np.array([func(p) for p in pop])
                for p, fv in zip(pop, fvals):
                    track_call(p, fv)
                
                # Sort by fitness
                order = np.argsort(fvals)
                y_sorted = y[order]
                fvals_sorted = fvals[order]
                
                # Weights
                w = np.zeros(mu)
                w[:] = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                w /= w.sum()
                mu_eff = 1.0 / np.sum(w ** 2)
                
                # Update mean
                x_new = x + sigma * np.sum(w[:, None] * y_sorted[:mu], axis=0)
                
                # Update covariance
                yw = y_sorted[:mu]
                delta_x = x_new - x
                pc = (1 - c_c) * pc + c_c * delta_x / sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * delta_x / sigma
                hsig = 1 - (ps @ ps) / (n * sigma ** 2) < (1 + 2 / (n + 1)) * c_sigma
                hs = (ps @ ps) / (n * (1 - (1 - c_sigma) ** (2 * (gen + 1)))) < (2 + 4 / (n + 1)) * c_c
                rank_pc = np.outer(pc, pc) if hsig else np.zeros((n, n))
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    rank_mu += w[i] * np.outer(yw[i], yw[i])
                C = (1 + c_1 * (2 - c_1) - c_1) * C + c_1 * (rank_pc + rank_mu)
                C = 0.5 * (C + C.T)  # symmetrize
                
                # Update sigma
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)
                
                x = x_new
                sigma_history.append(sigma)
                
                # Check for convergence
                if np.linalg.norm(delta_x) < 1e-8 * (1 + np.linalg.norm(x)):
                    break
                
                # Update best
                best_idx = np.argmin(fvals)
                if fvals[best_idx] < best_f:
                    best_f = fvals[best_idx]
                    best_x = pop[best_idx].copy()
                
                # Budget check
                if calls_used >= total_budget:
                    return
                
                # Early stopping if sigma is very small
                if sigma < 1e-7:
                    break
        
        def nm_polish(x0, budget_left):
            """Nelder-Mead polish."""
            nonlocal calls_used, best_x, best_f, deadline
            if time.time() > deadline or calls_used >= total_budget:
                return
            
            def wrapper(x):
                nonlocal calls_used
                if calls_used >= total_budget:
                    return best_f + 1e10
                calls_used += 1
                return func(np.clip(x, -5, 5))
            
            try:
                res = minimize(wrapper, x0, method='Nelder-Mead',
                              options={'maxiter': min(budget_left, 50),
                                       'xatol': 1e-8, 'fatol': 1e-10,
                                       'adaptive': True})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5).copy()
            except Exception:
                pass

        # Strategy:
        # 1. Brief random exploration to find a good starting point
        # 2. CMA-ES run from the best point with moderate sigma
        # 3. Nelder-Mead polish on the final best point

        rng = np.random.default_rng(12345)

        # Phase 1: Brief random exploration (10% of budget)
        explore_budget = max(5, total_budget // 10)
        for _ in range(explore_budget):
            if time.time() > deadline:
                break
            if calls_used >= total_budget:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            track_call(x, fx)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            track_call(best_x, best_f)

        remaining_budget = total_budget - calls_used

        # Phase 2: CMA-ES from the best point
        if remaining_budget > 20:
            cma_budget = remaining_budget * 3 // 4
            cma_es_run(rng, best_x, 1.5, cma_budget)
            remaining_budget = total_budget - calls_used

        # Phase 3: Another CMA-ES from a slightly perturbed best point
        if remaining_budget > 15:
            x2 = best_x + rng.standard_normal(dim) * 0.5
            x2 = np.clip(x2, -5, 5)
            cma_es_run(rng, x2, 0.8, remaining_budget * 2 // 3)
            remaining_budget = total_budget - calls_used

        # Phase 4: Nelder-Mead polish
        if remaining_budget > 5:
            nm_polish(best_x, remaining_budget)

        # Final safety: ensure we have a valid best_x
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            track_call(best_x, best_f)

        return best_x, best_f
# EVOLVE-BLOCK-END
