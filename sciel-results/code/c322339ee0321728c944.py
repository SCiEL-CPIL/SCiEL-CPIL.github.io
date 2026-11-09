import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es_single(self, func, x0, sigma0, budget, dim, rng, calls_ref, best_ref, deadline):
        """Run a single CMA-ES instance, updating shared state."""
        # Population size
        lam = max(4, 4 + int(3 * np.log(dim)))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (2 + 2.0 / (dim + 1)) / (dim + 4 + 2.0 / (dim + 1))
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        x_mean = x0.copy()
        sigma = sigma0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        calls = calls_ref[0]
        best_f = best_ref[0]
        best_x = best_ref[1].copy() if best_ref[1] is not None else x_mean.copy()

        def evaluate(x):
            nonlocal calls, best_f, best_x
            calls += 1
            calls_ref[0] = calls
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
                best_ref[0] = best_f
                best_ref[1] = best_x
            return f_val

        # Initial evaluation
        f_mean = evaluate(x_mean)
        
        D = np.linalg.cholesky(C)
        
        while calls < budget:
            if time.time() > deadline:
                break

            current_lam = lam
            if calls + current_lam > budget:
                current_lam = budget - calls
            if current_lam == 0:
                break

            z = rng.standard_normal((current_lam, dim))
            y = np.einsum('ij,kj->ki', D, z)
            x_offspring = x_mean + sigma * y
            x_offspring = np.clip(x_offspring, -5, 5)

            f_offspring = np.array([evaluate(xi) for xi in x_offspring])

            idx = np.argsort(f_offspring)
            x_offspring = x_offspring[idx]
            f_offspring = f_offspring[idx]

            x_mean_old = x_mean.copy()
            x_mean = np.sum(x_offspring[:mu] * weights[:, np.newaxis], axis=0)

            y_mean_weighted = np.sum(z[idx[:mu]] * weights[:, np.newaxis], axis=0)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean_weighted
            sigma = sigma * np.exp(c_sigma / chi_n * (np.linalg.norm(ps) - chi_n))
            sigma = np.clip(sigma, 1e-10, 1e10)

            pc = (1 - c_c) * pc + c_c / sigma * (x_mean - x_mean_old)
            
            y_selected = np.array([np.linalg.solve(D, (xi - x_mean_old) / sigma) for xi in x_offspring[:mu]])
            
            rank_one = np.outer(pc, pc)
            rank_mu = np.sum(weights[:, np.newaxis, np.newaxis] * y_selected[:, np.newaxis, :] * y_selected[:, np.newaxis, :].T, axis=0)
            
            C = (1 - c_1 - c_mu) * C + c_1 * (rank_one + (1 - (1 - c_c)**2) * C) + c_mu * rank_mu
            C = (C + C.T) / 2.0
            
            try:
                D = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-10 * np.eye(dim)
                try:
                    D = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    D = np.eye(dim)
                    pc = np.zeros(dim)
                    ps = np.zeros(dim)
                    sigma = 1.0

        return best_x, best_f

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0

        best_x = None
        best_f = np.inf
        calls = 0
        rng = np.random.default_rng()
        
        # Shared state for CMA-ES
        calls_ref = [0]
        best_ref = [np.inf, None]

        # IPOP-CMA-ES: restart with larger sigma and different initial point
        # For Weierstrass (multimodal), restarts are crucial
        num_restarts = 5
        calls_per_restart = max(1, budget // num_restarts)
        
        for r in range(num_restarts):
            if time.time() > deadline:
                break
            if calls >= budget:
                break

            # Different initial points for diversity
            if r == 0:
                x0 = np.zeros(dim)
                sigma0 = 2.0
            else:
                x0 = rng.uniform(-4, 4, dim)
                # Increase sigma for later restarts to explore more broadly
                sigma0 = 2.0 * (1.0 + 0.5 * r)

            # Budget for this restart
            budget_this = min(calls_per_restart, budget - calls)
            if budget_this < 4:
                break

            _, _ = self._cma_es_single(func, x0, sigma0, budget_this, dim, rng, calls_ref, best_ref, deadline)
            calls = calls_ref[0]
            best_f = best_ref[0]
            best_x = best_ref[1]

            # If we've made good progress, use remaining budget for a final polish
            if calls < budget and (best_f < 10.0 or r >= num_restarts - 1):
                # Final CMA-ES from best point with small sigma
                if best_x is not None and time.time() < deadline:
                    remaining = budget - calls
                    if remaining >= 4:
                        _, _ = self._cma_es_single(func, best_x, 0.5, remaining, dim, rng, calls_ref, best_ref, deadline)
                        calls = calls_ref[0]
                        best_f = best_ref[0]
                        best_x = best_ref[1]

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
