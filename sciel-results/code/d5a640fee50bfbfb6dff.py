import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

        # Weierstrass f16 is multimodal - use multiple restarts
        n_restarts = 3
        budget_per_restart = budget // n_restarts

        best_f_global = np.inf
        best_x_global = np.zeros(dim)

        for restart in range(n_restarts):
            if time.time() > deadline:
                break

            # Different starting points for each restart
            if restart == 0:
                x0 = np.zeros(dim)
            elif restart == 1:
                x0 = np.random.uniform(-3, 3, dim)
            else:
                x0 = np.random.uniform(-4, 4, dim)

            # CMA-ES Parameters
            pop_size = 4 + int(3 * np.log(dim))
            pop_size = max(2, pop_size)

            sigma = 2.0

            # CMA-ES State
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            cs = (pop_size + 2) / (dim + 4)
            ds = 1 + 2 * max(0, np.sqrt((np.sqrt(dim) - 1) / (np.sqrt(dim) + 1)) - 1) / cs
            cmu = 2 / ((dim + 1.3)**2 + pop_size)
            chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(20*dim**2))
            mu = pop_size // 2
            mu_eff = 1.0 / np.sum((np.arange(1, mu + 1))**2)
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)

            # Best solution tracking
            best_x = x0.copy()
            best_f = func(best_x)
            calls = 1

            while calls < budget_per_restart:
                if time.time() > deadline:
                    break

                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)

                z = np.random.randn(pop_size, dim)
                y = z @ L.T
                pop = x0 + sigma * y

                fvals = []
                for p in pop:
                    if calls >= budget_per_restart:
                        break
                    if time.time() > deadline:
                        break
                    p_clipped = np.clip(p, -5, 5)
                    f_val = func(p_clipped)
                    fvals.append(f_val)
                    calls += 1
                    if f_val < best_f:
                        best_f = f_val
                        best_x = p_clipped.copy()

                if len(fvals) < pop_size:
                    break

                idx = np.argsort(fvals)
                pop_sorted = pop[idx]

                x_new = np.sum(pop_sorted[:mu] * weights[:, np.newaxis], axis=0)

                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (L.T @ (x_new - x0) / sigma)
                pc = (1 - cs) * pc + np.sqrt(cs * (2 - cs) * mu_eff) * (x_new - x0) / sigma

                C = (1 - cs) * (1 - cmu) * C + cs * np.outer(pc, pc)
                diff = (pop_sorted[:mu] - x0) / sigma
                C += cmu * np.sum(weights[:, np.newaxis] * np.einsum('i,ij,ik->jk', diff, diff), axis=0)

                sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi_n - 1))
                x0 = x_new
                x0 = np.clip(x0, -5, 5)

            if best_f < best_f_global:
                best_f_global = best_f
                best_x_global = best_x.copy()

        return best_x_global, best_f_global
