"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES based black-box continuous optimizer with multi-start."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        n = self.dim
        total_budget = self.budget
        
        best_x_global = None
        best_f_global = np.inf
        calls_made = 0

        def run_cma_es(start_x, start_sigma, max_calls):
            nonlocal calls_made
            lam = max(4, int(4 + 3 * np.log(n)))
            mu = lam // 2
            
            weights = np.zeros(mu)
            for i in range(mu):
                weights[i] = np.log(mu + 0.5) - np.log(i + 1)
            weights /= weights.sum()
            mueff = 1 / np.sum(weights**2)
            
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            cs = (mu + 2) / (n + mu + 5)
            ds = 1 + 2 * max(0, np.sqrt((mu + 1) / (n + 1) - 1)) + cs
            cc = (4 + n/mu) / (n + 4 + 2*mu/n)
            c1 = (2 + 2*min(mu/n - 1, 1/(n+1))) / ((n+1.3)**2 + mu)
            cmu = min(1 - c1, 2 * (mu - 2 + 1/mu) / ((n+2)**2 + mu))

            x = start_x.copy()
            sigma = start_sigma
            C = np.eye(n)
            ps = np.zeros(n)
            pc = np.zeros(n)
            
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            local_calls = 1
            best_x_local = x_clipped.copy()
            best_f_local = f_val
            
            if local_calls >= max_calls:
                calls_made += local_calls
                return best_x_local, best_f_local

            while local_calls < max_calls:
                evals, evec = np.linalg.eigh(C)
                evals = np.maximum(evals, 1e-20)
                D = np.sqrt(evals)
                B = evec
                
                N = rng.standard_normal((lam, n))
                Y = (B @ (D * N.T)).T
                pop = x + sigma * Y
                pop = np.clip(pop, -5, 5)
                
                fitnesses = np.zeros(lam)
                for i in range(lam):
                    fitnesses[i] = func(pop[i])
                    local_calls += 1
                    if fitnesses[i] < best_f_local:
                        best_f_local = fitnesses[i]
                        best_x_local = pop[i].copy()
                    
                    if local_calls >= max_calls:
                        break
                
                if local_calls >= max_calls:
                    break

                order = np.argsort(fitnesses)
                pop_sorted = pop[order]
                
                m_old = x.copy()
                x_new = np.sum(pop_sorted[:mu] * weights[:, None], axis=0)
                x = x_new
                
                z = (x - m_old) / (sigma * chi_n)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * z
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2)) < (1.4 + 2/(n+1)) * chi_n
                sigma *= np.exp(cs/ds * (np.linalg.norm(ps)/chi_n - 1))
                
                y = (x - m_old) / sigma
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * y
                C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + cc*(1-cc)*C) + cmu * np.sum(
                    [np.outer(pop_sorted[i] - m_old, pop_sorted[i] - m_old) / (sigma**2) * weights[i] for i in range(mu)], axis=0
                )
                
                C = 0.5 * (C + C.T)
                x = np.clip(x, -5, 5)

            calls_made += local_calls
            return best_x_local, best_f_local

        n_starts = 3
        budget_per_start = total_budget // n_starts
        remaining = total_budget - budget_per_start * n_starts

        for i in range(n_starts):
            start_x = rng.uniform(-4, 4, n)
            start_sigma = 1.5
            
            current_budget = budget_per_start + (remaining if i == n_starts - 1 else 0)
            
            if calls_made + current_budget > total_budget:
                current_budget = total_budget - calls_made
            
            if current_budget <= 0:
                break

            x_res, f_res = run_cma_es(start_x, start_sigma, current_budget)
            
            if f_res < best_f_global:
                best_f_global = f_res
                best_x_global = x_res

        if best_x_global is None:
            best_x_global = np.zeros(n)
            best_f_global = func(best_x_global)
            calls_made += 1

        return best_x_global, best_f_global
# EVOLVE-BLOCK-END
