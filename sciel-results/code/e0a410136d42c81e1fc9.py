"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES with multi-start and Nelder-Mead refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin within 300s limit
        rng = np.random.default_rng(42)

        # Track best solution globally
        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        # Phase 1: CMA-ES with multiple restarts
        # We'll do several short CMA-ES runs from different starting points
        # to explore the 21-peak landscape

        def cma_es(x0, budget, sigma0, deadline):
            """Simple CMA-ES implementation."""
            dim = self.dim
            x = x0.copy()
            f_x = func(x)
            sigma = sigma0
            C = np.eye(dim)
            pc = np.zeros(dim)
            cs = np.zeros(dim)
            
            mu = max(2, dim // 2)
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mu_eff = 1.0 / (w ** 2).sum()
            cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            cs_ = (1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1)) / (dim + 3)
            c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
            damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs_
            
            B = np.linalg.inv(np.sqrt(C))
            D = np.linalg.inv(B).diagonal()
            D_sqrt = np.sqrt(D)
            
            calls = 1
            best_local_x = x.copy()
            best_local_f = f_x
            
            while calls < budget:
                if time.time() > deadline:
                    break
                
                # Generate offspring
                offspring = []
                for i in range(mu):
                    z = rng.standard_normal(dim)
                    y = x + sigma * (B @ z)
                    y = np.clip(y, -5, 5)
                    f_y = func(y)
                    calls += 1
                    offspring.append((f_y, y))
                
                offspring.sort(key=lambda t: t[0])
                
                # Update best
                if offspring[0][0] < best_local_f:
                    best_local_f = offspring[0][0]
                    best_local_x = offspring[0][1].copy()
                
                # Compute new mean
                x_new = sum(w[i] * offspring[i][1] for i in range(mu))
                
                # Update covariance matrix
                diff = (x_new - x) / sigma
                pc = (1 - cc) * pc + cc * diff
                hsig = (np.linalg.norm(pc) / np.sqrt(1 - (1 - cc) ** (2 * (calls - x0.shape[0])))) / np.sqrt(mu_eff) < (1.4 + 2 / (dim + 1)) * cs_
                C = (1 + c1 * hsig * cs_ * (2 - cs_) - c1 - cmu) * C
                C += c1 * (1 + hsig * cs_ * (2 - cs_) - 1) * np.outer(pc, pc)
                C += cmu * sum(w[i] * np.outer(offspring[i][1] - x, offspring[i][1] - x) / sigma ** 2 for i in range(mu))
                
                # Ensure positive definite
                try:
                    B = np.linalg.inv(np.sqrt(C))
                    D = np.linalg.inv(B).diagonal()
                except:
                    C = C + 0.01 * np.eye(dim)
                    B = np.linalg.inv(np.sqrt(C))
                    D = np.linalg.inv(B).diagonal()
                
                x = x_new
                f_x = func(x)
                calls += 1
                if f_x < best_local_f:
                    best_local_f = f_x
                    best_local_x = x.copy()
                
                # Adapt sigma
                sigma *= np.exp(cs_ / damps * (np.linalg.norm(x_new - x) / (sigma * np.sqrt(2 * mu_eff)) - 1))
                sigma = max(sigma, 1e-10)
                sigma = min(sigma, 10.0)
            
            return best_local_x, best_local_f, calls

        # Multiple restarts with CMA-ES
        n_restarts = 3
        budget_per_restart = self.budget // n_restarts
        
        for r in range(n_restarts):
            if time.time() > deadline:
                break
            
            if r == 0:
                x0 = np.zeros(self.dim)
                sigma0 = 3.0
            else:
                x0 = rng.uniform(-4, 4, self.dim)
                sigma0 = 2.0
            
            x_best, f_best, calls_used = cma_es(x0, budget_per_restart, sigma0, deadline)
            update_best(x_best, f_best)

        # Phase 2: Nelder-Mead refinement from best point
        if best_x is not None and time.time() < deadline:
            try:
                res = minimize(
                    lambda x: func(np.clip(x, -5, 5)),
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                if res.fun < best_f:
                    update_best(res.x, res.fun)
            except:
                pass

        # Phase 3: Additional local search around best point
        if best_x is not None and time.time() < deadline:
            # Coordinate descent
            x = best_x.copy()
            f = best_f
            step = 0.5
            for _ in range(50):
                if time.time() > deadline:
                    break
                improved = False
                for d in range(self.dim):
                    for delta in [step, -step]:
                        x_trial = x.copy()
                        x_trial[d] += delta
                        x_trial = np.clip(x_trial, -5, 5)
                        f_trial = func(x_trial)
                        if f_trial < f:
                            x, f = x_trial, f_trial
                            improved = True
                    if improved:
                        break
                if not improved:
                    step *= 0.5
                    if step < 1e-4:
                        break
            update_best(x, f)

        # Phase 4: Differential Evolution for exploration
        if time.time() < deadline:
            pop_size = 10
            population = [best_x.copy()]
            for _ in range(pop_size - 1):
                pop = best_x + rng.standard_normal(self.dim) * 1.5
                population.append(np.clip(pop, -5, 5))
            
            for gen in range(20):
                if time.time() > deadline:
                    break
                for i in range(pop_size):
                    a = rng.choice(pop_size)
                    b = rng.choice(pop_size)
                    c = rng.choice(pop_size)
                    while len({a, b, c, i}) < 4:
                        c = rng.choice(pop_size)
                    
                    v = population[a] + 0.8 * (population[b] - population[c])
                    v = np.clip(v, -5, 5)
                    f_v = func(v)
                    
                    f_i = func(population[i])
                    if f_v < f_i:
                        population[i] = v
                        if f_v < best_f:
                            update_best(population[i], f_v)

        return best_x, best_f
# EVOLVE-BLOCK-END
