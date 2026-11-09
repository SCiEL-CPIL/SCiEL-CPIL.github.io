"""Multi-start CMA-ES with Sobol initialization for BBOB f21."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """Multi-start CMA-ES with Sobol initialization for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        calls_made = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Sobol Initialization
        n_init = 20
        sampler = qmc.Sobol(d=dim, scramble=True)
        sobol_points = sampler.random(n=n_init)
        init_points = sobol_points * 10.0 - 5.0
        
        init_results = []
        for p in init_points:
            if calls_made >= budget or time.time() > deadline:
                break
            f = evaluate(p)
            init_results.append((p, f))
            
        if not init_results:
            return best_x, best_f

        init_results.sort(key=lambda x: x[1])
        
        # Phase 2: Multi-Start CMA-ES
        remaining = budget - calls_made
        if remaining <= 0:
            return best_x, best_f

        candidates = init_results[:3]
        allocs = [0.5, 0.3, 0.2]
        
        for i, (x0, f0) in enumerate(candidates):
            if time.time() > deadline or calls_made >= budget:
                break
            
            if i > 0 and f0 > best_f + 10.0:
                continue

            local_budget = int(remaining * allocs[i]) if i < len(allocs) else 10
            local_budget = min(local_budget, budget - calls_made)
            
            if local_budget < 10:
                continue

            sigma0 = 1.0
            if i > 0:
                dist = np.linalg.norm(x0 - init_results[0][0])
                sigma0 = max(0.5, dist * 0.5)
            
            self._run_cma_es(evaluate, x0, sigma0, local_budget, deadline, calls_made, budget)
            # Note: evaluate updates calls_made and best_f internally

        # Phase 3: Final Polish
        if time.time() < deadline and calls_made < budget and best_x is not None:
            remaining = budget - calls_made
            if remaining > 10:
                self._run_cma_es(evaluate, best_x, 0.1, remaining, deadline, calls_made, budget)

        return best_x, best_f

    def _run_cma_es(self, func, x0, sigma0, budget, deadline, calls_made, total_budget):
        dim = self.dim
        rng = np.random.default_rng(123) # Fixed seed for CMA-ES runs for reproducibility? Or use self.rng?
        
        # CMA-ES parameters
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        c1 = (2 + 2.0 / (dim + 1.3)) / ((dim + 1)**2 + 2)
        cm = (2 + 2.0 / (dim + 1.3)) / ((dim + 1.7)**2 + 2)
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c1
        
        x = np.array(x0, dtype=float)
        sigma = sigma0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        D = np.ones(dim)
        B = np.eye(dim)
        eigen_counter = 0
        eigen_interval = max(1, int(1.0 / (lam * c1) * dim))
        
        local_calls = 0
        while local_calls < budget:
            if time.time() > deadline:
                break
            
            pop = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = x + sigma * B.dot(D * z)
                pop.append(y)
            
            pop_f = []
            for y in pop:
                if time.time() > deadline:
                    break
                y_clipped = np.clip(y, -5.0, 5.0)
                f = func(y_clipped)
                local_calls += 1
                pop_f.append(f)
            
            if len(pop_f) < lam:
                break
                
            idx = np.argsort(pop_f)
            pop = [pop[i] for i in idx]
            pop_f = [pop_f[i] for i in idx]
            
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += weights[i] * pop[i]
                
            ps = (1 - cm) * ps + np.sqrt(cm * (2 - cm) * mueff) / D.dot(np.sqrt(C)).dot(x_new - x)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cm)**np.ceil(1 / cm))) < (1.4 + 2 / (dim + 1)) * np.sqrt((mueff - 1) / (dim + 1) + 1)
            
            pc = (1 - c1) * pc + hsig * np.sqrt(c1 * (2 - c1) * mueff) * (x_new - x)
            
            delta_hsig = 1 - hsig
            C = (1 + c1 * delta_hsig * mueff - c1) * C
            for i in range(mu):
                v = pop[i] - x
                C += c1 * weights[i] * np.outer(v, v)
            
            x = x_new
            sigma *= np.exp(c1 / damps * (np.linalg.norm(ps) / np.sqrt(dim) - 1))
            
            eigen_counter += 1
            if eigen_counter >= eigen_interval:
                eigen_counter = 0
                B, D = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D, 1e-12))
                
            if sigma < 1e-6:
                break
# EVOLVE-BLOCK-END
