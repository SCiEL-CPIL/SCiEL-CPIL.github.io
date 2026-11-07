import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        
        N = 10
        F = 0.8
        CR = 0.9
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(0)
        
        # Track calls manually to ensure we don't exceed budget
        calls_used = 0
        
        def safe_func(x):
            nonlocal calls_used
            if calls_used >= budget or time.time() > deadline:
                return np.inf
            fx = func(x)
            calls_used += 1
            return fx

        # Initialize DE
        pop = rng.uniform(-5, 5, (N, dim))
        fit = np.array([safe_func(x) for x in pop])
        best_idx = np.argmin(fit)
        best_x = pop[best_idx].copy()
        best_f = fit[best_idx]
        
        stagnation_counter = 0
        stagnation_limit = 10
        
        while calls_used < budget:
            if time.time() > deadline:
                break
            
            prev_best_f = best_f
            
            for i in range(N):
                if calls_used >= budget or time.time() > deadline:
                    break
                
                # Pick 3 distinct indices different from i
                candidates = [j for j in range(N) if j != i]
                r1, r2, r3 = rng.choice(candidates, 3, replace=False)
                
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                j_rand = rng.integers(0, dim)
                u = np.empty(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        u[j] = v[j]
                    else:
                        u[j] = pop[i][j]
                
                u = np.clip(u, -5, 5)
                
                fu = safe_func(u)
                
                if fu < fit[i]:
                    pop[i] = u
                    fit[i] = fu
                    if fu < best_f:
                        best_f = fu
                        best_x = u.copy()
            
            if best_f < prev_best_f:
                stagnation_counter = 0
            else:
                stagnation_counter += 1
                
            if stagnation_counter > stagnation_limit:
                # Restart non-best individuals
                for i in range(N):
                    if i == best_idx:
                        continue
                    pop[i] = rng.uniform(-5, 5, dim)
                    fit[i] = safe_func(pop[i])
                    if fit[i] < best_f:
                        best_f = fit[i]
                        best_x = pop[i].copy()
                        best_idx = i
                stagnation_counter = 0
                best_idx = np.argmin(fit)

        # Final Polish with Nelder-Mead if budget allows
        if calls_used < budget - 50 and time.time() < deadline - 5:
            try:
                res = minimize(func, best_x, method='Nelder-Mead', options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_f
