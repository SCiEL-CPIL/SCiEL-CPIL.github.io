"""
Optimizer for BBOB f24 (Lunacek bi-Rastrigin) using Differential Evolution with restarts.
DE is more robust to the multimodal landscape of Rastrigin functions.
"""
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = np.inf

        # Helper to update best
        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # DE Parameters
        # F: Scaling factor for differential variation
        # CR: Crossover rate
        # N: Population size
        # For dim=5, N=20 is reasonable.
        F = 0.8
        CR = 0.9
        N = 20
        
        # Initial population
        pop = np.random.uniform(-5, 5, size=(N, dim))
        fit = np.empty(N)
        
        for i in range(N):
            if calls >= budget or time.time() > deadline:
                return best_x, best_f
            fit[i] = func(pop[i])
            calls += 1
            update_best(pop[i], fit[i])
            
        # Main DE Loop
        gen = 0
        stagnation_count = 0
        prev_best = best_f
        
        while calls < budget:
            if time.time() > deadline:
                break
                
            # Generate offspring
            for i in range(N):
                if calls >= budget or time.time() > deadline:
                    break
                    
                # Select distinct random indices
                r1, r2, r3 = np.random.choice(N, 3, replace=False)
                while r1 == i or r2 == i or r3 == i:
                    r1, r2, r3 = np.random.choice(N, 3, replace=False)
                    
                # DE/rand/1
                trial = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover (Binomial)
                j_rand = np.random.randint(dim)
                mask = np.random.rand(dim) < CR
                mask[j_rand] = True
                
                trial = np.where(mask, trial, pop[i])
                
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                trial_f = func(trial)
                calls += 1
                update_best(trial, trial_f)
                
                # Selection
                if trial_f <= fit[i]:
                    pop[i] = trial
                    fit[i] = trial_f
                    
            # Check stagnation
            if best_f < prev_best - 1e-8:
                stagnation_count = 0
            else:
                stagnation_count += 1
            prev_best = best_f
            
            # Restart strategy: If stagnant, re-initialize part of the population
            # or do a full restart if very stagnant
            if stagnation_count > 10:
                # Re-initialize worst half of the population
                idx = np.argsort(fit)[N//2:]
                for j in idx:
                    if calls >= budget or time.time() > deadline:
                        break
                    new_x = np.random.uniform(-5, 5, size=dim)
                    new_f = func(new_x)
                    calls += 1
                    pop[j] = new_x
                    fit[j] = new_f
                    update_best(new_x, new_f)
                stagnation_count = 0
            elif stagnation_count > 20:
                # Full restart around best point with larger spread
                for j in range(N):
                    if calls >= budget or time.time() > deadline:
                        break
                    new_x = best_x + np.random.normal(0, 2.0, size=dim)
                    new_x = np.clip(new_x, -5, 5)
                    new_f = func(new_x)
                    calls += 1
                    pop[j] = new_x
                    fit[j] = new_f
                    update_best(new_x, new_f)
                stagnation_count = 0
                
            gen += 1

        # Local Polish Phase using Nelder-Mead
        # Nelder-Mead is derivative-free and robust, good for polishing DE results
        if calls < budget and best_x is not None and time.time() < deadline:
            def objective(x):
                nonlocal calls
                if calls >= budget or time.time() > deadline:
                    return best_f
                x_clipped = np.clip(x, -5, 5)
                f = func(x_clipped)
                calls += 1
                if f < best_f:
                    best_f = f
                    best_x = x_clipped.copy()
                return f

            try:
                # Simple Nelder-Mead implementation or use scipy if available and safe
                # To avoid scipy dependency issues or overhead, let's use a simple local search or Nelder-Mead
                # However, scipy.optimize.minimize is available. Let's use Nelder-Mead for robustness.
                from scipy.optimize import minimize
                result = minimize(objective, best_x, method='Nelder-Mead', 
                                  options={'maxiter': 100, 'maxfev': max(1, budget - calls - 1), 'xatol': 1e-8, 'fatol': 1e-8})
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5, 5).copy()
            except Exception:
                pass

        return best_x, best_f
