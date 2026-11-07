import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)
        
        # Reserve budget for local search
        # Büche-Rastrigin is multimodal. DE needs enough budget to find the basin.
        # Local search (Nelder-Mead) is efficient for the final convergence.
        # Let's give DE ~70% of budget, Local Search ~30%.
        # But NM can be inefficient if starting point is bad.
        # Let's try DE for most of it, then a few local searches on top candidates.
        
        pop_size = 20
        de_budget = int(budget * 0.7)
        
        # Initialize
        population = rng.uniform(-5, 5, (pop_size, dim))
        fitnesses = np.array([func(x) for x in population])
        calls_made = pop_size
        
        best_idx = np.argmin(fitnesses)
        best_x = population[best_idx].copy()
        best_f = fitnesses[best_idx]
        
        F = 0.8
        CR = 0.9
        
        # DE Loop
        # We need to be careful with the number of calls.
        # Each generation uses pop_size calls (one per individual).
        # We can run full generations until we run out of budget.
        
        while calls_made < de_budget:
            # Generate trials for all individuals in the population
            # We might not have enough budget for a full generation at the end,
            # but we can just break if we hit the limit.
            
            for i in range(pop_size):
                if calls_made >= de_budget:
                    break
                
                # Select r1, r2, r3 distinct from i and each other
                # Use rng.choice or integers
                # For small pop_size, simple while loop is fine, but let's optimize slightly
                # Or just use rng.integers and check
                # Since pop_size=20, collisions are rare but possible.
                
                # A faster way to pick 3 distinct from i:
                # candidates = list(range(pop_size))
                # candidates.pop(i)
                # r1, r2, r3 = rng.choice(candidates, 3, replace=False)
                
                # But let's stick to the previous robust method or similar
                # Actually, rng.choice is cleaner.
                candidates = list(range(pop_size))
                candidates.pop(i)
                r1, r2, r3 = rng.choice(candidates, 3, replace=False)
                
                # Mutation
                v = population[r1] + F * (population[r2] - population[r3])
                
                # Crossover
                j_rand = rng.integers(0, dim)
                trial = np.empty(dim)
                # Vectorized crossover
                mask = rng.random(dim) < CR
                mask[j_rand] = True
                trial = np.where(mask, v, population[i])
                
                trial = np.clip(trial, -5, 5)
                f_trial = func(trial)
                calls_made += 1
                
                # Selection
                if f_trial < fitnesses[i]:
                    population[i] = trial
                    fitnesses[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        
        # Local Search Phase
        # We have remaining_calls = budget - calls_made
        remaining_calls = budget - calls_made
        
        if remaining_calls > 0:
            # Sort population by fitness to get top candidates
            # Nelder-Mead is local, so starting from the best DE point is good.
            # But maybe starting from a few top points helps if NM gets stuck?
            # Given the budget, let's try NM on the best point.
            # If we have enough budget, we could try on top 2-3.
            
            # Let's allocate remaining budget to NM.
            # NM can use maxfev.
            
            # It is often better to use a more robust local search or multiple runs.
            # However, NM is standard.
            
            # One issue: NM might consume all remaining_calls and not improve much if stuck.
            # We can split the budget.
            
            # Let's just use NM on the global best.
            res = minimize(
                func, 
                best_x, 
                method='Nelder-Mead',
                options={'maxfev': remaining_calls, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            if res.success or res.fun < best_f:
                best_f = res.fun
                best_x = res.x
            else:
                # If NM failed or didn't improve, keep DE best
                pass
                
        return best_x, best_f
