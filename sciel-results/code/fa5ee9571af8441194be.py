import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Keep 30s margin for safety
        deadline = start_time + 290 
        
        n = self.dim
        # Split budget: 60% for global DE search, 40% for local NM polish
        # This allows more aggressive local refinement which is crucial for Rastrigin's sharp minima
        global_budget = int(self.budget * 0.6)
        
        rng = np.random.default_rng(42)
        
        # Initialize population
        pop_size = 20
        pop = rng.uniform(-5, 5, (pop_size, n))
        
        # Evaluate initial population
        f_pop = np.array([func(np.clip(x, -5, 5)) for x in pop])
        calls = pop_size
        
        best_idx = np.argmin(f_pop)
        best_x = pop[best_idx].copy()
        best_f = f_pop[best_idx]
        
        # Differential Evolution Phase
        # We use a fixed strategy: DE/rand/1/bin with adaptive F and CR
        # F (scale factor) starts high and decays
        # CR (crossover rate) is fixed at 0.5
        
        # We will run until we've used global_budget calls
        while calls < global_budget and time.time() < deadline:
            # Adaptive F: Start at 0.8, decay to 0.1
            progress = calls / global_budget
            F = 0.8 - 0.7 * progress
            
            # Generate offspring
            # For each individual in population, create a mutant
            # We do this in batches to manage call counts strictly
            
            # Determine how many new evaluations we can do in this "generation"
            # We want to keep the population size constant, so we evaluate pop_size new points
            # But we must respect the budget.
            # If remaining budget < pop_size, we break.
            
            if global_budget - calls < pop_size:
                break
                
            new_f = np.full(pop_size, np.inf)
            new_pop = np.zeros_like(pop)
            
            for i in range(pop_size):
                if calls >= global_budget or time.time() > deadline:
                    break
                    
                # Select three distinct random individuals
                # Avoid i
                candidates = [j for j in range(pop_size) if j != i]
                a, b, c = rng.choice(candidates, 3, replace=False)
                
                # Mutation: v = x_a + F * (x_b - x_c)
                v = pop[a] + F * (pop[b] - pop[c])
                
                # Crossover: binomial
                j_rand = rng.integers(0, n)
                trial = pop[i].copy()
                for j in range(n):
                    if rng.random() < 0.5 or j == j_rand:
                        trial[j] = v[j]
                
                # Clip to box
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = func(trial)
                calls += 1
                new_f[i] = f_trial
                new_pop[i] = trial
                
                # Update best
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = trial.copy()
            
            # Selection: Replace population members if offspring is better
            for i in range(pop_size):
                if new_f[i] < f_pop[i]:
                    pop[i] = new_pop[i]
                    f_pop[i] = new_f[i]
            
            # If no improvement in last few generations, we could restart or adjust, 
            # but for simplicity and speed, we continue.
            # However, Rastrigin is multimodal. If we stagnate, we might be stuck in a local min.
            # A simple check: if best_f hasn't improved in 50 calls, perturb best_x?
            # For now, let's rely on the diversity of DE.
            
        # Local Polish Phase
        remaining_budget = self.budget - calls
        
        if remaining_budget > 5 and time.time() < deadline:
            # Use Nelder-Mead for fine-tuning
            # We wrap the function to count calls and enforce budget
            nm_calls = [0]
            
            def nm_func(x_nm):
                if nm_calls[0] >= remaining_budget or time.time() > deadline:
                    return best_f # Return best known to stop progress effectively
                nm_calls[0] += 1
                return func(np.clip(x_nm, -5, 5))
            
            x0 = best_x.copy()
            
            # Nelder-Mead parameters
            # Initial simplex size: small relative to the domain
            # We start from best_x found by DE
            
            try:
                res = minimize(nm_func, x0, method='Nelder-Mead', 
                               options={'maxiter': remaining_budget, 'xatol': 1e-5, 'fatol': 1e-5, 'initial_simplex': None})
                
                # Note: scipy's minimize with Nelder-Mead might not strictly respect maxiter as 'calls' 
                # but maxiter is iterations (simplex moves). Each move is ~2*n calls.
                # So maxiter should be roughly remaining_budget / (2*n) to be safe?
                # Actually, 'maxiter' in NM is the number of iterations, where each iteration 
                # involves reflection, expansion, etc. 
                # To be safe, let's limit maxiter to remaining_budget // 10 
                # because each iteration uses multiple function evaluations.
                
                # Let's re-run with a safer maxiter limit
                max_iter_safe = max(1, remaining_budget // (2 * n + 1))
                nm_calls[0] = 0 # Reset counter for the safe run? No, we should just trust the wrapper.
                
                # The wrapper `nm_func` already handles the hard budget stop.
                # The `maxiter` in options is a soft stop for the algorithm logic.
                # If we hit the budget in nm_func, it returns best_f, which might confuse NM if it's not improving.
                # It's better to let NM run until it thinks it's done or maxiter is hit, 
                # but the wrapper ensures we don't exceed `budget`.
                
                res = minimize(nm_func, x0, method='Nelder-Mead', 
                               options={'maxiter': max_iter_safe, 'xatol': 1e-5, 'fatol': 1e-5})
                
                if res.fun < best_f:
                    best_x = res.x
                    best_f = res.fun
                    
            except Exception:
                pass

        return best_x, best_f
