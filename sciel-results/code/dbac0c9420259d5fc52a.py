import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0
        
        rng = np.random.default_rng(42)
        calls_made = 0
        best_f = np.inf
        best_x = np.zeros(dim)

        def eval_func(x):
            nonlocal calls_made, best_f, best_x
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def time_check():
            return time.time() < deadline and calls_made < budget

        # Phase 1: Differential Evolution
        # Parameters for 5D
        pop_size = 12
        F = 0.8  # Mutation factor
        CR = 0.9  # Crossover rate
        
        # Initialize population
        # Mix of random and near-origin to help if shift is small
        pop = []
        # 1 point near origin
        pop.append(np.zeros(dim) + 0.1 * rng.standard_normal(dim))
        # Rest random in [-5, 5]
        for _ in range(pop_size - 1):
            pop.append(rng.uniform(-5.0, 5.0, dim))
            
        pop = np.array(pop)
        pop_f = np.array([eval_func(p) for p in pop])
        
        # Track best
        best_idx = np.argmin(pop_f)
        best_x = pop[best_idx].copy()
        best_f = pop_f[best_idx]

        # DE Loop
        while time_check() and calls_made < int(budget * 0.8):
            # Generate new population
            new_pop = []
            new_pop_f = []
            
            for i in range(pop_size):
                if not time_check():
                    break
                    
                # Mutation: DE/rand/1
                a, b, c = rng.choice(pop_size, 3, replace=False)
                while a == i or b == i or c == i:
                    a, b, c = rng.choice(pop_size, 3, replace=False)
                
                v = pop[a] + F * (pop[b] - pop[c])
                
                # Crossover: binomial
                r = rng.random(dim)
                j_rand = rng.integers(0, dim)
                trial = np.where(r < CR, v, pop[i])
                trial[j_rand] = v[j_rand]
                
                # Clip
                trial = np.clip(trial, -5.0, 5.0)
                
                # Evaluate
                trial_f = eval_func(trial)
                
                # Selection
                if trial_f <= pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = trial_f
                    if trial_f < best_f:
                        best_f = trial_f
                        best_x = trial.copy()
                        
            # Check stagnation? If no improvement in many generations, restart or perturb
            # Simple stagnation check
            # For now, just run until budget

        # Phase 2: Local Refinement
        # Use Nelder-Mead from the best point found
        if time_check() and calls_made < budget:
            remaining_budget = budget - calls_made
            if remaining_budget > 10:
                try:
                    # Nelder-Mead options
                    # maxiter is usually iterations, not function calls. 
                    # In 5D, one NM iteration uses ~6-10 calls.
                    # So maxiter should be small if we want to respect call budget strictly, 
                    # but scipy's minimize doesn't have a strict call counter option that stops it cleanly mid-iteration.
                    # We can wrap the function to raise an exception if budget is exceeded, 
                    # or just trust that NM will finish quickly.
                    # Safer: use a wrapper that stops if calls_made >= budget.
                    
                    nm_wrapper = lambda x: eval_func(x)
                    
                    res = minimize(
                        nm_wrapper, 
                        best_x, 
                        method='Nelder-Mead',
                        options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
                    )
                except Exception:
                    pass

        return best_x, best_f
