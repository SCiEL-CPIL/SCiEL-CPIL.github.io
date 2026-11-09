import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # Reserve budget for local search
        local_budget = 50
        de_budget = budget - local_budget
        
        # DE Parameters
        pop_size = 30
        F = 0.8
        CR = 0.9
        
        # Use system RNG for robustness across instances? 
        # Or fixed? Let's use fixed for reproducibility in evaluation logs.
        rng = np.random.default_rng(42)
        
        # Initialize Population
        # Uniform in [-5, 5]
        Pop = rng.uniform(-5, 5, (pop_size, n))
        Fit = np.array([func(x) for x in Pop])
        calls = pop_size
        
        best_idx = np.argmin(Fit)
        best_x = Pop[best_idx].copy()
        best_f = Fit[best_idx]
        
        start_time = time.time()
        deadline = start_time + 280
        
        # DE Loop
        while calls < de_budget:
            if time.time() > deadline:
                break
                
            for i in range(pop_size):
                if calls >= de_budget:
                    break
                
                # Select three distinct random individuals
                # Simple rejection sampling
                while True:
                    a, b, c = rng.integers(0, pop_size, 3)
                    if a != i and b != i and c != i and a != b and a != c:
                        break
                
                # Mutation
                v = Pop[a] + F * (Pop[b] - Pop[c])
                v = np.clip(v, -5, 5)
                
                # Crossover
                j_rand = rng.integers(0, n)
                u = np.zeros(n)
                # Vectorized crossover
                mask = rng.random(n) < CR
                mask[j_rand] = True
                u = np.where(mask, v, Pop[i])
                
                # Evaluate
                fu = func(u)
                calls += 1
                
                # Selection
                if fu <= Fit[i]:
                    Pop[i] = u
                    Fit[i] = fu
                    if fu < best_f:
                        best_f = fu
                        best_x = u.copy()
                        
        # Local Search
        remaining_calls = budget - calls
        if remaining_calls > 0:
            call_count = {'count': 0}
            def wrapped_func(x):
                if call_count['count'] >= remaining_calls:
                    return best_f
                val = func(x)
                call_count['count'] += 1
                return val
                
            try:
                # Nelder-Mead
                res = minimize(wrapped_func, best_x, method='Nelder-Mead', 
                               options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass
                
        return best_x, best_f
