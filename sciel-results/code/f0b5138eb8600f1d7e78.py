import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        def track_call(x):
            nonlocal calls_used, best_f, best_x
            if calls_used >= budget:
                return best_f
            val = func(x)
            calls_used += 1
            if val < best_f:
                best_f = val
                best_x = np.array(x)
            return val

        # Strategy: Differential Evolution (DE) with adaptive F and CR
        # DE is robust, population-based, and often outperforms CMA-ES on multimodal functions like Schaffers F7
        # especially when bounds are handled via simple clipping or wrapping.
        
        pop_size = 10  # Standard for dim=5
        F_min, F_max = 0.5, 1.0
        CR = 0.5
        
        # Initialize population uniformly in the box
        pop = np.random.uniform(-5, 5, size=(pop_size, dim))
        fit = np.zeros(pop_size)
        
        for i in range(pop_size):
            if calls_used >= budget:
                break
            fit[i] = track_call(pop[i])
            
        # Track best
        best_idx = np.argmin(fit)
        best_x = pop[best_idx].copy()
        best_f = fit[best_idx]
        
        # DE Main Loop
        while calls_used < budget and time.time() < deadline - 2.0:
            # Update best
            current_best_idx = np.argmin(fit)
            if fit[current_best_idx] < best_f:
                best_f = fit[current_best_idx]
                best_x = pop[current_best_idx].copy()
            
            # Adaptive F: scale with diversity? Or just use a fixed good F?
            # Let's use a standard DE/rand/1 strategy with F=0.8, CR=0.9 which is often robust
            # But we can adapt F slightly. Let's stick to standard DE/rand/1 for simplicity and robustness.
            # Actually, for dim=5, DE/rand/1 with F=0.8, CR=0.9 is a strong baseline.
            # Let's try DE/best/1/2 or similar? No, DE/rand/1 is less prone to premature convergence.
            
            F = 0.8
            CR = 0.9
            
            new_pop = pop.copy()
            new_fit = fit.copy()
            
            for i in range(pop_size):
                if calls_used >= budget:
                    break
                
                # Select 3 distinct random individuals
                while True:
                    r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                    if r1 != i and r2 != i and r3 != i:
                        break
                
                # Differential variation
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Clip to bounds
                v = np.clip(v, -5, 5)
                
                # Crossover
                jrand = np.random.randint(dim)
                trial = pop[i].copy()
                mask = np.random.random(dim) < CR
                mask[jrand] = True
                trial[mask] = v[mask]
                
                # Evaluate
                f_trial = track_call(trial)
                
                # Selection
                if f_trial <= fit[i]:
                    new_pop[i] = trial
                    new_fit[i] = f_trial
            
            pop = new_pop
            fit = new_fit

        return best_x, best_f
