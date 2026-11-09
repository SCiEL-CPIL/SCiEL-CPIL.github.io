import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = None

    def __call__(self, func):
        self.start_time = time.time()
        rng = np.random.default_rng(42)
        n = self.dim
        N = 10  # Population size
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover rate

        # Initialize population
        pop = rng.uniform(-5, 5, (N, n))
        fpop = np.array([func(x) for x in pop])
        self.calls += N
        
        best_idx = np.argmin(fpop)
        x_best = pop[best_idx].copy()
        f_best = fpop[best_idx]
        
        sigma_local = 1.0

        # Main Loop
        while self.calls < self.budget:
            # Check time limit (safety margin 10s)
            if time.time() - self.start_time > 320:
                break
                
            # 1. Local Polishing on the best individual
            # We spend a few evaluations to refine the current best
            # This helps AOCC by reducing f_best quickly
            polish_steps = 3
            # Ensure we have enough budget for polish
            if self.calls + polish_steps <= self.budget:
                for _ in range(polish_steps):
                    y = np.clip(x_best + sigma_local * rng.standard_normal(n), -5, 5)
                    fy = func(y)
                    self.calls += 1
                    if fy < f_best:
                        x_best = y
                        f_best = fy
                        sigma_local *= 1.2
                    else:
                        sigma_local *= 0.7
            else:
                break

            # 2. DE Generation
            # We need to update the population. 
            # Note: x_best might not be in pop if it was improved by local polish 
            # and pop[best_idx] is stale. We should update pop[best_idx] to x_best.
            pop[best_idx] = x_best
            fpop[best_idx] = f_best

            for i in range(N):
                if self.calls >= self.budget:
                    break
                
                # Select 3 distinct random indices
                while True:
                    a, b, c = rng.choice(N, 3, replace=False)
                    if a != i and b != i and c != i:
                        break
                
                # Mutation
                v = pop[a] + F * (pop[b] - pop[c])
                # Crossover
                r = rng.random(n)
                j_rand = rng.integers(n)
                u = np.where(r < CR, v, pop[i])
                u[j_rand] = v[j_rand]
                u = np.clip(u, -5, 5)
                
                fu = func(u)
                self.calls += 1
                
                if fu < fpop[i]:
                    pop[i] = u
                    fpop[i] = fu
                    if fu < f_best:
                        f_best = fu
                        x_best = u.copy()
                        best_idx = i
                        # Reset local sigma? Or keep? Keeping is fine, it adapts.
        
        return x_best, f_best
