import numpy as np
import time

class Optimizer:
    """Differential Evolution optimizer for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        rng = np.random.default_rng(42)
        
        # Parameters
        N = 50  # Population size
        F = 0.8 # Mutation factor
        CR = 0.5 # Crossover rate
        
        # Initialization
        # Generate initial population in [-5, 5]^dim
        pop = rng.uniform(-5, 5, (N, self.dim))
        fit = np.empty(N)
        
        calls = 0
        for i in range(N):
            fit[i] = func(pop[i])
            calls += 1
            if time.time() > deadline:
                break
        
        best_idx = np.argmin(fit)
        best_x = pop[best_idx].copy()
        best_f = fit[best_idx]
        
        # Main Loop
        # We have used N calls. We can make (budget - N) more calls.
        # Each generation uses N calls.
        # Generations = (budget - N) // N
        remaining_budget = self.budget - calls
        generations = remaining_budget // N
        
        for g in range(generations):
            if time.time() > deadline:
                break
                
            for i in range(N):
                if time.time() > deadline:
                    break
                
                # Select three distinct random indices different from i
                # We need r1, r2, r3 != i and r1 != r2 != r3
                # Simple way: choose 3 distinct indices from [0, N-1] excluding i
                # Or just pick 3 randoms and ensure they are distinct and not i
                r1, r2, r3 = rng.choice(N, 3, replace=False)
                # Ensure none of them is i
                # The probability of collision is low, but let's be safe.
                # Actually, rng.choice(N, 3, replace=False) gives distinct r1, r2, r3.
                # We just need to ensure they are not i.
                # If one is i, we redraw.
                while r1 == i or r2 == i or r3 == i:
                    r1, r2, r3 = rng.choice(N, 3, replace=False)
                
                # Mutation: v = x_r1 + F * (x_r2 - x_r3)
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover
                j_rand = rng.integers(0, self.dim)
                u = np.empty(self.dim)
                for j in range(self.dim):
                    if rng.random() < CR or j == j_rand:
                        u[j] = v[j]
                    else:
                        u[j] = pop[i][j]
                
                # Clip to bounds
                u = np.clip(u, -5, 5)
                
                # Evaluate
                f_u = func(u)
                calls += 1
                
                # Selection
                if f_u <= fit[i]:
                    pop[i] = u
                    fit[i] = f_u
                    if f_u < best_f:
                        best_f = f_u
                        best_x = u.copy()
        
        return best_x, best_f
