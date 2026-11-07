"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Differential Evolution (DE) for BBOB f20."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.start_time = None

    def __call__(self, func):
        self.start_time = time.time()
        dim = self.dim
        budget = self.budget
        
        # DE Parameters
        N = 10  # Population size
        F = 0.8  # Scaling factor
        CR = 0.9  # Crossover rate
        
        # Ensure N is even? Not strictly necessary for Rand/1/DE but good practice.
        # Budget check: 10 initial + (budget - 10) // 10 * 10 evaluations.
        # If budget < N, we just do random search or simple hill climb.
        
        if budget < N:
            # Fallback for very small budgets
            x = np.random.uniform(-5, 5, dim)
            fx = func(x)
            for _ in range(budget - 1):
                y = np.random.uniform(-5, 5, dim)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
            return x, fx

        rng = np.random.default_rng(42)
        
        # Initialize population
        pop = rng.uniform(-5, 5, (N, dim))
        fit = np.array([func(x) for x in pop])
        
        best_idx = np.argmin(fit)
        best_x = pop[best_idx].copy()
        best_f = fit[best_idx]
        
        calls_made = N
        
        # Main loop
        # We have budget - calls_made calls left.
        # Each generation consumes N calls.
        max_generations = (budget - calls_made) // N
        
        for gen in range(max_generations):
            # Time guard: 300s limit, keep 10s margin
            if time.time() - self.start_time > 290:
                break
            
            for i in range(N):
                # Time guard inside loop just in case
                if time.time() - self.start_time > 290:
                    break
                
                # Select 3 distinct random indices different from i
                # Using rng.choice for simplicity, though sampling without replacement is faster
                while True:
                    a, b, c = rng.integers(0, N, 3)
                    if a != i and b != i and c != i and a != b and b != c and a != c:
                        break
                
                # Mutate: v = x_a + F * (x_b - x_c)
                v = pop[a] + F * (pop[b] - pop[c])
                
                # Clip to bounds
                v = np.clip(v, -5, 5)
                
                # Crossover
                j_rand = rng.integers(0, dim)
                u = np.where(rng.random(dim) < CR, v, pop[i])
                u[j_rand] = v[j_rand]  # Ensure at least one gene from mutant
                
                # Evaluate
                fu = func(u)
                calls_made += 1
                
                # Selection
                if fu < fit[i]:
                    pop[i] = u
                    fit[i] = fu
                    if fu < best_f:
                        best_f = fu
                        best_x = u.copy()
                        
        return best_x, best_f
# EVOLVE-BLOCK-END
