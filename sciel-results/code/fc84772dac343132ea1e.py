"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer using a two-stage strategy:
    1. Differential Evolution (DE) for global exploration to find good basins.
    2. (1+1)-ES for local refinement/polish around the best found solution.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)
        
        best_x = None
        best_f = float('inf')
        total_calls = 0

        def check_deadline():
            return time.time() > deadline or total_calls >= budget

        def update_best(x, fx):
            nonlocal best_x, best_f
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 1: Differential Evolution (Global Search)
        # Allocate 60% of budget to DE
        de_budget = int(budget * 0.6)
        pop_size = 12
        # Number of generations we can afford
        # Each generation costs pop_size calls (roughly, ignoring initial)
        # We initialize pop_size, then for each gen we do pop_size evaluations.
        
        # Initialize population
        population = []
        for i in range(pop_size):
            if i == 0:
                x = np.zeros(d)
            else:
                x = rng.uniform(-5, 5, d)
            fx = func(x)
            total_calls += 1
            population.append((x, fx))
            update_best(x, fx)
            
            if check_deadline():
                break

        # DE Parameters
        F = 0.8  # Mutation scale
        CR = 0.9  # Crossover rate
        
        # Run DE until budget for this phase is exhausted or deadline
        while total_calls + pop_size <= de_budget and not check_deadline():
            new_population = []
            for i in range(pop_size):
                if check_deadline():
                    break
                
                x_i, f_i = population[i]
                
                # Select 3 distinct random indices different from i
                r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                while r1 == i:
                    r1 = rng.choice(pop_size)
                while r2 == i or r2 == r1:
                    r2 = rng.choice(pop_size)
                while r3 == i or r3 == r1 or r3 == r2:
                    r3 = rng.choice(pop_size)
                    
                x_r1, _ = population[r1]
                x_r2, _ = population[r2]
                x_r3, _ = population[r3]
                
                # Mutation: v = x_r1 + F * (x_r2 - x_r3)
                v = x_r1 + F * (x_r2 - x_r3)
                
                # Crossover
                u = np.zeros(d)
                j_rand = rng.integers(0, d)
                for j in range(d):
                    if rng.random() < CR or j == j_rand:
                        u[j] = v[j]
                    else:
                        u[j] = x_i[j]
                        
                # Clip to bounds
                u = np.clip(u, -5, 5)
                
                f_u = func(u)
                total_calls += 1
                
                # Selection
                if f_u <= f_i:
                    new_population.append((u, f_u))
                    update_best(u, f_u)
                else:
                    new_population.append((x_i, f_i))
            
            if not new_population:
                break
            population = new_population

        # Phase 2: (1+1)-ES Polish (Local Refinement)
        if best_x is not None and not check_deadline():
            x = best_x.copy()
            fx = best_f
            sigma = 0.5 # Start with a moderate step size
            
            remaining_calls = budget - total_calls
            # Use all remaining budget for polish
            for _ in range(remaining_calls):
                if check_deadline():
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                fy = func(y)
                total_calls += 1
                
                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fx)
                    # Increase sigma slightly on success to explore more
                    sigma *= 1.1
                else:
                    # Decrease sigma on failure to refine
                    sigma *= 0.85
                
                # Stop if sigma is too small
                if sigma < 1e-8:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
