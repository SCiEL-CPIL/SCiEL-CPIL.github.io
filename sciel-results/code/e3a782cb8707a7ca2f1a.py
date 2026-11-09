"""Differential Evolution optimizer for BBOB-style f20 (Schwefel) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution optimizer with strict budget management."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit

        # DE Parameters
        pop_size = 10  # Must be at least 4
        F = 0.8
        CR = 0.5
        rng = np.random.default_rng(42)

        # Initialize population
        pop = rng.uniform(-5, 5, (pop_size, dim))
        fitness = np.array([func(x) for x in pop])
        calls = pop_size
        
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_f = fitness[best_idx]
        
        # Track stagnation
        no_improve = 0
        stagnation_limit = 20 # generations

        while calls + pop_size <= budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate offspring
            for i in range(pop_size):
                # Select distinct random indices
                r1, r2 = rng.choice(pop_size, 2, replace=False)
                while r1 == i or r2 == i:
                    r1, r2 = rng.choice(pop_size, 2, replace=False)
                
                # Mutation
                v = pop[r1] + F * (pop[r2] - pop[i])
                
                # Crossover
                u = pop[i].copy()
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        u[j] = v[j]
                
                # Clip to bounds
                u = np.clip(u, -5, 5)
                
                # Evaluate
                fu = func(u)
                calls += 1
                
                # Selection
                if fu <= fitness[i]:
                    pop[i] = u
                    fitness[i] = fu
                    if fu < best_f:
                        best_f = fu
                        best_x = u.copy()
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
            
            # Stagnation check
            if no_improve > stagnation_limit:
                # Only restart if we have budget for it
                restart_cost = pop_size - 1
                if calls + restart_cost <= budget:
                    for i in range(pop_size):
                        if i != np.argmin(fitness):
                            pop[i] = rng.uniform(-5, 5, dim)
                            fitness[i] = func(pop[i])
                            calls += 1
                            if fitness[i] < best_f:
                                best_f = fitness[i]
                                best_x = pop[i].copy()
                    no_improve = 0
                else:
                    # Not enough budget for restart, just break
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
