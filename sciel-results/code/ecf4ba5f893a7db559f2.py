"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution optimizer for Rastrigin-like functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # DE Parameters
        # NP must be small to allow multiple generations within budget
        NP = 12 
        F = 0.8
        CR = 0.9

        # Initialize population
        pop = rng.uniform(-5, 5, (NP, dim))
        fitness = np.array([func(x) for x in pop])
        calls_made = NP
        
        # Track best
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_f = fitness[best_idx]

        while calls_made < budget:
            # Iterate through population to generate trial vectors
            for i in range(NP):
                if calls_made >= budget:
                    break
                
                # Select three distinct indices r1, r2, r3 different from i
                # Simple way: sample without replacement from all indices except i
                candidates = [j for j in range(NP) if j != i]
                r1, r2, r3 = rng.choice(candidates, size=3, replace=False)
                
                # Mutation
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover
                j_rand = rng.integers(dim)
                u = pop[i].copy()
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        u[j] = v[j]
                
                # Boundary handling
                u = np.clip(u, -5, 5)
                
                # Evaluation
                fu = func(u)
                calls_made += 1
                
                # Selection
                if fu <= fitness[i]:
                    pop[i] = u
                    fitness[i] = fu
                    if fu < best_f:
                        best_f = fu
                        best_x = u.copy()
            
            # Optional: Shuffling or reordering? Standard DE doesn't require it, 
            # but maintaining diversity is key. The population update handles this.
            
        return best_x, best_f
# EVOLVE-BLOCK-END
