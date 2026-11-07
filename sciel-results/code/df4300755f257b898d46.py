"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution optimizer."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim
        self._call_count = 0

    def __call__(self, func):
        self._call_count = 0
        dim = self.dim
        budget = self.budget

        # Counting wrapper
        def wrapped(x):
            if self._call_count >= budget:
                return 1e300
            self._call_count += 1
            try:
                return float(func(x))
            except Exception:
                return 1e300

        rng = np.random.default_rng(42)
        
        # DE Parameters
        pop_size = 10
        F = 0.8  # Mutation factor
        CR = 0.9  # Crossover rate
        
        # Initialize population
        population = rng.uniform(-5.0, 5.0, (pop_size, dim))
        fitness = np.array([wrapped(ind) for ind in population])
        
        best_idx = np.argmin(fitness)
        best_x = population[best_idx].copy()
        best_f = fitness[best_idx]
        
        # Main loop
        while self._call_count < budget:
            for i in range(pop_size):
                if self._call_count >= budget:
                    break
                
                # Select 3 distinct random individuals
                r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                while r1 == r2 or r1 == r3 or r2 == r3:
                    r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                
                # Mutation: v = x[r1] + F * (x[r2] - x[r3])
                v = population[r1] + F * (population[r2] - population[r3])
                
                # Clip to bounds
                v = np.clip(v, -5.0, 5.0)
                
                # Crossover
                u = population[i].copy()
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        u[j] = v[j]
                
                # Evaluate trial vector
                u_f = wrapped(u)
                
                # Selection
                if u_f <= fitness[i]:
                    population[i] = u
                    fitness[i] = u_f
                    if u_f < best_f:
                        best_f = u_f
                        best_x = u.copy()

        return None
# EVOLVE-BLOCK-END
