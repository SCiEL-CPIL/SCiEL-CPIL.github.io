"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution optimizer for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.deadline = 290.0  # Safety margin for 300s limit

    def __call__(self, func):
        start_time = time.time()
        rng = np.random.default_rng(42)
        
        # Parameters
        NP = 10  # Population size
        F = 0.8  # Differential weight
        CR = 0.5 # Crossover rate
        
        # Initialize population uniformly in [-5, 5]^dim
        pop = rng.uniform(-5, 5, (NP, self.dim))
        fits = np.array([func(ind) for ind in pop])
        
        best_idx = np.argmin(fits)
        best_x = pop[best_idx].copy()
        best_f = fits[best_idx]
        
        evals_used = NP
        budget = self.budget
        
        # Differential Evolution Loop
        while evals_used < budget:
            # Check time limit
            if time.time() > self.deadline:
                break
                
            for i in range(NP):
                if evals_used >= budget:
                    break
                if time.time() > self.deadline:
                    break
                    
                # Select three distinct random indices different from i
                # We can just pick 3 random indices and ensure they are not i
                # For small NP, simple rejection sampling or permutation is fine
                indices = list(range(NP))
                indices.remove(i)
                a, b, c = rng.choice(indices, size=3, replace=False)
                
                # Create mutant vector
                v = pop[a] + F * (pop[b] - pop[c])
                
                # Create trial vector u
                # Standard DE/rand/1/bin
                r = rng.random()
                if r < CR:
                    j_rand = rng.integers(0, self.dim)
                else:
                    j_rand = i # dummy, will be overwritten or handled
                # Actually, standard DE picks j_rand once per trial vector
                j_rand = rng.integers(0, self.dim)
                
                u = np.empty_like(pop[i])
                for j in range(self.dim):
                    if rng.random() < CR or j == j_rand:
                        u[j] = v[j]
                    else:
                        u[j] = pop[i][j]
                
                # Clip to box
                u = np.clip(u, -5, 5)
                
                # Evaluate
                fu = func(u)
                evals_used += 1
                
                # Selection
                if fu < fits[i]:
                    pop[i] = u
                    fits[i] = fu
                    if fu < best_f:
                        best_f = fu
                        best_x = u.copy()
                        
        return best_x, best_f
# EVOLVE-BLOCK-END
