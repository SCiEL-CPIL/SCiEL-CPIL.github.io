"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Differential Evolution optimizer for Rastrigin-like functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        
    def __call__(self, func):
        # Deadline guard: 300s limit, use 290s as safety margin
        deadline = time.time() + 290.0
        
        # Parameters
        d = self.dim
        npop = 10  # Population size
        F = 0.8    # Mutation factor
        CR = 0.5   # Crossover rate
        
        # Initialize population
        # Use a fixed seed for reproducibility within the run, but the 
        # evaluator creates a new Optimizer instance for each problem instance.
        # Using a fixed seed ensures consistent behavior for the same problem.
        rng = np.random.default_rng(42)
        
        pop = np.random.uniform(self.bounds[0], self.bounds[1], (npop, d))
        # Clip just in case
        pop = np.clip(pop, self.bounds[0], self.bounds[1])
        
        # Evaluate initial population
        fvals = np.array([func(x) for x in pop])
        best_idx = np.argmin(fvals)
        best_x = pop[best_idx].copy()
        best_f = fvals[best_idx]
        
        calls_used = npop
        total_budget = self.budget
        
        # Main loop
        while calls_used + npop <= total_budget:
            # Check time
            if time.time() > deadline:
                break
            
            for i in range(npop):
                if time.time() > deadline:
                    break
                
                # Mutation: v = x_r1 + F * (x_r2 - x_r3)
                # Choose distinct random indices
                r1, r2, r3 = rng.choice(npop, 3, replace=False)
                # Ensure they are distinct from i if possible, but DE usually allows any
                # Standard DE: r1, r2, r3 are distinct from each other.
                # We can allow r1==i etc., but it's better to avoid degenerate cases.
                # Simple choice:
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover: Binomial
                # Trial vector u
                u = v.copy()
                j_rand = rng.integers(0, d)
                for j in range(d):
                    if rng.random() < CR or j == j_rand:
                        u[j] = v[j]
                    else:
                        u[j] = pop[i][j]
                
                # Clip to bounds
                u = np.clip(u, self.bounds[0], self.bounds[1])
                
                # Evaluate
                fu = func(u)
                calls_used += 1
                
                # Selection
                if fu <= fvals[i]:
                    pop[i] = u
                    fvals[i] = fu
                    if fu < best_f:
                        best_f = fu
                        best_x = u.copy()
                        
            # End of generation
            
        # If we have remaining budget and time, do a local refinement?
        # Or just return best.
        # With 500 budget and npop=10, we do 50 generations.
        # This is a good balance.
        
        return best_x, best_f
# EVOLVE-BLOCK-END
