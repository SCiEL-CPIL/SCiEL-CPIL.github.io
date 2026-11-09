"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer using Multi-Start Differential Evolution."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 320.0  # Safety margin before 330s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Use the global numpy random state as seeded by the evaluator
        rng = np.random

        # Total evaluations limit
        # We will use a small population for DE, e.g., 10-20
        # Budget is 200. 
        # Strategy: Split budget into "restarts" or just run one long DE?
        # DE with small pop is fast. Let's try a single DE run with population 20.
        # 200 evals / 20 pop = 10 generations. This is very short.
        # Alternative: Multiple restarts with smaller pops?
        # Or: A larger pop? 
        # The seed used pop=20 and ran 200 evals (10 gens).
        # Let's try a "Memetic" approach: DE + Local Search on non-dominated points.
        # Or simply better DE parameters.
        
        # Let's try: Population size 10. 
        # 200 evals / 10 = 20 generations.
        # Or Population size 5? 40 generations.
        # Let's stick to a robust DE with pop=10 and 20 generations.
        
        pop_size = 10
        max_gens = budget // pop_size
        
        # Initialize population with diversity
        # 1 corner, 1 center, rest random
        X = np.zeros((pop_size, dim))
        X[0] = lo
        X[1] = (lo + hi) / 2
        X[2:] = lo + span * rng.random((pop_size - 2, dim))
        
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Differential Evolution parameters
        CR = 0.9  # Crossover rate
        F_mut = 0.8 # Mutation factor (high for small pop)
        
        # Archive of non-dominated solutions found so far (for final return, though only recorded points matter)
        # The evaluator records EVERY returned objective vector.
        # So we just need to make sure the points we evaluate are good.
        
        while used < budget:
            # Check time
            if time.time() > deadline - 5.0:
                break
            
            # Generate trial vector
            # Standard DE/rand/1/bin
            i = rng.integers(pop_size)
            
            # Select 3 distinct random parents
            r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
            while r1 == i or r2 == i or r3 == i:
                r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
            
            # Mutation
            v = X[r1] + F_mut * (X[r2] - X[r3])
            
            # Crossover
            j_rand = rng.integers(dim)
            for j in range(dim):
                if rng.random() < CR or j == j_rand:
                    trial_x[j] = v[j]
                else:
                    trial_x[j] = X[i][j]
                    
            # Clip to bounds
            trial_x = np.clip(trial_x, lo, hi)
            
            # Evaluate
            trial_f = problem(trial_x)
            used += 1
            
            # Selection
            # If trial dominates current, replace
            # If current does not dominate trial, replace (to keep diversity?)
            # Standard DE: replace if trial is better.
            # In MOO, "better" is ambiguous. 
            # Common heuristic: Replace if trial dominates X[i].
            # To maintain diversity, we might also replace if X[i] is dominated by trial? No, that's the same.
            # Let's use a simple rule: Replace if trial dominates X[i] OR if they are non-dominated and trial has better crowding/distance?
            # Simpler: Replace if trial dominates X[i].
            # To explore more, we can also replace if X[i] dominates trial? No, that loses the better point.
            
            # Let's check dominance
            # trial_f < F[i] in all obj and <= in all
            dominates = np.all(trial_f < F[i]) or (np.all(trial_f <= F[i]) and np.any(trial_f < F[i]))
            dominated = np.all(F[i] < trial_f) or (np.all(F[i] <= trial_f) and np.any(F[i] < trial_f))
            
            # If trial is better (dominates), replace.
            # If they are non-dominated, we might want to keep both. 
            # Since we have a fixed size population, we need a replacement strategy.
            # Let's use a "worst" replacement strategy if non-dominated?
            # Or just stick to "replace if dominates". This can lead to stagnation.
            
            # Alternative: Replace if trial dominates X[i]. 
            # Also, to avoid premature convergence, we can do a "random replacement" with small probability if non-dominated?
            # Let's try: Replace if trial dominates X[i].
            # Additionally, we can maintain an external archive and inject new random points if stagnation is detected?
            # With only 200 evals, stagnation is a big risk.
            
            # Let's try a slightly more aggressive strategy:
            # Replace if trial dominates X[i].
            # If non-dominated, replace with probability 0.1 (to maintain diversity).
            
            if dominates:
                X[i] = trial_x
                F[i] = trial_f
            elif not dominated:
                # Non-dominated
                if rng.random() < 0.1:
                    X[i] = trial_x
                    F[i] = trial_f
            # If trial is dominated, discard
            
        return X
# EVOLVE-BLOCK-END
