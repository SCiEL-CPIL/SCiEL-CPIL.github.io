"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer using Differential Evolution with an external archive.
    Designed for WFG4 (12 dims, 3 objectives) with a budget of 200 evaluations.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Budget allocation
        # 200 evals total. 
        # Initialization: 20 random points.
        # Remaining: 180 evals for DE iterations.
        # Population size: 10 (small to allow more generations/evals)
        # 180 / 10 = 18 generations.
        
        pop_size = 10
        init_pop = 20
        
        # 1. Initialization
        rng = np.random.default_rng(0)
        X_init = lo + range_ * rng.random((init_pop, self.dim))
        F_init = np.array([problem(x) for x in X_init])
        
        # Initialize external archive
        archive_X = []
        archive_F = []
        
        # Helper to add to archive
        def add_to_archive(x, f):
            nonlocal archive_X, archive_F
            # Check if dominated by any in archive
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    return False
            # Remove any in archive dominated by new point
            new_archive_X = []
            new_archive_F = []
            for ax, af in zip(archive_X, archive_F):
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive_X.append(ax)
                    new_archive_F.append(af)
            new_archive_X.append(x)
            new_archive_F.append(f)
            archive_X = new_archive_X
            archive_F = new_archive_F
            return True

        for x, f in zip(X_init, F_init):
            add_to_archive(x, f)

        # Current population for DE (subset of init or random from archive?)
        # Let's start DE population with the first 'pop_size' points
        X = X_init[:pop_size].copy()
        F = F_init[:pop_size].copy()
        
        used_evals = init_pop
        
        # DE Parameters
        CR = 0.9  # Crossover rate
        F_factor = 0.8 # Mutation factor (scaled by range)
        
        # 2. Main Loop
        while used_evals < self.budget:
            # Check time
            if time.time() > deadline:
                break
            
            for i in range(pop_size):
                if used_evals >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                # Select two distinct indices
                idx1, idx2 = rng.choice(pop_size, size=2, replace=False)
                
                # DE/1/rand/bin mutation
                # v = x_rand1 + F * (x_rand2 - x_rand3) 
                # Wait, standard DE/1/rand is x_r1 + F*(x_r2 - x_r3).
                # We need 3 distinct indices from population.
                # If pop_size is small, ensure distinctness.
                indices = rng.choice(pop_size, size=3, replace=False)
                r1, r2, r3 = indices
                
                v = X[r1] + F_factor * (X[r2] - X[r3])
                
                # Clip to bounds
                v = np.clip(v, lo, hi)
                
                # Crossover (Binary)
                # Standard DE uses binary crossover
                jrand = rng.integers(self.dim)
                mask = rng.random(self.dim) < CR
                mask[jrand] = True
                
                trial = np.where(mask, v, X[i])
                
                # Evaluate
                f_trial = problem(trial)
                used_evals += 1
                
                # Selection
                # Compare trial with target X[i]
                if np.all(f_trial <= F[i]) and np.any(f_trial < F[i]):
                    # Trial dominates target
                    X[i] = trial
                    F[i] = f_trial
                    add_to_archive(trial, f_trial)
                elif not (np.all(F[i] <= f_trial) and np.any(F[i] < f_trial)):
                    # Target does not dominate trial (non-dominated relationship)
                    # We might want to keep the one that is better in some sense
                    # Or just keep target to maintain diversity?
                    # Standard DE keeps trial if it's non-dominated against target?
                    # Actually, for multi-objective, we often use a tournament or just keep
                    # the one that contributes better to the front.
                    # Simple strategy: If trial is non-dominated against target,
                    # replace if trial has better crowding distance or just random?
                    # Let's use a simple rule: If trial is non-dominated, keep it 
                    # if it's "better" in terms of average objective or just keep target 
                    # to avoid losing good solutions.
                    # However, to explore, let's replace if trial is non-dominated 
                    # AND (trial is closer to center of bounds or random).
                    # Let's stick to a robust method: 
                    # If non-dominated, we can keep the population diverse by 
                    # occasionally swapping. But simpler: only replace if dominated.
                    # This might stagnate.
                    # Alternative: Maintain a population of the best 'pop_size' 
                    # from the union of current pop and archive? No, that's expensive.
                    
                    # Let's try: If non-dominated, replace with 50% probability 
                    # to encourage exploration, or keep target.
                    # Given the small budget, exploration is key.
                    if rng.random() < 0.5:
                        X[i] = trial
                        F[i] = f_trial
                        add_to_archive(trial, f_trial)
                    else:
                        # Keep target, but ensure target is in archive
                        add_to_archive(X[i], F[i])
                else:
                    # Target dominates trial
                    # Do nothing, but ensure target is in archive
                    add_to_archive(X[i], F[i])
            
            # Optional: Refresh population from archive if stuck?
            # With pop_size 10 and budget 200, we do ~18 generations.
            # The population X evolves. The archive holds the global best non-dominated.

        # Final step: Ensure we return something valid (though return value is ignored)
        # The evaluator records all problem(x) calls.
        return X
# EVOLVE-BLOCK-END
