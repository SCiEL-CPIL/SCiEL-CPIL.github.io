"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Evolutionary multi-objective search with deadline guard and adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        pop = int(min(20, self.budget // 4))
        X = lo + range_ * rng.random((pop, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop
        best = F.copy()
        
        # Adaptive mutation rate
        base_mutation = 0.1
        mutation_scale = 1.0
        
        while used < self.budget:
            # Check deadline
            if time.time() > deadline:
                break
            
            # Non-dominated sorting
            dominated = np.array([
                np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1)) 
                for f in F
            ])
            parents = X[~dominated] if np.any(~dominated) else X
            
            # Adaptive mutation: decrease mutation rate as we converge
            progress = used / self.budget
            current_mutation = base_mutation * (1.0 - 0.5 * progress)
            
            # Select parent
            parent = parents[rng.integers(len(parents))]
            
            # Mutate
            child = parent + current_mutation * range_ * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            # Evaluate
            fc = problem(child)
            used += 1
            
            # Update best if child is better in at least one objective
            if np.any(fc < best):
                best = np.minimum(best, fc)
            
            # Replace worst individual (by sum of objectives)
            worst = int(np.argmax(np.sum(F, axis=1)))
            X[worst], F[worst] = child, fc
        
        return X
# EVOLVE-BLOCK-END
