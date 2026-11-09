"""Improved optimizer: Multi-Objective Differential Evolution (MODE) for RE34."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-Objective Differential Evolution for RE34."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        
        # Determine population size and number of generations
        # A good balance for budget=200 is pop_size=20, generations=10
        # Or pop_size=10, generations=20. 
        # Let's use pop_size=20 to have enough diversity for the archive.
        pop_size = 20
        n_generations = max(1, budget // pop_size)
        
        # If budget is small, adjust
        if pop_size > budget:
            pop_size = max(2, budget)
            n_generations = 1

        rng = np.random.default_rng(42)
        
        # Initialize population
        X = lo + span * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=float)
        
        used = pop_size
        
        # Initialize Archive
        archive_X = []
        archive_F = []
        
        # Helper to add to archive
        def add_to_archive(x, f):
            nonlocal archive_X, archive_F
            # Check if dominated by any in archive
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    return
            # Check if it dominates any in archive, remove them
            new_archive_X = []
            new_archive_F = []
            for i in range(len(archive_F)):
                if not (np.all(f <= archive_F[i]) and np.any(f < archive_F[i])):
                    new_archive_X.append(archive_X[i])
                    new_archive_F.append(archive_F[i])
            
            # If the new point is identical to an existing one, we might skip or keep one
            # For simplicity, we add it. Pruning will handle duplicates if needed, 
            # but DE usually generates distinct points.
            new_archive_X.append(x)
            new_archive_F.append(f)
            
            # Prune archive to limit size (e.g., 50) using crowding distance
            max_archive_size = 50
            if len(new_archive_F) > max_archive_size:
                new_archive_X, new_archive_F = self._prune_archive(new_archive_X, new_archive_F)
            
            archive_X = new_archive_X
            archive_F = new_archive_F

        # Initial population to archive
        for i in range(pop_size):
            add_to_archive(X[i], F[i])

        # DE Parameters
        CR = 0.8  # Crossover rate
        F_factor = 0.5 # Differential weight
        
        for gen in range(n_generations):
            if used >= budget:
                break
            
            new_X = np.zeros_like(X)
            new_F = np.zeros_like(F)
            
            # Select reference points from archive for biasing? 
            # Standard DE uses random population members. 
            # We can also bias towards the archive to exploit good regions.
            # Let's stick to standard DE with random selection from population for robustness.
            
            for i in range(pop_size):
                if used >= budget:
                    break
                
                # Select 3 distinct random individuals from population
                # Ensure they are distinct from i and each other
                indices = [i]
                while len(indices) < 4:
                    r = rng.integers(pop_size)
                    if r not in indices:
                        indices.append(r)
                
                r1, r2, r3 = indices[1], indices[2], indices[3]
                
                # Mutation
                v = X[r1] + F_factor * (X[r2] - X[r3])
                
                # Crossover
                j_rand = rng.integers(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        new_X[i, j] = v[j]
                    else:
                        new_X[i, j] = X[i, j]
                        
                # Clip to bounds
                new_X[i] = np.clip(new_X[i], lo, hi)
                
                # Evaluate
                new_F[i] = problem(new_X[i])
                used += 1
                
                # Update Archive with new point
                add_to_archive(new_X[i], new_F[i])
            
            # Selection: Keep the better of old and new for each individual
            # But since we are doing multi-objective, we can't simply compare.
            # Standard DE selection is tricky for MO. 
            # Common approach: Update the population with the new points if they are non-dominated 
            # or if they dominate the old point. 
            # Alternatively, just replace the population with the new ones if we are using 
            # a "generational" replacement strategy, but this can lose diversity.
            
            # A safer strategy for MO-DE: 
            # For each i, if new_F[i] dominates F[i], replace X[i], F[i].
            # If F[i] dominates new_F[i], keep old.
            # If neither dominates, keep both? No, population size is fixed.
            # We can use a simple rule: If new point is in the archive, it's "good".
            # Let's use a hybrid: If new dominates old, replace. 
            # If old dominates new, keep old.
            # If non-dominated, we can keep old to maintain stability, or replace randomly.
            # Given the small budget, keeping the best known points (archive) is more important.
            # The archive is already updated. 
            # We will update the population to include the new points if they are better or equal, 
            # or simply replace if we want to explore more.
            
            # Let's try: If new dominates old, replace. Otherwise, keep old.
            # This is conservative but ensures we don't lose good points.
            for i in range(pop_size):
                if used >= budget:
                    break
                if np.all(new_F[i] <= F[i]) and np.any(new_F[i] < F[i]):
                    X[i] = new_X[i]
                    F[i] = new_F[i]
                # If they are equal or non-dominated, we keep the old one 
                # to avoid oscillation, relying on the archive for the final result.
                # However, if we always keep old when non-dominated, we might stagnate.
                # Let's allow replacement if the new point is in the archive (i.e., it's non-dominated globally).
                # Checking if a point is in the archive is O(N). 
                # Simpler: Just replace if new is not dominated by old.
                # Wait, if new is not dominated by old, it could be worse in some objectives.
                # Let's stick to: Replace if new dominates old. 
                # To encourage exploration, if they are non-dominated, we can replace with probability 0.5.
                else:
                    # Check if new is dominated by old
                    if not (np.all(F[i] <= new_F[i]) and np.any(F[i] < new_F[i])):
                         # Non-dominated
                         if rng.random() < 0.5:
                             X[i] = new_X[i]
                             F[i] = new_F[i]

        return X[0]

    def _prune_archive(self, archive_X, archive_F):
        """Prune archive using crowding distance to maintain diversity."""
        if len(archive_F) <= 50:
            return archive_X, archive_F
        
        F_arr = np.array(archive_F)
        X_arr = np.array(archive_X)
        
        # Compute crowding distance
        cd = np.zeros(len(F_arr))
        n_obj = F_arr.shape[1]
        n = len(F_arr)
        
        for m in range(n_obj):
            vals = F_arr[:, m]
            mn, mx = vals.min(), vals.max()
            if mx - mn < 1e-15:
                continue
            order = np.argsort(vals)
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            for k in range(1, n - 1):
                cd[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / (mx - mn)
        
        # Keep points with highest crowding distance
        keep_idx = np.argsort(-cd)[:50]
        
        return [X_arr[i] for i in keep_idx], [F_arr[i] for i in keep_idx]
# EVOLVE-BLOCK-END
