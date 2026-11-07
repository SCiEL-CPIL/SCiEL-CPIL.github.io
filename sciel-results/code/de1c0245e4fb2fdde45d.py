import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self._start_time = None

    def __call__(self, problem, bounds):
        self._start_time = time.time()
        deadline = self._start_time + 290  # 300s limit, keep 10s margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        d = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        # Archive-based DE with adaptive mutation
        # Initialize with a diverse set
        pop_size = 20
        X = lo + (hi - lo) * rng.random((pop_size, d))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Build initial archive
        archive_X = [X[i] for i in range(pop_size)]
        archive_F = [F[i] for i in range(pop_size)]
        
        # Function to check if a point is dominated by any in the list
        def is_dominated(f_new, F_list):
            for f in F_list:
                if np.all(f <= f_new) and np.any(f < f_new):
                    return True
            return False
        
        # Function to remove points dominated by f_new from list
        def remove_dominated(f_new, X_list, F_list):
            new_X = []
            new_F = []
            for x, f in zip(X_list, F_list):
                if not (np.all(f_new <= f) and np.any(f_new < f)):
                    new_X.append(x)
                    new_F.append(f)
            return new_X, new_F
        
        # Update archive with new point
        def update_archive(x_new, f_new):
            nonlocal archive_X, archive_F
            if is_dominated(f_new, archive_F):
                return
            # Remove dominated points
            archive_X, archive_F = remove_dominated(f_new, archive_X, archive_F)
            archive_X.append(x_new)
            archive_F.append(f_new)
        
        # Update initial archive
        archive_X = []
        archive_F = []
        for i in range(pop_size):
            update_archive(X[i], F[i])
        
        # DE/rand/1 with archive-based selection
        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break
            
            # Select 3 parents from archive (or population if archive is small)
            source = archive if len(archive_X) >= 3 else None
            if source is not None:
                a, b, c = rng.choice(len(archive_X), size=3, replace=False)
                v = archive_X[a] + 0.5 * (archive_X[b] - archive_X[c])
            else:
                a, b, c = rng.choice(pop_size, size=3, replace=False)
                v = X[a] + 0.5 * (X[b] - X[c])
            
            # Clip to bounds
            v = np.clip(v, lo, hi)
            
            # Evaluate
            fv = problem(v)
            used += 1
            
            # Update archive
            update_archive(v, fv)
            
            # Occasionally replace a random member of population to maintain diversity
            if used % 5 == 0 and len(archive_X) > 0:
                idx = rng.integers(pop_size)
                X[idx] = v
                F[idx] = fv
        
        # Return the best non-dominated points found
        # Use the archive as the result
        if len(archive_X) > 0:
            return np.array(archive_X)
        return X
