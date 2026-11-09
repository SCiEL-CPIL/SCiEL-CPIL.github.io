"""Rewritten program using Differential Evolution with Non-Dominated Archive for multi-objective optimization."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-objective optimizer using Differential Evolution (DE) with a Non-Dominated Archive."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        budget = self.budget
        
        # DE Parameters
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover rate
        pop_size = min(20, budget // 10)
        if pop_size < 10:
            pop_size = 10
            
        # Ensure we have enough budget for initialization
        if budget < pop_size + 10:
             pop_size = budget - 10
             if pop_size < 5:
                 pop_size = 5

        # Initialize Population
        population_x = lo + rng.random((pop_size, dim)) * span
        population_f = np.array([problem(x) for x in population_x])
        used = pop_size
        
        # Archive of non-dominated solutions
        archive_x = []
        archive_f = []
        
        def is_dominated(f1, f2):
            """Check if f1 dominates f2. f1 <= f2 and f1 != f2."""
            if np.all(f1 <= f2) and np.any(f1 < f2):
                return True
            return False

        def add_to_archive(x, f):
            """Add point to archive if non-dominated, removing dominated points."""
            # Check if dominated by any existing point
            for ef in archive_f:
                if is_dominated(ef, f):
                    return False
            
            # Remove points dominated by new point
            keep_indices = []
            for i, ef in enumerate(archive_f):
                if is_dominated(f, ef):
                    continue
                keep_indices.append(i)
            
            if len(keep_indices) < len(archive_f):
                archive_x[:] = [archive_x[i] for i in keep_indices]
                archive_f[:] = [archive_f[i] for i in keep_indices]
            
            archive_x.append(x)
            archive_f.append(f)
            return True

        def prune_archive(max_size=50):
            """Prune archive if it exceeds max_size using crowding distance."""
            if len(archive_f) <= max_size:
                return
            
            f_arr = np.array(archive_f)
            f_min = f_arr.min(axis=0)
            f_max = f_arr.max(axis=0)
            f_range = f_max - f_min
            f_range[f_range == 0] = 1.0
            f_norm = (f_arr - f_min) / f_range
            
            n = len(f_norm)
            crowd = np.zeros(n)
            
            for m in range(self.n_obj):
                order = np.argsort(f_norm[:, m])
                if n > 1:
                    crowd[order[0]] = np.inf
                    crowd[order[-1]] = np.inf
                if n > 2:
                    for i in range(1, n - 1):
                        idx_prev = order[i-1]
                        idx_next = order[i+1]
                        dist = f_norm[idx_next, m] - f_norm[idx_prev, m]
                        crowd[order[i]] += dist / f_range[m]
            
            # Remove points with smallest crowding distance until size <= max_size
            while len(archive_f) > max_size:
                min_crowd_idx = np.argmin(crowd)
                archive_x.pop(min_crowd_idx)
                archive_f.pop(min_crowd_idx)
                # Recalculate crowd distances for remaining points
                f_arr = np.array(archive_f)
                f_min = f_arr.min(axis=0)
                f_max = f_arr.max(axis=0)
                f_range = f_max - f_min
                f_range[f_range == 0] = 1.0
                f_norm = (f_arr - f_min) / f_range
                n = len(f_norm)
                crowd = np.zeros(n)
                for m in range(self.n_obj):
                    order = np.argsort(f_norm[:, m])
                    if n > 1:
                        crowd[order[0]] = np.inf
                        crowd[order[-1]] = np.inf
                    if n > 2:
                        for i in range(1, n - 1):
                            idx_prev = order[i-1]
                            idx_next = order[i+1]
                            dist = f_norm[idx_next, m] - f_norm[idx_prev, m]
                            crowd[order[i]] += dist / f_range[m]

        # Initialize Archive with population
        for i in range(pop_size):
            add_to_archive(population_x[i].copy(), population_f[i].copy())
        prune_archive(50)

        # DE Loop
        while used < budget:
            # Select 3 distinct random indices for DE vector
            i = rng.integers(0, pop_size)
            a, b, c = rng.choice(pop_size, size=3, replace=False)
            
            # Ensure a, b, c are distinct from i and each other
            # rng.choice with replace=False ensures a,b,c are distinct.
            # We need to ensure i is not in {a,b,c} for standard DE, 
            # but even if it is, it's a valid variant (self-adaptive or similar).
            # To be strict, let's just pick 3 distinct from population, 
            # and if one matches i, we just proceed (it's fine).
            
            # Create trial vector
            # v = x_a + F * (x_b - x_c)
            v = population_x[a] + F * (population_x[b] - population_x[c])
            
            # Crossover
            trial_x = np.copy(population_x[i])
            j_rand = rng.integers(0, dim)
            for j in range(dim):
                if rng.random() < CR or j == j_rand:
                    trial_x[j] = v[j]
            
            # Clip to bounds
            trial_x = np.clip(trial_x, lo, hi)
            
            # Evaluate
            trial_f = problem(trial_x)
            used += 1
            
            # Selection: Replace if trial is non-dominated by target or dominates target
            # Simple non-dominated selection:
            # If trial dominates target -> replace
            # If target dominates trial -> keep target
            # If mutually non-dominated -> keep both? 
            # In DE, we usually replace if trial is "better". 
            # For multi-objective, we can use a dominance check.
            # If trial_f is non-dominated by target_f (i.e., target doesn't dominate trial), 
            # and (trial dominates target OR trial is non-dominated), we might replace.
            # A common strategy: Replace if trial is non-dominated by target.
            # This encourages diversity.
            
            if not is_dominated(population_f[i], trial_f):
                # Target does not dominate trial. 
                # This includes: Trial dominates target, or they are non-dominated.
                # We replace to allow exploration of non-dominated regions.
                population_x[i] = trial_x
                population_f[i] = trial_f
            
            # Update Archive
            add_to_archive(trial_x.copy(), trial_f.copy())
            
            if len(archive_f) > 50:
                prune_archive(50)

        return population_x

# EVOLVE-BLOCK-END
