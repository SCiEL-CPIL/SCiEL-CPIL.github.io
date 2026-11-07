"""Multi-objective optimizer on WFG4 using a two-phase strategy: LHS coverage + archive-based refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-phase multi-objective optimizer for WFG4: LHS initialization + archive-based DE."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin for 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        archive = []  # List of {'x': np.array, 'f': np.array}
        evals = 0

        def is_dominated(f_new, f_ref):
            """Check if f_new is dominated by f_ref."""
            return np.all(f_ref <= f_new) and np.any(f_ref < f_new)

        def is_non_dominated(f_new, archive_F):
            """Check if f_new is non-dominated with respect to the archive."""
            for f in archive_F:
                if is_dominated(f_new, f):
                    return False
            return True

        def update_archive(x, f):
            """Add point to archive if non-dominated, remove dominated points."""
            # Check if new point is dominated by any existing point
            for p in archive:
                if is_dominated(f, p['f']):
                    return  # Dominated, do not add

            # Remove points dominated by the new point
            non_dominated_indices = []
            for i, p in enumerate(archive):
                if not is_dominated(p['f'], f):
                    non_dominated_indices.append(i)

            # Keep non-dominated existing points
            archive[:] = [archive[i] for i in non_dominated_indices]
            # Add new point
            archive.append({'x': x.copy(), 'f': f.copy()})

        def compute_spread(f, archive_F):
            """Compute a simple spread measure for diversity."""
            if len(archive_F) < 2:
                return np.inf
            min_dist = np.inf
            for other_f in archive_F:
                dist = np.linalg.norm(f - other_f)
                min_dist = min(min_dist, dist)
            return min_dist

        # Phase 1: Latin Hypercube Sampling for broad coverage
        n_lhs = min(60, self.budget // 3)
        
        # Generate LHS points
        lhs_points = []
        for _ in range(n_lhs):
            x = np.zeros(self.dim)
            for j in range(self.dim):
                # For each dimension, sample uniformly from a different stratum
                x[j] = lo[j] + (hi[j] - lo[j]) * rng.random()
            lhs_points.append(x)
        
        # Actually, proper LHS: for each dimension, partition into n_lhs strata
        # and sample one point from each stratum per dimension
        lhs_points = []
        for j in range(self.dim):
            # Create strata boundaries for dimension j
            strata = np.linspace(lo[j], hi[j], n_lhs + 1)
            # Sample one point from each stratum
            samples = np.array([strata[i] + (strata[i+1] - strata[i]) * rng.random() for i in range(n_lhs)])
            if j == 0:
                lhs_points = [np.array([s]) for s in samples]
            else:
                for i in range(n_lhs):
                    lhs_points[i] = np.append(lhs_points[i], samples[i])
        
        # Evaluate LHS points
        for x in lhs_points:
            if evals >= self.budget:
                break
            if time.time() > deadline - 5.0:
                break
            f = problem(x)
            evals += 1
            update_archive(x, f)

        # Phase 2: Archive-based DE with local refinement
        F = 0.7  # Mutation factor
        CR = 0.8  # Crossover rate

        # Track best spread for stagnation detection
        best_spread = 0.0
        stagnation_counter = 0
        stagnation_window = 15

        while evals < self.budget:
            if time.time() > deadline - 5.0:
                break
            
            if len(archive) < 2:
                # Not enough points for DE, do random sampling
                x = lo + (hi - lo) * rng.random(self.dim)
                f = problem(x)
                evals += 1
                update_archive(x, f)
                continue

            # Select parents using tournament selection based on spread
            # Pick 3 distinct parents from archive
            if len(archive) >= 3:
                idxs = rng.choice(len(archive), 3, replace=False)
                x1 = archive[idxs[0]]['x']
                x2 = archive[idxs[1]]['x']
                x3 = archive[idxs[2]]['x']
            else:
                # If less than 3 in archive, use random
                x1 = archive[0]['x']
                x2 = archive[0]['x']
                x3 = archive[0]['x']

            # DE Mutation: v = x1 + F * (x2 - x3)
            v = x1 + F * (x2 - x3)
            
            # Crossover: Binomial crossover
            j_rand = rng.integers(0, self.dim)
            child = np.zeros(self.dim)
            for j in range(self.dim):
                if rng.random() < CR or j == j_rand:
                    child[j] = v[j]
                else:
                    child[j] = x1[j]
            
            child = np.clip(child, lo, hi)
            
            # Evaluate child
            if evals >= self.budget:
                break
            f_child = problem(child)
            evals += 1
            
            # Check if non-dominated
            archive_F = [p['f'] for p in archive]
            
            dominated = False
            for f in archive_F:
                if is_dominated(f_child, f):
                    dominated = True
                    break
            
            if not dominated:
                # Compute spread before update
                current_spread = compute_spread(f_child, archive_F)
                
                # Update archive
                update_archive(child, f_child)
                
                # Check for improvement in spread
                new_spread = compute_spread(f_child, [p['f'] for p in archive if p['f'] is not f_child])
                if new_spread > best_spread:
                    best_spread = new_spread
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                
                # If stagnant, do a local search around this point
                if stagnation_counter >= stagnation_window and self.budget - evals >= 5:
                    # Local search: try small perturbations
                    sigma = 0.05 * (hi - lo)
                    for _ in range(3):
                        if evals >= self.budget:
                            break
                        noise = rng.standard_normal(self.dim)
                        x_try = child + sigma * noise
                        x_try = np.clip(x_try, lo, hi)
                        f_try = problem(x_try)
                        evals += 1
                        
                        # Check if non-dominated
                        archive_F_updated = [p['f'] for p in archive]
                        dominated_try = False
                        for f in archive_F_updated:
                            if is_dominated(f_try, f):
                                dominated_try = True
                                break
                        
                        if not dominated_try:
                            update_archive(x_try, f_try)
                            break

        return archive
# EVOLVE-BLOCK-END
