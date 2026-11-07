"""Rewritten multi-objective optimizer for WFG4 using two-phase LHS + archive-based DE."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-phase multi-objective optimizer: LHS exploration then archive-based DE refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 280.0  # Safety margin from 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Reference point for hypervolume (only points better than this count)
        ref = np.array([3.0, 5.0, 7.0])

        # Archive of non-dominated solutions
        archive_X = []
        archive_F = []

        def is_dominated(f, archive_F):
            """Check if f is dominated by any point in archive_F."""
            if len(archive_F) == 0:
                return False
            af = np.array(archive_F)
            # f is dominated if there exists af_i such that af_i <= f in all objectives and af_i < f in at least one
            dominated = np.any(np.all(af <= f, axis=1) & np.any(af < f, axis=1))
            return dominated

        def add_to_archive(x, f):
            """Add (x, f) to archive if non-dominated, removing any dominated points."""
            non_local = []
            # Check if f is dominated by existing archive
            if len(archive_F) > 0:
                af = np.array(archive_F)
                if np.any(np.all(af <= f, axis=1) & np.any(af < f, axis=1)):
                    return  # f is dominated, don't add
            # Remove archive points dominated by f
            keep = []
            for i, af in enumerate(archive_F):
                if not (np.all(f <= af) and np.any(f < af)):
                    keep.append(i)
            for i in keep:
                non_local.append((archive_X[i], archive_F[i]))
            # Add new point
            non_local.append((x, f))
            archive_X.clear()
            archive_F.clear()
            for px, pf in non_local:
                archive_X.append(px)
                archive_F.append(pf)

        def time_left():
            return deadline - (time.time() - start_time)

        # Phase 1: Latin Hypercube Sampling for broad exploration
        # Use about 40% of budget for LHS
        lhs_count = min(budget, max(30, budget // 3))
        
        # Generate LHS points
        rng = np.random.default_rng(42)
        lhs_X = np.zeros((lhs_count, dim))
        for d in range(dim):
            perm = rng.permutation(lhs_count)
            lhs_X[:, d] = (perm + rng.random(lhs_count)) / lhs_count
        
        # Map to bounds
        lhs_X = lo + lhs_X * (hi - lo)
        
        # Evaluate LHS points
        used = 0
        for i in range(lhs_count):
            if time_left() < 5.0:
                break
            f = problem(lhs_X[i])
            used += 1
            if np.all(f < ref):  # Only consider points better than reference
                add_to_archive(lhs_X[i].copy(), f.copy())

        if used >= budget or time_left() < 10.0:
            return np.array(archive_X) if archive_X else np.zeros((1, dim))

        # Phase 2: Archive-based Differential Evolution for refinement
        remaining = budget - used
        pop_size = min(50, remaining // 2)
        if pop_size < 5:
            pop_size = 5

        # Initialize population from archive and random points
        pop_X = []
        pop_F = []
        
        # Seed with archive points
        for i in range(min(len(archive_X), pop_size)):
            pop_X.append(archive_X[i].copy())
            pop_F.append(archive_F[i].copy())
        
        # Fill remaining with random points
        while len(pop_X) < pop_size and used < budget and time_left() > 10.0:
            x = lo + rng.random(dim) * (hi - lo)
            f = problem(x)
            used += 1
            pop_X.append(x)
            pop_F.append(f)
            if np.all(f < ref):
                add_to_archive(x.copy(), f.copy())

        pop_X = np.array(pop_X)
        pop_F = np.array(pop_F)

        # DE parameters
        F_factor = 0.8
        CR = 0.9
        max_generations = (budget - used) // max(1, pop_size)
        
        for gen in range(max_generations):
            if time_left() < 10.0 or used >= budget:
                break
            
            for i in range(pop_size):
                if time_left() < 5.0 or used >= budget:
                    break
                
                # Select three distinct parents
                indices = [i]
                while len(indices) < 3:
                    j = rng.integers(pop_size)
                    if j not in indices:
                        indices.append(j)
                a, b, c = indices[1], indices[2], indices[0]
                
                # Mutation
                v = pop_X[a] + F_factor * (pop_X[b] - pop_X[c])
                v = np.clip(v, lo, hi)
                
                # Crossover
                cross_mask = rng.random(dim) < CR
                if not np.any(cross_mask):
                    cross_mask[rng.integers(dim)] = True
                trial = np.where(cross_mask, v, pop_X[i])
                
                # Evaluate
                f_trial = problem(trial)
                used += 1
                
                # Selection
                if np.all(f_trial <= pop_F[i]) and np.any(f_trial < pop_F[i]):
                    pop_X[i] = trial
                    pop_F[i] = f_trial
                elif np.all(pop_F[i] <= f_trial) and np.any(pop_F[i] < f_trial):
                    pass  # Keep current
                else:
                    # Non-dominated: add to archive if better than ref
                    pass
                
                # Add to archive if non-dominated and better than reference
                if np.all(f_trial < ref):
                    add_to_archive(trial.copy(), f_trial.copy())
            
            # Update archive with current population non-dominated points
            for i in range(pop_size):
                if np.all(pop_F[i] < ref):
                    add_to_archive(pop_X[i].copy(), pop_F[i].copy())

        # Final: ensure we haven't exceeded budget
        # Return archive (the recorded objective vectors are what matters)
        if archive_X:
            return np.array(archive_X)
        else:
            return np.zeros((1, dim))
# EVOLVE-BLOCK-END
