"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start Differential Evolution for Multi-Objective Optimization (jDE style)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # Archive of non-dominated solutions
        archive_X = []
        archive_F = []
        
        def is_dominated(f, archive_F):
            """Check if f is dominated by any point in archive_F."""
            if len(archive_F) == 0:
                return False
            # Vectorized check: is there any point in archive that is <= f in all obj and < in at least one?
            # f shape: (n_obj,)
            # archive_F shape: (N, n_obj)
            # We want: any j such that archive_F[j] <= f AND any k such that archive_F[j][k] < f[k]
            # Actually, simpler:
            # dominated = any( (archive_F <= f).all(axis=1) & (archive_F < f).any(axis=1) )
            # But we need to be careful with strictness.
            # A point p dominates q if p_i <= q_i for all i and p_j < q_j for some j.
            # Here we check if 'f' is dominated by any 'a' in archive.
            # a <= f and a < f (any)
            le = (archive_F <= f)
            lt = (archive_F < f)
            # For each a in archive: (le for all obj) AND (lt for at least one obj)
            dominated_mask = le.all(axis=1) & lt.any(axis=1)
            return dominated_mask.any()

        def update_archive(x, f):
            """Add (x, f) to archive if non-dominated. Remove dominated points from archive."""
            non_local = True
            if len(archive_F) > 0:
                # Check if f is dominated by any existing point
                if is_dominated(f, np.array(archive_F)):
                    return
            
            # Check if f dominates any existing points
            if len(archive_F) > 0:
                A = np.array(archive_F)
                # f dominates a if f <= a and f < a (any)
                le = (f <= A)
                lt = (f < A)
                dom_mask = le.all(axis=1) & lt.any(axis=1)
                if dom_mask.any():
                    # Remove dominated points
                    keep_mask = ~dom_mask
                    archive_X = [archive_X[i] for i in range(len(archive_X)) if keep_mask[i]]
                    archive_F = [archive_F[i] for i in range(len(archive_F)) if keep_mask[i]]
            
            archive_X.append(x)
            archive_F.append(f)

        # Initialize with a small random population
        init_pop = 10
        for _ in range(init_pop):
            x = lo + span * rng.random(self.dim)
            f = np.array(problem(x))
            update_archive(x, f)
        
        # jDE parameters
        F = 0.5  # Differential weight
        CR = 0.9 # Crossover rate
        # We will use a simple strategy: pick 2 distinct parents from archive (or random if archive small)
        # and mutate.
        
        evals_used = init_pop
        
        # To ensure we don't get stuck in a local region, we occasionally inject random points.
        # But jDE with random restarts or just random parent selection from a diverse archive is good.
        
        # For WFG4, the Pareto front is disconnected. Random restarts are helpful.
        # We will mix: 80% DE mutation from archive, 20% random restart.
        
        while evals_used < self.budget:
            if time.time() > deadline - 2:
                break
            
            # Decide mutation type
            if len(archive_X) >= 3 and rng.random() < 0.8:
                # DE mutation: x_new = x_r1 + F * (x_r2 - x_r3)
                # Select 3 distinct indices from archive
                # If archive is small, we might have issues with distinctness, but archive grows.
                # Let's just sample indices with replacement first, then ensure distinctness if possible.
                # Actually, for small archives, simple random selection is fine.
                idxs = rng.integers(0, len(archive_X), size=3)
                # Ensure r1, r2, r3 are distinct if archive size >= 3
                if len(archive_X) >= 3:
                    # Simple retry to get distinct
                    while idxs[0] == idxs[1] or idxs[1] == idxs[2] or idxs[0] == idxs[2]:
                        idxs = rng.integers(0, len(archive_X), size=3)
                
                x_r1 = archive_X[idxs[0]]
                x_r2 = archive_X[idxs[1]]
                x_r3 = archive_X[idxs[2]]
                
                # Mutation
                x_mut = x_r1 + F * (x_r2 - x_r3)
                
                # Crossover (Binomial)
                x_trial = x_mut.copy()
                jrand = rng.integers(self.dim)
                for j in range(self.dim):
                    if rng.random() <= CR or j == jrand:
                        x_trial[j] = x_mut[j]
                    else:
                        x_trial[j] = x_r1[j]
            else:
                # Random restart
                x_trial = lo + span * rng.random(self.dim)
            
            # Clip to bounds
            x_trial = np.clip(x_trial, lo, hi)
            
            # Evaluate
            f_trial = np.array(problem(x_trial))
            evals_used += 1
            
            # Update archive
            update_archive(x_trial, f_trial)
            
            # Optional: If archive gets too large, we might want to thin it out?
            # Hypervolume calculation on the evaluator side handles the final set.
            # Keeping a large archive helps DE find diverse solutions.
            # However, selecting parents from a huge archive might be slow if we do O(N) operations.
            # Our parent selection is O(1) random index.
            # The dominance check in update_archive is O(N * n_obj).
            # With N up to 200, this is 200*3 = 600 ops per eval. 200 evals * 600 = 120,000 ops. Very fast.
            
        return archive_X

# EVOLVE-BLOCK-END
