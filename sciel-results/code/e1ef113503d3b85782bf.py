"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution with elitist archive, front-seeded, for WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _dominates(self, a, b):
        """Check if a dominates b (minimization)."""
        return np.all(a <= b) and np.any(a < b)

    def _non_dominated_sort(self, F):
        """Return indices of non-dominated points in F."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=int)
        
        is_dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if is_dominated[i]:
                continue
            for j in range(n):
                if i == j or is_dominated[j]:
                    continue
                if self._dominates(F[i], F[j]):
                    is_dominated[j] = True
        
        return np.where(~is_dominated)[0]

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        ref_point = np.array([3.0, 5.0, 7.0])
        
        # WFG7 specific: The Pareto front is a curved surface.
        # For WFG7, the front is defined by:
        # f1 = 1 - sqrt(x1)
        # f2 = 1 - sqrt(x1)*(1-x2)
        # f3 = 1 - x3
        # But the decision variables are complex. 
        # Instead of guessing the analytic mapping, we use a robust heuristic:
        # 1. Initialize with a mix of random points and points that are "promising"
        #    based on the structure of multi-objective problems (often near bounds or center).
        
        pop_size = 30  # Fixed population size for DE
        archive_size = 50 # Max archive size
        
        # Initialize population
        X = np.empty((pop_size, self.dim))
        F = np.empty((pop_size, self.n_obj))
        
        # Strategy: 50% random, 50% "structured" random (e.g., near boundaries or center)
        # This helps escape local optima and cover the front better.
        for i in range(pop_size):
            if i < pop_size // 2:
                x = lo + range_ * rng.random(self.dim)
            else:
                # Structured: mix of low, high, and mid values
                r = rng.random(self.dim)
                x = np.where(r < 0.3, lo, np.where(r > 0.7, hi, (lo + hi) / 2 + 0.1 * range_ * (r - 0.5)))
            X[i] = x
            F[i] = problem(x)
        
        used = pop_size
        
        # Archive: list of (x, f) tuples that are non-dominated
        archive_X = []
        archive_F = []
        
        # Initial archive population
        nd_idx = self._non_dominated_sort(F)
        for idx in nd_idx:
            archive_X.append(X[idx].copy())
            archive_F.append(F[idx].copy())
        
        # Sort archive by sum of objectives for easier access
        # We will maintain the archive as a list, and prune it if it gets too large
        
        # Differential Evolution Loop
        CR = 0.9  # Crossover rate
        F_mut = 0.8  # Mutation factor
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Select 3 distinct parents
            i = rng.integers(pop_size)
            a, b, c = rng.choice(pop_size, 3, replace=False)
            
            # Ensure i is not in a,b,c
            while i in [a, b, c]:
                i = rng.integers(pop_size)
            
            # Mutation: v = Xa + F * (Xb - Xc)
            v = X[a] + F_mut * (X[b] - X[c])
            v = np.clip(v, lo, hi)
            
            # Crossover
            j_rand = rng.integers(self.dim)
            u = np.empty(self.dim)
            for j in range(self.dim):
                if rng.random() < CR or j == j_rand:
                    u[j] = v[j]
                else:
                    u[j] = X[i][j]
            
            # Evaluate child
            f_u = problem(u)
            used += 1
            
            # Selection: Keep the better of u and X[i] based on dominance
            # If u dominates X[i], replace
            # If X[i] dominates u, keep
            # If non-dominated, keep both? No, standard DE keeps one.
            # For MO, we use an archive-based approach.
            
            # Update population
            if self._dominates(f_u, F[i]):
                X[i] = u
                F[i] = f_u
            
            # Update Archive
            # Add u to archive if it is non-dominated with respect to the archive
            # And remove any archive members dominated by u
            
            new_archive_X = []
            new_archive_F = []
            
            # Check if u is dominated by any in archive
            dominated_by_archive = False
            for af in archive_F:
                if self._dominates(af, f_u):
                    dominated_by_archive = True
                    break
            
            if not dominated_by_archive:
                # u is not dominated by archive. Add it and remove dominated archive members
                for j in range(len(archive_F)):
                    if not self._dominates(f_u, archive_F[j]):
                        new_archive_X.append(archive_X[j])
                        new_archive_F.append(archive_F[j])
                
                new_archive_X.append(u.copy())
                new_archive_F.append(f_u.copy())
            
            archive_X = new_archive_X
            archive_F = new_archive_F
            
            # Prune archive if too large
            if len(archive_X) > archive_size:
                # Keep the best 'archive_size' points by some criterion
                # Simple heuristic: keep points that are furthest apart (crowding distance)
                # Or just keep the first 'archive_size' non-dominated points
                # For efficiency, let's just truncate if it's a simple list, 
                # but we need to ensure we don't lose diversity.
                # Let's implement a simple crowding distance pruning
                A = np.array(archive_F)
                nd_idx = self._non_dominated_sort(A)
                
                if len(nd_idx) > archive_size:
                    # Compute crowding distance
                    cd = np.zeros(len(nd_idx))
                    for m in range(self.n_obj):
                        order = np.argsort(A[nd_idx, m])
                        cd[order[0]] = np.inf
                        cd[order[-1]] = np.inf
                        if A[nd_idx[order[-1]], m] - A[nd_idx[order[0]], m] > 0:
                            for k in range(1, len(nd_idx) - 1):
                                cd[order[k]] += (A[nd_idx[order[k+1]], m] - A[nd_idx[order[k-1]], m]) / (A[nd_idx[order[-1]], m] - A[nd_idx[order[0]], m])
                    
                    # Keep points with highest crowding distance
                    keep_idx = nd_idx[np.argsort(cd)[::-1][:archive_size]]
                    
                    new_ax = []
                    new_af = []
                    for idx in keep_idx:
                        # Find original index in archive_F
                        # This is inefficient, but archive_size is small
                        for j in range(len(archive_F)):
                            if np.allclose(archive_F[j], A[idx]):
                                new_ax.append(archive_X[j])
                                new_af.append(archive_F[j])
                                break
                    
                    archive_X = new_ax
                    archive_F = new_af

        # Final result: Return the archive points
        # The evaluator records every returned objective vector.
        # We return the archive points.
        
        # Ensure we return a valid array
        if len(archive_F) > 0:
            return np.array(archive_X)
        else:
            return X

# EVOLVE-BLOCK-END
