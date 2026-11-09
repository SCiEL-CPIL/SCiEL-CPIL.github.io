"""New optimizer: Multi-Objective Differential Evolution (MODE)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-Objective Differential Evolution (MODE) for RE34."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Parameters
        N = 20  # Population size
        F = 0.8  # Mutation scale
        CR = 0.9  # Crossover rate
        max_generations = (budget - N) // N
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((N, dim))
        F_vals = np.array([problem(x) for x in X], dtype=float)
        used = N
        
        # Initialize Archive
        archive_X = []
        archive_F = []
        for i in range(N):
            if self._is_non_dominated(F_vals[i], archive_F):
                # Check if it dominates any in archive
                archive_X, archive_F = self._update_archive(archive_X, archive_F, X[i], F_vals[i])
                
        # Main Loop
        for gen in range(max_generations):
            if used >= budget:
                break
                
            # Generate trial vectors
            for i in range(N):
                if used >= budget:
                    break
                
                # Select 3 distinct random indices
                indices = [j for j in range(N) if j != i]
                r1, r2, r3 = rng.choice(indices, size=3, replace=False)
                
                # Mutation: DE/rand/1
                v = X[r1] + F * (X[r2] - X[r3])
                v = np.clip(v, lo, hi)
                
                # Crossover: Binomial
                j_rand = rng.integers(dim)
                u = np.copy(X[i])
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        u[j] = v[j]
                
                u = np.clip(u, lo, hi)
                
                # Evaluate
                f_u = problem(u)
                used += 1
                
                # Selection
                f_xi = F_vals[i]
                
                if self._dominates(f_u, f_xi):
                    # u dominates x_i
                    X[i] = u
                    F_vals[i] = f_u
                elif not self._dominates(f_xi, f_u):
                    # Non-dominated
                    # Strategy: Keep x_i to maintain population stability, 
                    # but update archive with u if it's good.
                    # Alternatively, replace if u has better crowding distance?
                    # Let's stick to keeping x_i for population, but archive handles diversity.
                    pass
                # else: x_i dominates u, do nothing
                
                # Update Archive with u (and potentially X[i] if it changed)
                # It's safer to update archive with the best candidate
                if self._is_non_dominated(f_u, archive_F):
                    archive_X, archive_F = self._update_archive(archive_X, archive_F, u, f_u)
                elif self._is_non_dominated(F_vals[i], archive_F):
                    # If X[i] was replaced, it might be new and non-dominated
                    # But if X[i] was kept, it was already in archive or dominated.
                    # Actually, if X[i] was replaced by u, u is in archive.
                    # If X[i] was kept, it was already in archive from previous step?
                    # Not necessarily, it might have been dominated by something else added later.
                    # So we should check both.
                    archive_X, archive_F = self._update_archive(archive_X, archive_F, X[i], F_vals[i])
                    
            # Prune archive if it grows too large
            if len(archive_F) > 100:
                archive_X, archive_F = self._prune_archive(archive_X, archive_F)

        # Return a representative point (ignored by evaluator)
        return X[0]

    def _dominates(self, f1, f2):
        """Check if f1 dominates f2 (minimization)."""
        return np.all(f1 <= f2) and np.any(f1 < f2)

    def _is_non_dominated(self, f_new, archive_F_list):
        """Check if f_new is not dominated by any point in archive_F_list."""
        for f in archive_F_list:
            if self._dominates(f, f_new):
                return False
        return True

    def _update_archive(self, archive_X, archive_F, new_x, new_f):
        """Update archive with a new point, removing dominated points."""
        # Check if new point is dominated by any in archive
        for f in archive_F:
            if self._dominates(f, new_f):
                return archive_X, archive_F
        
        # Add new point
        archive_X.append(new_x)
        archive_F.append(new_f)
        
        # Remove points dominated by new point
        new_archive_X = []
        new_archive_F = []
        for i in range(len(archive_F)):
            if not self._dominates(new_f, archive_F[i]):
                new_archive_X.append(archive_X[i])
                new_archive_F.append(archive_F[i])
            # If new_f dominates archive_F[i], we drop archive_F[i]
            # If they are equal, we drop the old one (keep new one)
            # This is handled by the condition: if new_f dominates old, old is dropped.
            # If old dominates new, new was not added (returned earlier).
            # If non-dominated, both kept.
            
        return new_archive_X, new_archive_F

    def _prune_archive(self, archive_X, archive_F):
        """Prune archive using crowding distance if size > 100."""
        if len(archive_F) <= 100:
            return archive_X, archive_F
        
        F_arr = np.array(archive_F)
        X_arr = np.array(archive_X)
        cd = self._crowding_distance(F_arr)
        
        # Keep 100 points with highest crowding distance
        # We want to keep diverse points
        keep_idx = np.argsort(-cd)[:100]
        
        return [X_arr[i] for i in keep_idx], [F_arr[i] for i in keep_idx]

    def _crowding_distance(self, F_arr):
        """Compute crowding distance for a set of non-dominated points."""
        n = len(F_arr)
        if n <= 2:
            return np.full(n, np.inf)
        cd = np.zeros(n)
        n_obj = F_arr.shape[1]
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
        return cd
# EVOLVE-BLOCK-END
