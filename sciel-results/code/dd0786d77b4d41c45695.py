# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE34 using adaptive mutation with non-dominated sorting."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _non_dominated_mask(self, F):
        """Return a boolean mask of non-dominated points in F (n x n_obj array)."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=bool)
        
        # Vectorized dominance check
        diff = F[:, np.newaxis, :] - F[np.newaxis, :, :]  # shape (n, n, n_obj)
        leq = np.all(diff <= 0, axis=2)  # shape (n, n)
        lt = np.any(diff < 0, axis=2)  # shape (n, n)
        dominates = leq & lt  # shape (n, n)
        dominated = np.any(dominates, axis=0)  # shape (n,)
        return ~dominated

    def _crowding_distance(self, F, nd_mask):
        """Compute crowding distance for non-dominated points."""
        n = len(F)
        cd = np.zeros(n)
        nd_idx = np.where(nd_mask)[0]
        if len(nd_idx) <= 2:
            cd[nd_idx] = np.inf
            return cd
        
        for k in range(self.n_obj):
            sorted_idx = nd_idx[np.argsort(F[nd_idx, k])]
            if sorted_idx.size < 2:
                continue
            cd[sorted_idx[0]] = np.inf
            cd[sorted_idx[-1]] = np.inf
            for i in range(1, sorted_idx.size - 1):
                if F[sorted_idx[i+1], k] - F[sorted_idx[i-1], k] > 1e-10:
                    cd[sorted_idx[i]] += (F[sorted_idx[i+1], k] - F[sorted_idx[i-1], k]) / (F[sorted_idx[i+1], k] - F[sorted_idx[i-1], k] + 1e-10)
        
        return cd

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # 285s safety margin before 300s limit
        
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_obj = self.n_obj
        dim = self.dim
        
        # Initial population using Latin Hypercube for better spread
        pop_size = min(40, self.budget // 4)
        X = np.zeros((pop_size, dim))
        for d in range(dim):
            perm = rng.permutation(pop_size)
            X[:, d] = lo[d] + (hi[d] - lo[d]) * (perm + 0.5) / pop_size
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Track all evaluated points for hypervolume
        all_F = F.copy()
        
        # Precompute some constants
        range_vec = hi - lo
        
        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Find non-dominated points
            nd_mask = self._non_dominated_mask(F)
            nd_idx = np.where(nd_mask)[0]
            
            if len(nd_idx) == 0:
                # Fallback: use best single point
                nd_idx = np.array([np.argmin(np.sum(F, axis=1))])
            
            # Compute crowding distance for diversity
            cd = self._crowding_distance(F, nd_mask)
            
            # Adaptive mutation scale: start larger, shrink over time
            progress = used / self.budget
            mut_scale = 0.25 * (1 - progress) ** 0.7 + 0.03
            
            # Generate children from non-dominated parents using tournament selection
            n_children = min(8, self.budget - used)
            for _ in range(n_children):
                if time.time() > deadline:
                    break
                if used >= self.budget:
                    break
                    
                # Tournament selection from non-dominated set
                if len(nd_idx) > 1:
                    # Use crowding distance for diversity-aware selection
                    tournament_size = min(3, len(nd_idx))
                    candidates = nd_idx[rng.integers(len(nd_idx), size=tournament_size)]
                    # Select candidate with highest crowding distance
                    parent_idx = candidates[np.argmax(cd[candidates])]
                else:
                    parent_idx = nd_idx[0]
                
                parent = X[parent_idx]
                
                # Mix of Gaussian (local) and uniform (global) mutation
                if rng.random() < 0.7:
                    # Local Gaussian mutation
                    child = parent + mut_scale * range_vec * rng.standard_normal(dim)
                else:
                    # Global uniform mutation (exploration)
                    child = lo + range_vec * rng.random(dim)
                
                child = np.clip(child, lo, hi)
                
                fc = problem(child)
                used += 1
                
                # Replace worst point in population based on quality + diversity
                # Find dominated points first
                dominated_mask = ~nd_mask
                if np.any(dominated_mask):
                    # Among dominated, replace the worst by normalized sum
                    dom_idx = np.where(dominated_mask)[0]
                    F_min = F.min(axis=0)
                    F_max = F.max(axis=0)
                    F_range = F_max - F_min + 1e-10
                    F_norm = (F - F_min) / F_range
                    quality = np.sum(F_norm, axis=1)
                    worst_idx = dom_idx[np.argmax(quality[dom_idx])]
                else:
                    # All non-dominated: replace lowest crowding distance
                    worst_idx = nd_idx[np.argmin(cd[nd_idx])]
                
                X[worst_idx] = child
                F[worst_idx] = fc
                
                # Update all_F for hypervolume tracking
                all_F = np.vstack([all_F, fc])
        
        return X
# EVOLVE-BLOCK-END
