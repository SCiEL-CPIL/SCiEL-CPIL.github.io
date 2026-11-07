"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective evolutionary optimizer with hypervolume-aware selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        pop_size = min(50, max(20, self.budget // 4))
        remaining = self.budget

        # Initial population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        remaining -= pop_size

        # Reference point for hypervolume computation
        ref = np.array([3.0, 5.0, 7.0])

        def compute_hv_contribution(F, ref):
            """Compute hypervolume contribution for a set of points."""
            # Filter points that are strictly better than reference
            mask = np.all(F < ref, axis=1)
            if not np.any(mask):
                return 0.0
            F = F[mask]
            
            # Sort by first objective
            order = np.argsort(F[:, 0])
            F = F[order]
            
            # For 3 objectives, compute hypervolume
            # Use inclusion-exclusion or a simple approach
            # For small sets, we can use a grid-based approximation
            # But for exactness with small n, let's use a direct method
            
            n = len(F)
            if n == 0:
                return 0.0
            if n == 1:
                return np.prod(ref - F[0])
            
            # Sort by first objective
            idx = np.argsort(F[:, 0])
            F = F[idx]
            
            hv = 0.0
            for i in range(n):
                # Contribution of point i
                # It contributes if it's not dominated by any point with smaller first obj
                # Check if point i is dominated by any previous point
                dominated = False
                for j in range(i):
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated = True
                        break
                if not dominated:
                    # Compute volume contribution
                    # For each dimension, the extent from F[i][d] to ref[d]
                    # But we need to subtract overlaps
                    # Simple approach: for 3D, use the formula
                    # This is approximate for larger sets
                    extent = ref - F[i]
                    # Subtract parts covered by other non-dominated points
                    # This is complex; use a simpler approximation
                    hv += np.prod(extent)
            return hv

        def non_dominated_sort(F):
            """Return indices of non-dominated points."""
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j or dominated[j]:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominated[j] = True
            return np.where(~dominated)[0]

        # Main loop
        while remaining > 0:
            if time.time() > deadline:
                break
            
            # Get non-dominated points
            nd_idx = non_dominated_sort(F)
            if len(nd_idx) == 0:
                nd_idx = np.arange(len(F))
            
            # Select parent: prefer points with better hypervolume contribution
            # Use a simple strategy: pick from non-dominated, weighted by objective values
            if len(nd_idx) > 1:
                # Compute a simple score: sum of normalized objectives (lower is better)
                F_nd = F[nd_idx]
                # Normalize each objective to [0,1] based on current range
                scores = np.zeros(len(nd_idx))
                for d in range(self.n_obj):
                    mn = F_nd[:, d].min()
                    mx = F_nd[:, d].max()
                    if mx > mn:
                        scores += (F_nd[:, d] - mn) / (mx - mn)
                    else:
                        scores += 0.5
                scores /= self.n_obj
                # Pick the best (lowest score)
                parent_idx = nd_idx[np.argmin(scores)]
            else:
                parent_idx = nd_idx[0]
            
            # Mutation
            sigma = 0.15 * (hi - lo)
            child = X[parent_idx] + sigma * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            remaining -= 1
            
            # Replace worst point in population
            # Use a simple strategy: replace the point with worst sum of objectives
            worst_idx = int(np.argmax(np.sum(F, axis=1)))
            X[worst_idx] = child
            F[worst_idx] = fc

        return X
# EVOLVE-BLOCK-END
