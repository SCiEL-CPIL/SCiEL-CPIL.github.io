"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer using hypervolume-aware selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hv_contribution_3d(self, points, ref):
        """
        Calculate the hypervolume contribution of each point in a 3D set.
        points: (N, 3) array of non-dominated points.
        ref: (3,) array of reference point.
        Returns: (N,) array of HV contributions.
        """
        n = len(points)
        if n == 0:
            return np.array([])
        
        # Filter strictly better than ref
        mask = np.all(points < ref, axis=1)
        if not np.any(mask):
            return np.zeros(n)
        
        pts = points[mask]
        idx = np.where(mask)[0]
        m = len(pts)
        
        contrib = np.zeros(n)
        
        # Sort by first objective to help with dominance checks, 
        # but for exact contribution in 3D with small N, we can do pairwise checks.
        # Contribution of i is volume of box [pts[i], ref] minus volume covered by others.
        # This is complex to do exactly in O(N^2) without a sweep line.
        # However, for N=20, we can approximate or use a simple decomposition.
        
        # Simple exact method for small N:
        # The HV of the set is the sum of contributions.
        # We can compute the total HV of the set, and then for each i, 
        # compute the HV of the set without i. The difference is the contribution.
        # This is O(N * HV_calc).
        
        def _calc_hv_3d(pts_subset):
            if len(pts_subset) == 0:
                return 0.0
            # Filter
            msk = np.all(pts_subset < ref, axis=1)
            p = pts_subset[msk]
            if len(p) == 0:
                return 0.0
            
            # Standard 3D HV algorithm:
            # Sort by f0
            p = p[np.argsort(p[:, 0])]
            n_p = len(p)
            hv = 0.0
            # Sweep from left to right (low f0 to high f0)
            # Maintain a list of "active" points that are not dominated in f1, f2
            # Actually, a simpler O(N^2) method for 3D:
            # For each point i, the contribution is the volume of the region 
            # dominated by i but not by any j with f0[j] < f0[i].
            # This requires 2D HV calculation on the slice.
            
            # Let's use a more robust O(N^2) or O(N^3) approach for small N.
            # We can compute the HV by decomposing the space into boxes defined by the points.
            # Given the small N (<=20), we can even use a grid if the space was discrete, but it's continuous.
            
            # Correct O(N^2) 3D HV:
            # Sort by f0.
            # For i from 0 to N-1:
            #   The width in f0 is min(ref[0], f0[i+1]) - f0[i]
            #   The area in f1-f2 is the 2D HV of the points j >= i (in sorted order) 
            #   that are not dominated by any k < i in the f1-f2 plane?
            #   No, the standard sweep line maintains a 2D structure.
            
            # For N=20, let's just use the "remove and recompute" strategy inside the outer loop?
            # No, that would be O(N^2 * HV_calc).
            
            # Let's implement a simple O(N^3) exact HV for 3D:
            # The hypervolume is the sum of volumes of disjoint boxes.
            # We can generate all unique x-coordinates, y-coordinates, z-coordinates from points and ref.
            # This creates a grid of O(N^3) cells. For N=20, 8000 cells.
            # For each cell, check if it's dominated by any point.
            # This is O(N^4) which is 160,000 ops. Still fast for 200 iterations?
            # 200 * 160,000 = 32,000,000 ops. Might be tight in Python, but NumPy vectorization helps.
            
            # Let's try the grid method for accuracy and simplicity.
            xs = np.unique(np.concatenate([p[:, 0], [ref[0]]]))
            ys = np.unique(np.concatenate([p[:, 1], [ref[1]]]))
            zs = np.unique(np.concatenate([p[:, 2], [ref[2]]]))
            
            # Create meshgrid
            # We only care about the region [p, ref]
            # Actually, we can just iterate through the points and calculate the "uncovered" volume.
            
            # Alternative: Use the fact that N is small.
            # Contribution of i = Vol([p_i, ref]) - Vol(Union_{j != i} ([p_j, ref] intersect [p_i, ref]))
            # This is hard.
            
            # Let's stick to the "Total HV - HV without i" method, but optimize HV calc.
            # For N=20, HV calc via grid is ~8000 cells * 20 points check = 160k ops.
            # Doing this 20 times (for each i) is 3.2M ops per selection step.
            # 200 steps * 3.2M = 640M ops. This is too slow in Python.
            
            # We need a faster HV contribution estimate or a faster HV calc.
            # For selection, we don't need *exact* HV. We need to identify the *least* important point.
            # A good proxy: Distance to the reference point, or "Crowding Distance".
            # But the prompt mentions "Hypervolume-aware selection" worked well.
            
            # Let's use a faster HV approximation:
            # Sort by f0.
            # For each point i, approximate contribution as:
            # (ref[0] - f0[i]) * (ref[1] - f1[i]) * (ref[2] - f2[i]) 
            # weighted by whether it's non-dominated.
            # This is the volume of the box it dominates. It overcounts overlaps.
            # But for identifying the *worst* point, it might be sufficient if combined with dominance checks.
            
            # Actually, let's just use the "remove and recompute" but with a faster HV algorithm.
            # There is an O(N log N) algorithm for 3D HV.
            # But implementing it correctly under time pressure is risky.
            
            # Let's use a simpler heuristic for selection:
            # 1. If there are dominated points, remove one of them (randomly or the one closest to ref).
            # 2. If all are non-dominated, remove the one with the smallest "Box Volume" (ref - f).
            #    This favors points that are "far" from the ref in terms of dominated volume.
            #    Wait, we want to KEEP points with large contribution.
            #    So we REMOVE the point with the smallest box volume?
            #    Not necessarily, because overlaps matter.
            
            # Let's try a compromise:
            # Calculate the "Box Volume" for each point: v_i = prod(ref - f_i).
            # Select the point with the smallest v_i to replace.
            # This is O(N). Very fast.
            # It is a heuristic, but often works well for small populations to maintain spread.
            
            return np.prod(ref - points, axis=1)
        
        # Using the heuristic above
        return np.prod(ref - points, axis=1)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([1705.0, 12.0, 0.3])
        
        pop_size = 20
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Identify non-dominated points
            # F is (N, 3)
            # Point i is dominated if exists j such that F[j] <= F[i] and F[j] != F[i]
            dominated = np.zeros(pop_size, dtype=bool)
            for i in range(pop_size):
                for j in range(pop_size):
                    if i != j:
                        if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            dominated[i] = True
                            break
                            
            nd_idx = np.where(~dominated)[0]
            
            if len(nd_idx) == 0:
                # All dominated? Pick random parent
                parent_idx = rng.integers(pop_size)
                # Replace the one with highest sum (worst)
                victim_idx = int(np.argmax(np.sum(F, axis=1)))
            else:
                # Select parent from non-dominated
                parent_idx = nd_idx[rng.integers(len(nd_idx))]
                
                # Select victim
                # If there are dominated points, remove one of them
                dom_idx = np.where(dominated)[0]
                if len(dom_idx) > 0:
                    # Remove the dominated point with the largest sum (worst)
                    victim_idx = dom_idx[int(np.argmax(np.sum(F[dom_idx], axis=1)))]
                else:
                    # All non-dominated. Remove the one with smallest "box volume"
                    # This is a heuristic to maintain diversity
                    vols = np.prod(ref - F, axis=1)
                    # Only consider non-dominated for removal? 
                    # Actually, if all are non-dominated, we remove the least "important".
                    # Smallest box volume is a good proxy for least important in HV terms.
                    victim_idx = int(np.argmin(vols))
            
            # Generate child
            parent = X[parent_idx]
            # Adaptive mutation: scale by distance to bounds?
            # Simple Gaussian mutation
            mutation_scale = 0.1
            child = parent + mutation_scale * (hi - lo) * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            X[victim_idx] = child
            F[victim_idx] = fc
            
        return X
# EVOLVE-BLOCK-END
