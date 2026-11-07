# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for WFG7 using hypervolume-driven search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _calc_hv_3d(self, points, ref):
        """
        Calculate the exact hypervolume of a set of points in 3D.
        points: Nx3 array
        ref: 3-array
        """
        if len(points) == 0:
            return 0.0
        
        # Filter points that are better than ref in all objectives
        mask = np.all(points < ref, axis=1)
        pts = points[mask]
        
        if len(pts) == 0:
            return 0.0
            
        # Sort by first objective (f0)
        idx = np.argsort(pts[:, 0])
        pts = pts[idx]
        
        # We integrate the area of the union of projections in (f1, f2)
        # over the range of f0.
        # The active set changes at each unique f0 value.
        # Let f0_values be the sorted unique f0s.
        # For an interval [z_j, z_{j+1}), the active points are those with f0 <= z_j.
        # Actually, if we sort pts, pts[0] has min f0.
        # Let's iterate through the points in sorted order.
        # Let current_f0 be the f0 of the current point being added.
        # The volume contributed by the "slab" from prev_f0 to current_f0 is:
        # (current_f0 - prev_f0) * Area(Union of projections of points added so far).
        # Wait, the box for point i is [f_i0, ref0].
        # So for z in [f_i0, f_{i+1}0), the active points are those with f0 <= z.
        # Since f0 is sorted, for z in [f_k0, f_{k+1}0), the active points are pts[0..k].
        # The volume is sum_{k=0}^{N-2} (f_{k+1}0 - f_k0) * Area(Union(pts[0..k])) 
        # + (ref0 - f_{N-1}0) * Area(Union(pts[0..N-1])).
        # Note: The "Area" is the 2D hypervolume (area of union of rectangles) in the (f1, f2) plane
        # with reference (ref1, ref2).
        
        def area_2d(points_2d, ref_2d):
            # points_2d: Mx2 array (f1, f2)
            # ref_2d: 2-array (ref1, ref2)
            # Calculate area of union of rectangles [p1, ref1] x [p2, ref2]
            if len(points_2d) == 0:
                return 0.0
            
            # Sort by f1
            idx2 = np.argsort(points_2d[:, 0])
            p2 = points_2d[idx2]
            
            # Sweep line over f1.
            # Similar logic:
            # Sum over intervals [p_j1, p_{j+1}1) of (p_{j+1}1 - p_j1) * Height(Union of p_k2 for k<=j).
            # Height is the length of the union of intervals [p_k2, ref2].
            
            total_area = 0.0
            active_intervals = [] # List of (start, end) i.e., (p2, ref2)
            
            # We need to efficiently compute the union length of intervals.
            # Since M is small, we can just collect all active intervals and compute union length each step.
            
            for i in range(len(p2)):
                # Add current point's interval
                active_intervals.append((p2[i][1], ref_2d[1]))
                
                # If it's the last point, we integrate up to ref1
                if i == len(p2) - 1:
                    width = ref_2d[0] - p2[i][0]
                else:
                    width = p2[i+1][0] - p2[i][0]
                
                # Calculate union length of active_intervals
                if not active_intervals:
                    h = 0.0
                else:
                    # Sort by start
                    active_intervals.sort(key=lambda x: x[0])
                    merged = []
                    for interval in active_intervals:
                        if not merged:
                            merged.append(list(interval))
                        else:
                            last = merged[-1]
                            if interval[0] <= last[1]:
                                last[1] = max(last[1], interval[1])
                            else:
                                merged.append(list(interval))
                    h = sum(end - start for start, end in merged)
                
                total_area += width * h
                
            return total_area

        total_hv = 0.0
        # We need to be careful with the "prev_f0".
        # The first point has f0 = pts[0][0]. The box starts at pts[0][0].
        # There is no volume below pts[0][0] because no point dominates there.
        # So we start integrating from pts[0][0].
        
        # However, the formula above assumes we add points one by one.
        # Let's trace:
        # k=0: Add pts[0]. Active: {0}.
        #   Interval for f0: [pts[0][0], pts[1][0]).
        #   Area is computed from pts[0].
        #   Vol += (pts[1][0] - pts[0][0]) * Area(pts[0])
        # k=1: Add pts[1]. Active: {0, 1}.
        #   Interval for f0: [pts[1][0], pts[2][0]).
        #   Vol += (pts[2][0] - pts[1][0]) * Area(pts[0], pts[1])
        # ...
        # k=N-1: Add pts[N-1]. Active: {0..N-1}.
        #   Interval for f0: [pts[N-1][0], ref0).
        #   Vol += (ref0 - pts[N-1][0]) * Area(pts[0..N-1])
        
        ref_2d = ref[1:]
        
        for i in range(len(pts)):
            # Get projections of pts[0..i]
            proj = pts[0:i+1, 1:]
            a = area_2d(proj, ref_2d)
            
            if i == len(pts) - 1:
                width = ref[0] - pts[i][0]
            else:
                width = pts[i+1][0] - pts[i][0]
            
            total_hv += width * a
            
        return total_hv

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        width = hi - lo

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Reference point for hypervolume (scoring reference)
        ref = np.array([3.0, 5.0, 7.0])

        # --- Initial population ---
        pop_size = 20
        X = lo + width * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # --- Main search loop ---
        mut_scale = 0.2

        while used < budget:
            if time.time() > deadline - 2.0:
                break

            # Select parent: Random from population
            parent_idx = np.random.randint(0, len(X))
            parent = X[parent_idx].copy()

            # Mutation
            if np.random.random() < 0.1:
                # Large jump
                child = parent + width * np.random.standard_normal(dim) * 0.5
            else:
                child = parent + width * np.random.standard_normal(dim) * mut_scale
            
            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1

            # Add to population
            X = np.vstack([X, child])
            F = np.vstack([F, fc])

            # If population exceeds size, remove worst point based on HV contribution
            if len(X) > pop_size:
                # Calculate total HV
                hv_total = self._calc_hv_3d(F, ref)
                
                min_contrib = float('inf')
                min_idx = -1
                
                for i in range(len(X)):
                    # Create F without i
                    mask = np.ones(len(X), dtype=bool)
                    mask[i] = False
                    F_without_i = F[mask]
                    hv_without_i = self._calc_hv_3d(F_without_i, ref)
                    contrib = hv_total - hv_without_i
                    
                    if contrib < min_contrib:
                        min_contrib = contrib
                        min_idx = i
                
                # Remove min_idx
                mask = np.ones(len(X), dtype=bool)
                mask[min_idx] = False
                X = X[mask]
                F = F[mask]

            # Adapt mutation scale
            if used % 20 == 0:
                mut_scale *= 0.95
                mut_scale = max(mut_scale, 0.02)

        return X
# EVOLVE-BLOCK-END
