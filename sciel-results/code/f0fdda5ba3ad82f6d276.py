"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Lightweight evolutionary multi-objective optimizer for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _fast_non_dominated(self, F):
        """Return indices of non-dominated points (lightweight version)."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=int)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]:
                continue
            for j in range(n):
                if i == j or dominated[j]:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated[i] = True
                    break
        return np.where(~dominated)[0]

    def _hypervolume_3d(self, points, ref):
        """Compute exact 3D hypervolume for points strictly below ref in all dims."""
        # Filter points strictly better than reference in all objectives
        valid = np.all(points < ref, axis=1)
        pts = points[valid]
        if len(pts) == 0:
            return 0.0
        # Sort by first objective
        order = np.argsort(pts[:, 0])
        pts = pts[order]
        # Compute hypervolume using inclusion-exclusion in 3D
        # For each point, its contribution is the volume of the box it uniquely covers
        # Use the standard O(n^2) algorithm for 3D
        n = len(pts)
        hyp = 0.0
        # Sort by f0 ascending
        for i in range(n):
            p = pts[i]
            if p[0] >= ref[0] or p[1] >= ref[1] or p[2] >= ref[2]:
                continue
            # Volume of box from p to ref
            w0 = ref[0] - p[0]
            w1 = ref[1] - p[1]
            w2 = ref[2] - p[2]
            # Subtract overlap with other points that dominate in at least one dim
            # A point j dominates the box of i if p_j[k] <= p_i[k] for some k and
            # the boxes overlap. Simpler: use the standard sweep-line approach.
            # For 3D with small n, use the O(n^2) method:
            # Contribution of point i = volume of region where i is the "best" in f0
            # among points that cover that region.
            pass
        # Use a simpler exact method: enumerate all combinations
        # For 3D, hypervolume can be computed by decomposing into boxes.
        # Standard algorithm: sort by f0, then for each point compute the area in
        # the (f1,f2) plane not covered by any previously processed point.
        hyp = 0.0
        for i in range(n):
            p = pts[i]
            if p[0] >= ref[0]:
                break
            # For this point, compute the 2D area in (f1,f2) plane
            # that is not covered by any point j with j != i and p_j[0] <= p[i][0]
            # that also covers part of this region.
            # Actually, the standard approach: 
            # The hypervolume contribution of point i (sorted by f0 ascending) is:
            # (ref[0] - p[i][0]) * area_2d
            # where area_2d is the area of the rectangle [p[i][1], ref[1]] x [p[i][2], ref[2]]
            # minus the areas covered by other points that have f0 <= p[i][0] and
            # whose (f1,f2) boxes overlap.
            # This is complex. Let me use a different exact method.
            
            # For 3D with n <= ~50, we can use the following:
            # Hypervolume = sum over all subsets... no, too expensive.
            # Use the recursive decomposition:
            # Sort by f0. The hypervolume is:
            # (ref[0] - f0[0]) * H2D(pts[0:]) - ... 
            # Actually, let me just use a known correct O(n^2) for 3D.
            
            # The hypervolume can be computed as:
            # Sort points by f0 ascending. 
            # For each point i, its contribution to hypervolume is:
            #   (ref[0] - p_i[0]) * A_i
            # where A_i is the 2D area in (f1,f2) of the region [p_i[1], ref[1]] x [p_i[2], ref[2]]
            # that is NOT covered by any point j with j < i (i.e., p_j[0] <= p_i[0]) 
            # whose (f1,f2) box intersects this region.
            # Wait, that's not quite right either because points with larger f0 
            # don't cover regions for smaller f0 points.
            
            # Correct approach: 
            # The hypervolume is the volume of the union of boxes [p_i, ref] for all i.
            # Sort by f0 ascending. Process points in order.
            # Maintain a set of 2D rectangles (in f1,f2 space) representing 
            # the "uncovered" region at the current f0 level.
            # This is getting complex. Let me use a simpler but still exact method
            # for small n: enumerate all vertices of the arrangement.
            
            # For 3D with n points, the hypervolume can be computed by:
            # 1. Collect all unique f0, f1, f2 values from points and ref.
            # 2. This creates a grid. But that's O(n^3) cells.
            # 3. For each cell, check if it's covered by any point.
            # This is O(n^4) which is fine for n <= 50.
            
            pass
        
        # Let me use a cleaner exact O(n^2) algorithm for 3D hypervolume
        # Reference: "Fast Hypervolume Calculation for 3D" 
        # 
        # Algorithm:
        # 1. Sort points by f0 ascending.
        # 2. For each point i, compute the 2D hypervolume contribution in the 
        #    (f1, f2) plane, considering only points j with p_j[0] <= p_i[0].
        #    The 2D HV in (f1,f2) for a set of points is computed similarly.
        # 
        # Simpler: since n is small (<= 50), use the grid-based approach.
        
        # Grid-based exact hypervolume for 3D
        # Collect all unique coordinate values
        f0_vals = np.unique(np.concatenate([pts[:, 0], [ref[0]]]))
        f1_vals = np.unique(np.concatenate([pts[:, 1], [ref[1]]]))
        f2_vals = np.unique(np.concatenate([pts[:, 2], [ref[2]]]))
        
        # For each cell in the grid, check if it's dominated by at least one point
        # A cell [f0_i, f0_{i+1}] x [f1_j, f1_{j+1}] x [f2_k, f2_{k+1}]
        # is covered if there exists a point p such that 
        # p[0] <= f0_i AND p[1] <= f1_j AND p[2] <= f2_k
        # (since the point's box extends from p to ref, it covers all points 
        #  that are >= p in all dims, i.e., the cell is covered if the cell's 
        #  lower corner is >= p in all dims... wait, no.)
        
        # Actually, point p covers the box [p[0], ref[0]] x [p[1], ref[1]] x [p[2], ref[2]].
        # A grid cell with corners (a,b,c) and (d,e,f) where a<b, c<d... 
        # Let me redefine: cell indexed by (i,j,k) spans:
        #   f0 in [f0_vals[i], f0_vals[i+1]]
        #   f1 in [f1_vals[j], f1_vals[j+1]]
        #   f2 in [f2_vals[k], f2_vals[k+1]]
        # This cell is covered by point p if:
        #   p[0] <= f0_vals[i] AND p[1] <= f1_vals[j] AND p[2] <= f2_vals[k]
        # Because the point's box starts at p and goes to ref, so it covers 
        # all points (x,y,z) with x >= p[0], y >= p[1], z >= p[2].
        # The cell is entirely within the point's box if the cell's minimum 
        # corner (f0_vals[i], f1_vals[j], f2_vals[k]) is >= p in all dims.
        
        n0 = len(f0_vals) - 1
        n1 = len(f1_vals) - 1
        n2 = len(f2_vals) - 1
        
        total_hyp = 0.0
        for i in range(n0):
            for j in range(n1):
                for k in range(n2):
                    cell_min = np.array([f0_vals[i], f1_vals[j], f2_vals[k]])
                    # Check if any point covers this cell
                    covered = False
                    for p in pts:
                        if np.all(p <= cell_min):
                            covered = True
                            break
                    if covered:
                        vol = (f0_vals[i+1] - f0_vals[i]) * (f1_vals[j+1] - f1_vals[j]) * (f2_vals[k+1] - f2_vals[k])
                        total_hyp += vol
        
        return total_hyp

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random  # Use evaluator's seeded RNG

        # Reference point for hypervolume
        ref = np.array([3.0, 5.0, 7.0], dtype=np.float64)

        # Initial population
        pop_size = min(30, self.budget // 6)
        X = lo + (hi - lo) * rng.random_sample((pop_size, self.dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # Track best known non-dominated points
        all_F = F.copy()

        # Adaptive mutation scale: start large, shrink
        base_scale = 0.3
        min_scale = 0.02

        # Reserve budget for coordinate descent
        cd_budget = min(60, self.budget // 3)
        ea_budget = self.budget - cd_budget

        while used < ea_budget:
            if time.time() > deadline:
                break

            # Get non-dominated front
            nd_indices = self._fast_non_dominated(F)
            if len(nd_indices) == 0:
                nd_indices = np.array([np.argmin(np.sum(F, axis=1))])

            # Select parent from non-dominated front with some randomness
            if len(nd_indices) >= 2:
                i = nd_indices[rng.randint(len(nd_indices))]
                j = nd_indices[rng.randint(len(nd_indices))]
                if np.sum(F[i]) <= np.sum(F[j]):
                    parent_idx = i
                else:
                    parent_idx = j
            else:
                parent_idx = nd_indices[0]

            # Adaptive mutation: scale decreases as we use more budget
            progress = used / self.budget
            scale = base_scale * (1 - progress) + min_scale * progress

            # Gaussian mutation with adaptive scale
            mutation = scale * (hi - lo) * rng.standard_normal(self.dim)
            child = X[parent_idx] + mutation
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replace the worst individual in the population
            worst_idx = int(np.argmax(np.sum(F, axis=1)))
            X[worst_idx] = child
            F[worst_idx] = fc

            # Also maintain a global archive of non-dominated points
            all_F = np.vstack([all_F, fc.reshape(1, -1)])

        # Coordinate descent phase: refine non-dominated points
        if used < self.budget and time.time() < deadline:
            nd_indices = self._fast_non_dominated(F)
            if len(nd_indices) > 0:
                # Select up to 4 non-dominated points for coordinate descent
                n_cd = min(4, len(nd_indices), max(1, (self.budget - used) // 15))
                cd_points = nd_indices[:n_cd]
                
                for pi in cd_points:
                    if used >= self.budget or time.time() > deadline:
                        break
                    
                    # Coordinate descent with ternary search on each dimension
                    current_x = X[pi].copy()
                    current_f = F[pi].copy()
                    
                    # Get current non-dominated set for HV calculation
                    nd_idx = self._fast_non_dominated(F)
                    if len(nd_idx) == 0:
                        nd_idx = np.array([pi])
                    
                    # For speed, limit the set used for HV to at most 15 points
                    if len(nd_idx) > 15:
                        sums = np.sum(F[nd_idx], axis=1)
                        top_idx = np.argsort(sums)[:15]
                        nd_idx = nd_idx[top_idx]
                    
                    hv_set_F = F[nd_idx].copy()
                    pi_in_nd = np.where(nd_idx == pi)[0]
                    if len(pi_in_nd) > 0:
                        hv_set_F[pi_in_nd[0]] = current_f
                    else:
                        hv_set_F = np.vstack([hv_set_F, current_f.reshape(1, -1)])
                    
                    # Compute initial HV
                    hv_before = self._hypervolume_3d(hv_set_F, ref)
                    
                    # Coordinate descent: for each dimension, do ternary search
                    for dim in range(self.dim):
                        if used >= self.budget or time.time() > deadline:
                            break
                        
                        # Ternary search on dimension dim
                        a = lo[dim]
                        b = hi[dim]
                        
                        # Do 2 iterations of ternary search (4 evaluations)
                        for _ in range(2):
                            if used >= self.budget or time.time() > deadline:
                                break
                            m1 = a + (b - a) / 3
                            m2 = b - (b - a) / 3
                            
                            # Evaluate m1
                            new_x1 = current_x.copy()
                            new_x1[dim] = m1
                            new_x1 = np.clip(new_x1, lo, hi)
                            new_f1 = problem(new_x1)
                            used += 1
                            if len(pi_in_nd) > 0:
                                hv_set_F[pi_in_nd[0]] = new_f1
                            else:
                                hv_set_F = np.vstack([hv_set_F, new_f1.reshape(1, -1)])
                            hv1 = self._hypervolume_3d(hv_set_F, ref)
                            
                            # Evaluate m2
                            new_x2 = current_x.copy()
                            new_x2[dim] = m2
                            new_x2 = np.clip(new_x2, lo, hi)
                            new_f2 = problem(new_x2)
                            used += 1
                            if len(pi_in_nd) > 0:
                                hv_set_F[pi_in_nd[0]] = new_f2
                            else:
                                hv_set_F = np.vstack([hv_set_F, new_f2.reshape(1, -1)])
                            hv2 = self._hypervolume_3d(hv_set_F, ref)
                            
                            if hv1 < hv2:
                                b = m2
                            elif hv1 > hv2:
                                a = m1
                            else:
                                a = m1
                                b = m2
                        
                        # Final evaluation at midpoint
                        if used >= self.budget or time.time() > deadline:
                            break
                        mid = (a + b) / 2
                        new_x_mid = current_x.copy()
                        new_x_mid[dim] = mid
                        new_x_mid = np.clip(new_x_mid, lo, hi)
                        new_f_mid = problem(new_x_mid)
                        used += 1
                        if len(pi_in_nd) > 0:
                            hv_set_F[pi_in_nd[0]] = new_f_mid
                        else:
                            hv_set_F = np.vstack([hv_set_F, new_f_mid.reshape(1, -1)])
                        hv_mid = self._hypervolume_3d(hv_set_F, ref)
                        
                        # Update if improved
                        if hv_mid > hv_before:
                            current_x = new_x_mid
                            current_f = new_f_mid
                            X[pi] = current_x
                            F[pi] = current_f
                            all_F = np.vstack([all_F, current_f.reshape(1, -1)])
                            if len(pi_in_nd) > 0:
                                hv_set_F[pi_in_nd[0]] = current_f
                            else:
                                hv_set_F = np.vstack([hv_set_F, current_f.reshape(1, -1)])
                            hv_before = hv_mid

        # Final local polish with small steps
        if used < self.budget and time.time() < deadline:
            polish_budget = min(self.budget - used, 10)
            nd_indices = self._fast_non_dominated(F)
            if len(nd_indices) > 0:
                n_polish = min(2, len(nd_indices), polish_budget)
                polish_indices = nd_indices[:n_polish]
                for pi in polish_indices:
                    if used >= self.budget or time.time() > deadline:
                        break
                    for _ in range(1):
                        if used >= self.budget or time.time() > deadline:
                            break
                        small_scale = 0.01
                        mutation = small_scale * (hi - lo) * rng.standard_normal(self.dim)
                        child = X[pi] + mutation
                        child = np.clip(child, lo, hi)
                        fc = problem(child)
                        used += 1
                        if np.all(fc <= F[pi]):
                            X[pi] = child
                            F[pi] = fc
                            all_F = np.vstack([all_F, fc.reshape(1, -1)])

        return X
# EVOLVE-BLOCK-END
