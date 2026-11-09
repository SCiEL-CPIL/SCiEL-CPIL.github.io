"""Multi-objective optimizer for RE34: Hypervolume-guided multi-start local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-guided multi-start local search for RE34."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _exact_hv_3d(self, points, ref):
        """Compute exact 3D hypervolume for a set of points relative to ref.
        points: (N, 3) array
        ref: (3,) array
        Returns: float, the hypervolume.
        """
        # Filter points that are strictly better than ref in all objectives
        valid = np.all(points < ref, axis=1)
        pts = points[valid]
        if len(pts) == 0:
            return 0.0
        
        # Remove dominated points
        n = len(pts)
        if n == 0:
            return 0.0
        if n == 1:
            return float(np.prod(ref - pts[0]))
        
        # Sort by first objective
        order = np.argsort(pts[:, 0])
        pts = pts[order]
        
        # Compute non-dominated front and their contributions
        # In 3D, we can use a sweep-line approach or inclusion-exclusion
        # For simplicity with small N, use a grid-free method:
        # Sort by x, then for each point, compute the 2D hypervolume of the 
        # region it uniquely contributes.
        
        # Alternative: use the "divide and conquer" or "layered" approach
        # For 3 objectives, we can sort by obj0, then for each point in order,
        # compute the 2D area of the region in (obj1, obj2) space that is 
        # not covered by previously processed points.
        
        # Let's use a simpler approach for small N:
        # Sort by obj0 ascending
        pts_sorted = pts[order]
        
        # Compute HV using the standard algorithm:
        # HV = sum over non-dominated points of (delta_x * delta_y * delta_z)
        # where deltas are determined by the next point in each objective
        
        # Find non-dominated points
        nd_mask = np.zeros(n, dtype=bool)
        for i in range(n):
            dominated = False
            for j in range(n):
                if i != j:
                    if np.all(pts_sorted[j] <= pts_sorted[i]) and np.any(pts_sorted[j] < pts_sorted[i]):
                        dominated = True
                        break
            nd_mask[i] = not dominated
        
        nd_pts = pts_sorted[nd_mask]
        nd_idx = np.where(nd_mask)[0]
        
        if len(nd_pts) == 0:
            return 0.0
        if len(nd_pts) == 1:
            return float(np.prod(ref - nd_pts[0]))
        
        # For 3D, use the "slice" method:
        # Sort non-dominated points by obj0
        # For each point, the contribution is:
        # (next_x - x_i) * (contribution in 2D of the region above y_i and z_i)
        
        # Actually, let's use a simpler but exact method for 3D:
        # Iterate over all points, for each point compute the volume of the box
        # [point, ref] minus the union of boxes from other points that overlap.
        # This is O(N^2) but N is small.
        
        # Even simpler: use the fact that for 3D, we can project and use
        # a 2D HV computation at each slice.
        
        # Let me use a direct approach:
        # Sort by obj0
        order2 = np.argsort(nd_pts[:, 0])
        nd_pts = nd_pts[order2]
        
        # For each consecutive pair in obj0, the slice between them has a 
        # 2D hypervolume in (obj1, obj2).
        
        total_hv = 0.0
        n_nd = len(nd_pts)
        
        for i in range(n_nd):
            x_i = nd_pts[i, 0]
            x_next = nd_pts[i+1, 0] if i + 1 < n_nd else ref[0]
            
            # The 2D region for this slice: all points with obj0 <= x_next
            # that have obj1 < y_i or obj2 < z_i... 
            # Actually, the standard approach:
            # For point i, the contribution is:
            # (x_next - x_i) * HV_2D({(y_j, z_j) for j with x_j <= x_next and (y_j < y_i or z_j < z_i)}, (ref[1], ref[2]))
            # But this is complex. Let me use a different method.
            
            pass
        
        # Let me use a simpler O(N^3) method that works for small N:
        # For each point, compute the volume of the box it uniquely contributes.
        # A point i contributes the box [p_i, ref] minus the union of boxes [p_j, ref] 
        # for all j != i where p_j dominates part of that box.
        
        # Actually, the cleanest exact method for 3D with small N:
        # Use inclusion-exclusion over the non-dominated front.
        # HV = union volume of boxes [p_i, ref] for all non-dominated i.
        
        # For N <= 30, we can use a grid-free method:
        # Collect all unique coordinates in each dimension
        xs = np.unique(np.concatenate([nd_pts[:, 0], [ref[0]]]))
        ys = np.unique(np.concatenate([nd_pts[:, 1], [ref[1]]]))
        zs = np.unique(np.concatenate([nd_pts[:, 2], [ref[2]]]))
        
        # Create grid cells
        total = 0.0
        for i in range(len(xs) - 1):
            for j in range(len(ys) - 1):
                for k in range(len(zs) - 1):
                    # Cell: [xs[i], xs[i+1]] x [ys[j], ys[j+1]] x [zs[k], zs[k+1]]
                    # Check if this cell is covered by any point
                    # A point p covers this cell if p[0] <= xs[i] and p[1] <= ys[j] and p[2] <= zs[k]
                    covered = False
                    for p in nd_pts:
                        if p[0] <= xs[i] + 1e-15 and p[1] <= ys[j] + 1e-15 and p[2] <= zs[k] + 1e-15:
                            covered = True
                            break
                    if covered:
                        vol = (xs[i+1] - xs[i]) * (ys[j+1] - ys[j]) * (zs[k+1] - zs[k])
                        total += vol
        
        return total

    def _hv_contribution(self, point, front, ref):
        """Compute the hypervolume contribution of adding point to front."""
        old_hv = self._exact_hv_3d(front, ref)
        new_front = np.vstack([front, [point]])
        new_hv = self._exact_hv_3d(new_front, ref)
        return new_hv - old_hv

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([1705.0, 12.0, 0.3])
        
        dim = self.dim
        budget = self.budget

        # Phase 1: Deterministic initialization with stratified sampling
        n_init = 40
        # Use Halton sequence for well-spread initialization
        def halton(n, dim):
            points = np.zeros((n, dim))
            for d in range(dim):
                base = 2 + d
                for i in range(n):
                    x = 0.0
                    f = 1.0
                    idx = i + 1
                    while idx > 0:
                        f /= base
                        x += f * (idx % base)
                        idx //= base
                    points[i, d] = x
            return points

        X_init = lo + (hi - lo) * halton(n_init, dim)
        F_init = np.array([problem(x) for x in X_init])
        n_evals = n_init

        # Build initial front
        front = np.array([F_init[i] for i in range(len(F_init))])
        all_points = F_init.copy()

        # Phase 2: Local search from best points
        # Find the point with highest HV contribution (or just the best in each objective)
        # Sort by first objective to find diverse candidates
        order = np.argsort(F_init[:, 0])
        
        # Select diverse starting points for local search
        # Take the best in each objective and a few random ones
        start_indices = []
        for obj in range(3):
            start_indices.append(np.argmin(F_init[:, obj]))
        # Add a few more diverse points
        for idx in order[::max(1, len(order)//5)]:
            if idx not in start_indices:
                start_indices.append(idx)
            if len(start_indices) >= 8:
                break
        
        start_indices = list(set(start_indices))[:10]

        # Local search parameters
        n_local = 20  # iterations per start
        step_sizes = [0.1, 0.05, 0.02, 0.01]

        for si, start_idx in enumerate(start_indices):
            if n_evals >= budget:
                break
            if time.time() > deadline:
                break

            x = X_init[start_idx].copy()
            f_x = F_init[start_idx].copy()

            for it in range(n_local):
                if n_evals >= budget:
                    break
                if time.time() > deadline:
                    break

                # Try a random perturbation
                scale = step_sizes[min(it // 5, len(step_sizes)-1)]
                direction = np.random.randn(dim)
                direction /= (np.linalg.norm(direction) + 1e-10)
                step = scale * (hi - lo)
                x_new = x + direction * step
                x_new = np.clip(x_new, lo, hi)

                f_new = problem(x_new)
                n_evals += 1

                # Check if this improves the hypervolume
                hv_contrib = self._hv_contribution(f_new, front, ref)
                
                if hv_contrib > 1e-10:
                    # Accept: update front
                    front = np.vstack([front, f_new])
                    x = x_new
                    f_x = f_new
                else:
                    # Reject: keep x, but maybe still explore nearby
                    # Try a smaller step in a different direction
                    pass

                # Occasionally jump to a random point to escape local optima
                if it % 5 == 4 and n_evals < budget:
                    x_rand = lo + np.random.rand(dim) * (hi - lo)
                    f_rand = problem(x_rand)
                    n_evals += 1
                    hv_rand = self._hv_contribution(f_rand, front, ref)
                    if hv_rand > 1e-10:
                        front = np.vstack([front, f_rand])
                        x = x_rand
                        f_x = f_rand

        return front
# EVOLVE-BLOCK-END
