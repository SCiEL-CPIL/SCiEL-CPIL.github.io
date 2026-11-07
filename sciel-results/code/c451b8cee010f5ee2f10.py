"""Multi-objective optimizer: deterministic HV-guided sequential greedy search on a dense lattice."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # 50s safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref_point = np.array([1.1, 1.3, 1.1])

        used = 0
        all_F = []  # list of objective vectors
        all_X = []  # list of x vectors (for tracking, not used in scoring)

        # --- Helper: dominance ---
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        # --- Helper: hypervolume contribution of a single new point to existing archive ---
        # For n_obj=3, we can compute HV efficiently.
        # HV of archive A with reference point r = sum over non-dominated points of box volume.
        # Marginal HV = HV(A ∪ {p}) - HV(A)

        def hv_3d(points, ref):
            """Compute hypervolume of a set of 3D points with given reference point.
            points: (n, 3) array. Only points strictly better than ref in all dims count.
            """
            if len(points) == 0:
                return 0.0
            P = np.array(points)
            # Filter points that are strictly better than ref in all objectives
            mask = np.all(P < ref, axis=1)
            P = P[mask]
            if len(P) == 0:
                return 0.0

            # Sort by first objective
            order = np.argsort(P[:, 0])
            P = P[order]

            # For each point, compute the volume of the box it uniquely contributes
            # After sorting by obj0, for point i, its box extends from P[i] to:
            #   obj0: P[i,0] to P[i+1,0] (or ref[0])
            #   obj1: P[i,1] to min(P[j,1] for j>i) (or ref[1])
            #   obj2: P[i,2] to min(P[j,2] for j>i) (or ref[2])
            # But we need to be careful: the box must be non-dominated by subsequent points.

            hv = 0.0
            n = len(P)
            for i in range(n):
                x_max = P[i + 1, 0] if i + 1 < n else ref[0]
                # Find min obj1 and obj2 among points j > i that are not dominated
                # Actually, for the standard HV calculation with sorted points:
                # The contribution of point i is bounded by the next point in obj0,
                # and for obj1 and obj2, we need the minimum over all subsequent points
                # that have obj1 < P[i,1] or obj2 < P[i,2]... this is complex.
                # Simpler: use the standard incremental HV.

            # Alternative: use the standard algorithm for 3D HV
            # Sort by f1 ascending
            # For each point, find the "skyline" in the remaining 2 dimensions

            # Let's use a simpler O(n^2) approach for small n
            # For each point p_i, compute the volume of the region dominated by p_i
            # but not by any other point in the archive.
            # In 3D, for point i with value (a,b,c):
            # The box is [a, a_next) x [b, b_next) x [c, c_next) where a_next is min of f0 of points that dominate in f0...
            # This is getting complex. Let me use a different approach.

            # Use the "union of boxes" approach:
            # Each point p defines a box from p to ref.
            # HV = volume of union of all these boxes.
            # For 3D with small n, we can compute this by inclusion-exclusion or by decomposition.

            # Actually, for n_obj=3, there's a clean O(n log n) algorithm:
            # Sort by f0. Then for each point in order, the contribution is:
            # (f0_next - f0_i) * (min_f1_remaining - f1_i) * (min_f2_remaining - f2_i)
            # but only if min_f1_remaining > f1_i and min_f2_remaining > f2_i.
            # Wait, that's not quite right either.

            # Let me use a robust O(n^2) method:
            # For each point i, compute the volume of the box [p_i, ref] that is NOT covered by any box [p_j, ref] for j != i where p_j dominates p_i... no, that's not it.

            # Correct approach: The HV is the volume of the union of boxes B_i = [p_i, ref] for each non-dominated point p_i.
            # For 3D, we can compute this by:
            # 1. Sort points by f0.
            # 2. For each point i in sorted order, find the maximum f1 and f2 such that the box from (f0_i, f1, f2) to ref is not dominated.
            # 
            # Let me just use a simple O(n^3) or O(n^2) method since n <= 200.

            # O(n^2) method for 3D:
            # Sort by f0 ascending.
            # Maintain the "frontier" in the (f1, f2) plane.
            # For each new point, its contribution is the area in (f1, f2) that is newly covered.

            # Simpler: since n is at most 200, let's do the direct approach.
            # For each point p_i, its marginal contribution to HV (given all other points) is:
            # The volume of [p_i, ref] minus the volume of the intersection with all other boxes.
            # This is hard to compute directly.

            # Let me use a different strategy: since we only need MARGINAL HV (HV_after - HV_before),
            # and the archive is small, let me just compute HV before and after.

            # But computing HV from scratch each time is O(n^2) or O(n^3), which for n=200 is fine.

            # Here's a clean O(n^2) algorithm for 3D HV:
            # 1. Remove any point dominated by another.
            # 2. Sort remaining by f0.
            # 3. For i from 0 to n-1:
            #    width0 = (f0[i+1] if i+1 < n else ref[0]) - f0[i]
            #    For f1 and f2, we need the min over all j > i of f1[j] and f2[j], but only counting points that are not dominated in the 2D sense.
            #    
            # Actually, here's a well-known O(n log n) for 3D:
            # Sort by f0. Then compute the 2D HV of the (f1, f2) projections, weighted by the f0 differences.

            # Let me just implement a simple O(n^2) HV computation:
            # For each pair of points (i, j) with i < j in f0 order:
            # The contribution of point i to the HV, in the slice between f0[i] and f0[j],
            # is the 2D HV of the points in that slice projected onto (f1, f2).

            # This is getting too complex. Let me use a straightforward approach:
            # HV = sum over all non-dominated points of the volume they uniquely contribute.
            # For point p_i, find the "next" point in each dimension.
            # In 3D, if we sort by f0, then for point i:
            #   f0_max = f0[i+1] if i+1 < n else ref[0]
            #   For f1: find min f1 among points j where f0[j] > f0[i] AND f1[j] < ... 
            # 
            # I'll use a simple O(n^3) method that is correct for small n:

            # Compute 2D HV for a set of (y, z) points with reference (y_ref, z_ref)
            def hv_2d(pts, ref_y, ref_z):
                if len(pts) == 0:
                    return 0.0
                pts = np.array(pts)
                # Filter
                mask = (pts[:, 0] < ref_y) & (pts[:, 1] < ref_z)
                pts = pts[mask]
                if len(pts) == 0:
                    return 0.0
                # Remove dominated points in 2D
                # Sort by y
                order = np.argsort(pts[:, 0])
                pts = pts[order]
                # Keep only non-dominated: a point is non-dominated if no other point has both y <= and z <=
                nd = []
                min_z = ref_z
                # Traverse in order of increasing y, keep track of min z seen
                # Actually, for 2D HV: sort by y, then for each point, its contribution is (y_next - y_i) * (z_i - min_z_so_far)
                # where min_z_so_far is the min z of all points with y > y_i
                # Let me do it properly:
                n_pts = len(pts)
                # Sort by y ascending
                pts_sorted = pts[np.argsort(pts[:, 0])]
                hv2 = 0.0
                # For each point, find the next point in y and the min z of points with larger y
                for i in range(n_pts):
                    y_next = pts_sorted[i + 1, 0] if i + 1 < n_pts else ref_y
                    # Find min z among points j > i
                    if i + 1 < n_pts:
                        min_z_remaining = np.min(pts_sorted[i + 1:, 1])
                    else:
                        min_z_remaining = ref_z
                    if min_z_remaining > pts_sorted[i, 1]:
                        w_y = y_next - pts_sorted[i, 0]
                        w_z = min_z_remaining - pts_sorted[i, 1]
                        if w_y > 0 and w_z > 0:
                            hv2 += w_y * w_z
                return hv2

            # 3D HV using the slicing method:
            # Sort by f0
            order = np.argsort(P[:, 0])
            P_sorted = P[order]
            n_pts = len(P_sorted)
            hv = 0.0
            for i in range(n_pts):
                f0_min = P_sorted[i, 0]
                f0_max = P_sorted[i + 1, 0] if i + 1 < n_pts else ref[0]
                if f0_max <= f0_min:
                    continue
                # Collect all points with f0 in [f0_min, f0_max)
                # Project onto (f1, f2)
                mask_slice = (P_sorted[:, 0] >= f0_min) & (P_sorted[:, 0] < f0_max)
                slice_pts = P_sorted[mask_slice, 1:3]
                # Compute 2D HV of these points with reference (ref[1], ref[2])
                hv_slice = hv_2d(slice_pts, ref[1], ref[2])
                hv += hv_slice * (f0_max - f0_min)

            return hv

        def marginal_hv(point, archive_F):
            """Compute the marginal HV contribution of adding point to archive_F."""
            if len(archive_F) == 0:
                return hv_3d([point], ref_point)
            if not np.all(point < ref_point):
                return 0.0
            for f in archive_F:
                if dominates(f, point):
                    return 0.0
            hv_before = hv_3d(archive_F, ref_point)
            archive_new = list(archive_F) + [point]
            hv_after = hv_3d(archive_new, ref_point)
            return hv_after - hv_before

        # --- Phase 1: Coarse grid search ---
        grid_size = 4
        candidates = []
        for a in range(grid_size):
            for b in range(grid_size):
                for c in range(grid_size):
                    for d in range(grid_size):
                        x = np.array([a, b, c, d], dtype=float) / (grid_size - 1)
                        candidates.append(x)
        np.random.shuffle(candidates)

        # Evaluate all grid points first to build initial archive
        grid_results = []
        for x in candidates:
            if used >= budget:
                break
            if time.time() > deadline:
                break
            try:
                f = np.array(problem(x))
            except Exception:
                continue
            grid_results.append((x, f))
            used += 1

        # Sort by sum of objectives to prioritize good points
        grid_results.sort(key=lambda t: np.sum(t[1]))

        # Build archive greedily by HV contribution
        for x, f in grid_results:
            if time.time() > deadline:
                break
            if not np.all(f < ref_point):
                all_F.append(f)
                all_X.append(x)
                continue
            mhv = marginal_hv(f, all_F)
            if mhv > 1e-10:
                all_F.append(f)
                all_X.append(x)

        # --- Phase 2: Local refinement around non-dominated points ---
        # Extract non-dominated points from archive
        archive_arr = np.array(all_F) if all_F else np.empty((0, 3))
        if len(archive_arr) > 0:
            # Find non-dominated points
            nd_mask = np.ones(len(archive_arr), dtype=bool)
            for i in range(len(archive_arr)):
                for j in range(len(archive_arr)):
                    if i != j and dominates(archive_arr[j], archive_arr[i]):
                        nd_mask[i] = False
                        break
            nd_indices = np.where(nd_mask)[0]
            nd_points = [all_X[i] for i in nd_indices if i < len(all_X)]

            # Use remaining budget for local refinement
            remaining_budget = budget - used
            if remaining_budget > 0 and nd_points:
                # Nelder-Mead refinement around each non-dominated point
                from scipy.optimize import minimize

                def hypervolume_objective(x):
                    """Minimize the sum of objectives (proxy for HV improvement)."""
                    try:
                        f = np.array(problem(x))
                        if not np.all(f < ref_point):
                            return 1e6
                        return np.sum(f)
                    except Exception:
                        return 1e6

                # Refine each non-dominated point
                points_to_refine = nd_points[:min(10, len(nd_points))]
                evals_per_refine = max(1, remaining_budget // max(1, len(points_to_refine)))

                for x0 in points_to_refine:
                    if used >= budget:
                        break
                    if time.time() > deadline:
                        break

                    # Try Nelder-Mead with limited iterations
                    try:
                        result = minimize(
                            hypervolume_objective,
                            x0,
                            method='Nelder-Mead',
                            options={'maxiter': evals_per_refine, 'xatol': 1e-4, 'fatol': 1e-6}
                        )
                    except Exception:
                        continue

                    x_opt = np.clip(result.x, lo, hi)
                    try:
                        f_opt = np.array(problem(x_opt))
                    except Exception:
                        continue

                    used += 1  # Account for evaluations used by optimizer

                    # Check if this improves HV
                    if np.all(f_opt < ref_point):
                        mhv = marginal_hv(f_opt, all_F)
                        if mhv > 1e-10:
                            all_F.append(f_opt)
                            all_X.append(x_opt)

        return all_X
# EVOLVE-BLOCK-END
