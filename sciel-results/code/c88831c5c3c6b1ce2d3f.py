"""Multi-objective optimizer on RE37 using hypervolume-guided sequential search with local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-guided sequential search with local Nelder-Mead refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        ref_point = np.array([1.1, 1.3, 1.1])

        # Generate a dense candidate lattice in decision space
        # 4D grid with 8 points per dimension = 8^4 = 4096 candidates
        grid_size = 8
        axes = [np.linspace(lo[i], hi[i], grid_size) for i in range(dim)]
        grid = np.array(np.meshgrid(*axes, indexing='ij')).reshape(dim, -1).T
        n_grid = len(grid)

        # Shuffle grid to avoid systematic bias
        rng = np.random.default_rng(123)
        perm = rng.permutation(n_grid)
        grid = grid[perm]

        # Phase 1: Evaluate a subset of the grid to build initial archive
        # Use Latin Hypercube-like sampling from the grid for initial diversity
        init_size = min(30, budget // 5)
        init_indices = rng.choice(n_grid, size=init_size, replace=False)
        X_init = grid[init_indices]
        F_init = np.array([problem(x) for x in X_init])
        used = init_size

        # Build initial archive (non-dominated points)
        archive_X = X_init.copy()
        archive_F = F_init.copy()
        archive = self._non_dominated(archive_F)
        archive_X = archive_X[archive]
        archive_F = archive_F[archive]

        # Phase 2: Sequential hypervolume-guided selection
        # For each remaining evaluation, pick the grid point that maximizes
        # marginal hypervolume contribution when added to the current archive
        remaining_indices = np.setdiff1d(np.arange(n_grid), init_indices)
        rng.shuffle(remaining_indices)

        # Pre-compute: we'll evaluate grid points one by one, but guided by HV
        # To be efficient, we batch-evaluate and then select the best HV contributor
        batch_size = 20

        while used < budget:
            if time.time() > deadline - 15:
                break

            # Determine how many we can evaluate in this batch
            n_eval = min(batch_size, budget - used, len(remaining_indices))
            if n_eval <= 0:
                break

            # Select candidates: mix of top-HV grid points and random
            # First, compute HV contribution for a subset of remaining grid points
            # to guide selection
            candidate_subset_size = min(100, len(remaining_indices))
            candidate_subset = remaining_indices[:candidate_subset_size]

            # For each candidate, compute marginal HV if added to archive
            best_contrib = -1
            best_idx = -1
            for ci in candidate_subset:
                x_candidate = grid[ci]
                # Don't re-evaluate if already in archive
                f_candidate = problem(x_candidate)
                used += 1
                if used > budget:
                    break
                # Check if this point is dominated by archive
                if not self._is_dominated(f_candidate, archive_F):
                    # Compute marginal HV
                    new_archive_F = np.vstack([archive_F, [f_candidate]])
                    new_hv = self._hypervolume(new_archive_F, ref_point)
                    old_hv = self._hypervolume(archive_F, ref_point)
                    contrib = new_hv - old_hv
                    if contrib > best_contrib:
                        best_contrib = contrib
                        best_idx = ci
                    # Add to archive if not dominated
                    if not self._is_dominated(f_candidate, archive_F):
                        archive_X = np.vstack([archive_X, [x_candidate]])
                        archive_F = np.vstack([archive_F, [f_candidate]])
                        # Remove points dominated by new point
                        archive = self._non_dominated(archive_F)
                        archive_X = archive_X[archive]
                        archive_F = archive_F[archive]
                # Remove from remaining
                remaining_indices = np.delete(remaining_indices, np.where(remaining_indices == ci)[0][0])

            if used >= budget:
                break

        # Phase 3: Local refinement of archive points
        if time.time() < deadline - 10 and used < budget:
            from scipy.optimize import minimize

            # Sort archive by HV contribution (approximate: by distance to reference)
            # Refine the most promising points
            n_refine = min(len(archive_F), budget - used)
            for idx in range(n_refine):
                if time.time() > deadline - 5:
                    break
                if used >= budget:
                    break

                x0 = archive_X[idx]
                f0 = archive_F[idx]

                # Multi-start Nelder-Mead on scalarization with different weights
                # Try a few weight vectors
                weight_sets = [
                    np.ones(n_obj) / n_obj,
                    np.array([0.5, 0.25, 0.25]),
                    np.array([0.25, 0.5, 0.25]),
                    np.array([0.25, 0.25, 0.5]),
                ]

                for weights in weight_sets:
                    if used >= budget:
                        break

                    def scalarized(x, w=weights):
                        fx = problem(x)
                        return np.dot(fx, w)

                    try:
                        res = minimize(scalarized, x0, method='Nelder-Mead',
                                       bounds=list(zip(lo, hi)),
                                       options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-6})
                        if used < budget:
                            fx = problem(res.x)
                            used += 1
                            # Check if this improves the archive
                            if not self._is_dominated(fx, archive_F):
                                archive_X = np.vstack([archive_X, [res.x]])
                                archive_F = np.vstack([archive_F, [fx]])
                                archive = self._non_dominated(archive_F)
                                archive_X = archive_X[archive]
                                archive_F = archive_F[archive]
                    except Exception:
                        pass

        return archive_X

    def _is_dominated(self, f, archive_F):
        """Check if point f is dominated by any point in archive_F."""
        for af in archive_F:
            if np.all(af <= f) and np.any(af < f):
                return True
        return False

    def _non_dominated(self, F):
        """Return indices of non-dominated points in F."""
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
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    dominated[j] = True
        return np.where(~dominated)[0]

    def _hypervolume(self, F, ref_point):
        """Compute hypervolume of non-dominated points F with respect to ref_point."""
        # Filter points that are better than reference in all objectives
        valid = np.all(F < ref_point, axis=1)
        F = F[valid]
        if len(F) == 0:
            return 0.0

        n = len(F)
        n_obj = F.shape[1]

        # For 3 objectives, use the inclusion-exclusion approach
        # Sort by first objective
        order = np.argsort(F[:, 0])
        F_sorted = F[order]

        hv = 0.0
        for i in range(n):
            # For each point, compute the volume of the region it uniquely contributes
            # Region: [F[i][0], min(F[j][0] for j with F[j][0] > F[i][0] and F[j][1] > F[i][1] and F[j][2] > F[i][2])]
            # This is complex; use a simpler but exact method for 3D

            # The contribution of point i is the volume of the box:
            # [F[i][0], next_x) x [F[i][1], next_y) x [F[i][2], next_z)
            # where next_x = min of F[j][0] for j that dominate in the sense that
            # F[j][1] <= F[i][1] and F[j][2] <= F[i][2] and F[j][0] > F[i][0]
            # This is getting complicated. Use a different approach.

            pass

        # Use a simpler exact method: for each point, the HV contribution is
        # the volume of the orthogonal box from the point to the reference,
        # minus the volume already covered by other points.
        # For exactness in 3D, we can use the following:
        # Sort points by x-coordinate. For each point, its contribution is:
        # (ref_x - F[i][0]) * (ref_y - F[i][1]) * (ref_z - F[i][2])
        # minus overlaps with previously processed points.

        # Actually, let's use a clean recursive or iterative approach.
        # For 3 objectives, we can compute HV exactly using the following:
        # Sort by first objective. The HV is the sum over each point of
        # the volume of the region where it is the "first" point encountered.

        # Let's use a different exact method:
        # HV = sum over all subsets S of points of (-1)^(|S|+1) * volume(intersection of boxes)
        # This is 2^n which is too slow for large n.

        # For our purposes, n is at most ~50, so we can use a different approach.
        # Let's use the standard algorithm: sort by first objective, then for each point,
        # compute the "skyline" in the remaining 2 objectives.

        # Simpler: use the fact that for a small number of points, we can compute
        # the HV by decomposing into rectangular boxes.

        # Let's just use a straightforward O(n^3) method for 3D:
        # Sort by x. For each point i, the contribution is:
        # dx_i * area_2d(point i in 2D after removing points that dominate in y,z)

        # Actually, the cleanest exact method for small n:
        # The hypervolume is the volume of the union of boxes [F[i], ref] for all i.
        # We can compute this by inclusion-exclusion or by decomposing into non-overlapping boxes.

        # For simplicity and correctness, let's use a different approach:
        # Since n is small (<= 50), we can use the following:
        # Sort points by first coordinate. Process in order. Maintain the "envelope"
        # in the remaining 2D space.

        # Let me just implement a correct 3D HV calculation:
        return self._hv3d(F, ref_point)

    def _hv3d(self, F, ref):
        """Exact 3D hypervolume computation."""
        n = len(F)
        if n == 0:
            return 0.0

        # Sort by first objective
        order = np.argsort(F[:, 0])
        F = F[order]

        total_hv = 0.0
        for i in range(n):
            # The box for point i: [F[i][0], ref[0]] x [F[i][1], ref[1]] x [F[i][2], ref[2]]
            # But we need to subtract the part already covered by points j < i
            # that have F[j][0] <= F[i][0] and cover part of this box.

            # The contribution of point i is the volume of the region in its box
            # that is not covered by any box of points j with F[j][0] < F[i][0].

            # For points j with F[j][0] < F[i][0], they cover the region:
            # [F[j][0], ref[0]] x [F[j][1], ref[1]] x [F[j][2], ref[2]]
            # The part of point i's box covered by point j is:
            # [F[i][0], ref[0]] x [max(F[i][1], F[j][1]), ref[1]] x [max(F[i][2], F[j][2]), ref[2]]
            # (if F[i][1] < ref[1] and F[i][2] < ref[2] and the max values < ref)

            # This is still complex due to overlaps between multiple j's.
            # Let's use a different approach: compute the 2D HV in the yz-plane
            # for the "skyline" of points that are not dominated in yz by earlier points.

            # Actually, the standard algorithm:
            # Sort by x. For each point, compute the 2D HV of the set of points
            # (y, z) that are not dominated in yz by any earlier point's (y, z),
            # but only considering the region where x > F[i][0].

            # This is getting too complex. Let's use a simpler but correct method:
            # For each point, its unique contribution is the volume of the box
            # [F[i], ref] minus the union of boxes [F[j], ref] for j != i.
            # We can approximate this by computing the 2D HV in yz for the
            # "active" set of points.

            # Let me just use a clean implementation:
            pass

        # Let's use a completely different, simpler approach:
        # The HV is the volume of the union of boxes. For 3D with small n,
        # we can use the following exact method:
        # Sort by x. For each point i, the "new" volume it adds is:
        # (ref[0] - F[i][0]) * HV_2d({(F[j][1], F[j][2]) for j <= i, F[j][0] >= F[i][0]} minus dominated)
        # This is still not quite right.

        # Let's just use a numerical approach that's exact enough:
        # Decompose the union of boxes into non-overlapping rectangular boxes.
        # For 3D, we can do this by:
        # 1. Collect all unique x-coordinates: F[i][0] and ref[0]
        # 2. For each pair of consecutive x-values, compute the 2D HV in yz
        #    for the set of points whose x-range includes this interval.

        x_coords = sorted(set(np.concatenate([F[:, 0], [ref[0]]])))
        total = 0.0
        for k in range(len(x_coords) - 1):
            x_lo = x_coords[k]
            x_hi = x_coords[k + 1]
            if x_hi <= x_lo:
                continue
            # Points that span this x-interval: F[i][0] <= x_lo and ref[0] >= x_hi
            # i.e., F[i][0] <= x_lo (since all points have F[i][0] < ref[0])
            active = F[:, 0] <= x_lo + 1e-10
            if not np.any(active):
                continue
            F_active = F[active]
            # 2D HV of F_active in yz with reference (ref[1], ref[2])
            hv2d = self._hv2d(F_active[:, 1:3], np.array([ref[1], ref[2]]))
            total += (x_hi - x_lo) * hv2d

        return total

    def _hv2d(self, F, ref):
        """Exact 2D hypervolume computation."""
        n = len(F)
        if n == 0:
            return 0.0

        # Filter valid points
        valid = np.all(F < ref, axis=1)
        F = F[valid]
        n = len(F)
        if n == 0:
            return 0.0

        # Sort by first coordinate
        order = np.argsort(F[:, 0])
        F = F[order]

        total = 0.0
        for i in range(n):
            # The box for point i: [F[i][0], ref[0]] x [F[i][1], ref[1]]
            # Contribution: width * height, where width = ref[0] - F[i][0]
            # and height = ref[1] - max(F[i][1], max of F[j][1] for j < i with F[j][0] < F[i][0] and F[j][1] < F[i][1])
            # Actually, in 2D, the contribution of point i (sorted by x) is:
            # (ref[0] - F[i][0]) * (ref[1] - F[i][1])
            # minus the part covered by earlier points.
            # The earlier point j covers [F[j][0], ref[0]] x [F[j][1], ref[1]].
            # The overlap with point i's box is:
            # [F[i][0], ref[0]] x [max(F[i][1], F[j][1]), ref[1]]
            # But we need the union of all such overlaps.

            # In 2D, the skyline is monotone. After sorting by x, the y-values
            # of non-dominated points are decreasing. So the contribution of point i
            # is (ref[0] - F[i][0]) * (ref[1] - F[i][1]) minus
            # (ref[0] - F[i][0]) * (F[i][1] - F[i-1][1]) if F[i-1][1] > F[i][1]
            # Wait, that's not quite right either.

            # Correct 2D HV: sort by x. For each point i, the new area added is:
            # (ref[0] - F[i][0]) * (ref[1] - F[i][1]) - (ref[0] - F[i][0]) * (ref[1] - max_y_so_far)
            # where max_y_so_far is the minimum y among points j < i (since we want the
            # highest y that's been seen, which corresponds to the smallest y value... no)

            # Let me think again. In 2D, with minimization:
            # Sort by x ascending. The non-dominated front has decreasing y.
            # The HV is the sum of rectangles. For point i (0-indexed, sorted by x):
            # width_i = ref[0] - F[i][0]
            # height_i = ref[1] - F[i][1]
            # But the rectangle for point i overlaps with rectangle for point i-1.
            # The unique contribution of point i is:
            # width_i * (F[i-1][1] - F[i][1]) if i > 0 and F[i-1][1] > F[i][1]
            # plus width_i * (ref[1] - F[i][1]) if i == 0 or F[i-1][1] <= F[i][1]

            # Actually, the standard formula:
            # After sorting by x, the HV = sum over i of (ref[0] - F[i][0]) * (ref[1] - F[i][1])
            # minus sum over i of (ref[0] - F[i][0]) * (F[i][1] - F[i-1][1]) for i where F[i][1] < F[i-1][1]
            # This is getting confused. Let me just use the simple approach:

            # The HV in 2D is the area under the "staircase" curve.
            # Sort by x. For each consecutive pair, the area is:
            # (F[i+1][0] - F[i][0]) * (ref[1] - F[i][1]) + (ref[0] - F[n-1][0]) * (ref[1] - F[n-1][1])
            # But we need to handle the non-dominated points only.

            pass

        # Let's just compute it directly:
        # The HV is the area of the union of rectangles [F[i][0], ref[0]] x [F[i][1], ref[1]].
        # In 2D, this can be computed by sorting by x and keeping track of the minimum y.
        # Actually, for minimization, we want the area where both coordinates are below ref.
        # The union of boxes [x_i, ref_x] x [y_i, ref_y] has area:
        # Sort by x. For each point, the new area added is:
        # (ref_x - x_i) * (min(ref_y, y_i) - min(ref_y, y_{i-1})) if y_i < y_{i-1}
        # Hmm, let's just do it simply:

        # Filter to non-dominated points
        nd = self._non_dominated(F)
        F_nd = F[nd]
        if len(F_nd) == 0:
            return 0.0

        # Sort by x
        order = np.argsort(F_nd[:, 0])
        F_nd = F_nd[order]

        total = 0.0
        for i in range(len(F_nd)):
            x0 = F_nd[i][0]
            y0 = F_nd[i][1]
            # The rectangle is [x0, ref[0]] x [y0, ref[1]]
            # The part not covered by previous rectangles:
            # Previous rectangles cover [x_j, ref[0]] x [y_j, ref[1]] for j < i
            # The union of previous rectangles in the y-direction covers [min_y, ref[1]]
            # where min_y = min(y_j for j < i)
            # So the new area is (ref[0] - x0) * (min_y - y0) if y0 < min_y
            # But min_y is the minimum y among previous points, and since points are
            # non-dominated and sorted by x, y is decreasing, so min_y = y_{i-1} (the previous point's y)
            # Wait, non-dominated means if x increases, y must decrease. So y is decreasing.
            # So the previous point has y_{i-1} > y_i (since x_{i-1} < x_i and non-dominated).
            # The previous rectangles cover y from min(y_j) to ref[1], where min(y_j) = y_{i-1} (the largest y among previous, since y is decreasing... no, min is the smallest y, which is y_{i-1} since y is decreasing).
            # Actually, y is decreasing, so y_0 > y_1 > ... > y_{n-1}.
            # The union of rectangles covers y from y_{n-1} to ref[1] for the last point,
            # but for point i, the previous points cover y from y_{i-1} to ref[1] (since y_{i-1} is the smallest y among points 0..i-1).
            # So the new area added by point i is:
            # (ref[0] - x_i) * (y_{i-1} - y_i) if i > 0
            # For i = 0, it's (ref[0] - x_0) * (ref[1] - y_0)

            if i == 0:
                total += (ref[0] - x0) * (ref[1] - y0)
            else:
                prev_y = F_nd[i-1][1]
                if prev_y > y0:
                    total += (ref[0] - x0) * (prev_y - y0)

        return total
# EVOLVE-BLOCK-END

