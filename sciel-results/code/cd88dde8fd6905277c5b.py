# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 270.0  # safety margin before 300s kill

        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng_ = hi - lo

        # Reference point for hypervolume
        ref = np.array([3.0, 5.0, 7.0])

        # We'll track all evaluated points
        all_X = []
        all_F = []

        def hypervolume_3d(points, ref_point):
            """Compute exact 3D hypervolume for points dominated by ref_point."""
            # Filter points that are better than reference in all objectives
            mask = np.all(points < ref_point, axis=1)
            pts = points[mask]
            if len(pts) == 0:
                return 0.0

            # For 3D hypervolume, use the standard algorithm
            # Sort by first objective
            order = np.argsort(pts[:, 0])
            pts = pts[order]

            # Remove dominated points (keep only non-dominated)
            nd_mask = np.ones(len(pts), dtype=bool)
            for i in range(len(pts)):
                if not nd_mask[i]:
                    continue
                # Check if any point with index < i dominates i
                # Since sorted by obj0, only points before i can dominate
                if i > 0:
                    prev = pts[:i]
                    # A point p dominates i if p[1] <= pts[i][1] and p[2] <= pts[i][2]
                    # and at least one strict
                    dom = (prev[:, 1] <= pts[i][1]) & (prev[:, 2] <= pts[i][2])
                    if np.any(dom):
                        nd_mask[i] = False

            pts_nd = pts[nd_mask]
            if len(pts_nd) == 0:
                return 0.0

            # Compute hypervolume using the "layered" approach
            # For 3D: sort by x, then for each point compute the 2D hypervolume of the
            # rectangle from (x_i, y_i, z_i) to (x_{i+1}, ref_y, ref_z)
            # Actually, let's use a cleaner approach:
            # Sort by first coordinate
            order = np.argsort(pts_nd[:, 0])
            pts_nd = pts_nd[order]

            hv = 0.0
            for i in range(len(pts_nd)):
                x_i = pts_nd[i, 0]
                x_next = pts_nd[i + 1, 0] if i + 1 < len(pts_nd) else ref_point[0]
                if x_next <= x_i:
                    continue
                # The slice between x_i and x_next has width (x_next - x_i)
                # The 2D area in y-z plane for points with obj0 <= x_next
                # is the hypervolume of points with index >= i in the y-z plane
                # bounded by ref_y, ref_z
                # Points from i onward have y, z values
                sub = pts_nd[i:, 1:]  # y, z for points i..end
                # 2D hypervolume: area of union of rectangles [y_j, ref_y] x [z_j, ref_z]
                # Sort by y
                order2 = np.argsort(sub[:, 0])
                sub = sub[order2]
                area = 0.0
                max_z = ref_point[2]
                for j in range(len(sub)):
                    y_j = sub[j, 0]
                    z_j = sub[j, 1]
                    if z_j >= max_z:
                        continue
                    # This point contributes area (ref_y - y_j) * (max_z - z_j)
                    # but we need to be careful about overlaps
                    # Actually for 2D hypervolume with reference:
                    # Sort by y, then the contribution of point j is:
                    # (y_{j+1} - y_j) * (ref_z - min_z_so_far) where min_z_so_far is the
                    # minimum z among points j..end... no, that's not right either.
                    # Let me use a simpler exact 2D HV computation.
                    pass

                # Let me just use a direct 2D HV computation
                # Sort by y
                order2 = np.argsort(sub[:, 0])
                sub_sorted = sub[order2]
                area = 0.0
                current_max_z = ref_point[2]
                for j in range(len(sub_sorted)):
                    y_j = sub_sorted[j, 0]
                    z_j = sub_sorted[j, 1]
                    if z_j < current_max_z:
                        # Next y boundary
                        y_next = sub_sorted[j + 1, 0] if j + 1 < len(sub_sorted) else ref_point[1]
                        area += (y_next - y_j) * (current_max_z - z_j)
                        current_max_z = min(current_max_z, z_j)

                hv += (x_next - x_i) * area

            return hv

        # Better: use a well-known exact 3D HV algorithm
        def compute_hv(points, ref_point):
            """Exact 3D hypervolume."""
            # Filter
            mask = np.all(points < ref_point, axis=1)
            pts = points[mask]
            if len(pts) == 0:
                return 0.0

            # Remove dominated points
            n = len(pts)
            nd = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd[i]:
                    continue
                for j in range(n):
                    if i == j or not nd[j]:
                        continue
                    if np.all(pts[j] <= pts[i]) and np.any(pts[j] < pts[i]):
                        nd[i] = False
                        break

            pts = pts[nd]
            if len(pts) == 0:
                return 0.0

            # Sort by first objective
            order = np.argsort(pts[:, 0])
            pts = pts[order]

            hv = 0.0
            for i in range(len(pts)):
                x_i = pts[i, 0]
                x_next = pts[i + 1, 0] if i + 1 < len(pts) else ref_point[0]
                if x_next <= x_i:
                    continue
                # 2D HV of points i..end in y-z plane
                sub = pts[i:, 1:]
                # Sort by y
                order2 = np.argsort(sub[:, 0])
                sub = sub[order2]
                area = 0.0
                max_z = ref_point[2]
                for j in range(len(sub)):
                    y_j = sub[j, 0]
                    z_j = sub[j, 1]
                    if z_j < max_z:
                        y_next = sub[j + 1, 0] if j + 1 < len(sub) else ref_point[1]
                        area += (y_next - y_j) * (max_z - z_j)
                        max_z = min(max_z, z_j)
                hv += (x_next - x_i) * area

            return hv

        # Initialize with a diverse set of points
        # Use Latin hypercube or just random spread
        init_size = min(30, budget // 5)
        X_init = lo + rng_ * np.random.default_rng(42).random((init_size, dim))
        for x in X_init:
            if time.time() > deadline:
                break
            f = problem(x)
            all_X.append(x)
            all_F.append(f)

        # Now use remaining budget for local search around best points
        # First, find non-dominated points
        def get_nd_points(F):
            F = np.array(F)
            n = len(F)
            nd = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd[i]:
                    continue
                for j in range(n):
                    if i == j or not nd[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nd[i] = False
                        break
            return nd

        used = len(all_X)

        # Local search phase: for each non-dominated point, try small perturbations
        # and keep improvements
        while used < budget:
            if time.time() > deadline:
                break

            F_arr = np.array(all_F)
            nd_mask = get_nd_points(F_arr)
            nd_indices = np.where(nd_mask)[0]

            if len(nd_indices) == 0:
                # All dominated, pick random point
                idx = np.random.randint(len(all_X))
            else:
                # Pick a random non-dominated point
                idx = nd_indices[np.random.randint(len(nd_indices))]

            x_base = all_X[idx].copy()
            f_base = all_F[idx].copy()

            # Try several perturbation directions
            n_trials = min(5, budget - used)
            for trial in range(n_trials):
                if used >= budget or time.time() > deadline:
                    break

                # Random perturbation with decreasing step size
                step = 0.1 * (1.0 - used / budget) + 0.01
                direction = np.random.randn(dim)
                direction /= (np.linalg.norm(direction) + 1e-10)
                x_new = x_base + step * rng_ * direction
                x_new = np.clip(x_new, lo, hi)

                f_new = problem(x_new)
                used += 1

                # Check if this is better (non-dominated or improves HV)
                # Simple check: is it non-dominated w.r.t. current set?
                F_arr = np.array(all_F)
                # Check if f_new is dominated by any existing point
                dominated = False
                for i in range(len(F_arr)):
                    if np.all(F_arr[i] <= f_new) and np.any(F_arr[i] < f_new):
                        dominated = True
                        break

                if not dominated:
                    all_X.append(x_new)
                    all_F.append(f_new)
                # Also check if it dominates any existing points (remove them)
                else:
                    # Still add it, and we'll clean up dominated points later
                    all_X.append(x_new)
                    all_F.append(f_new)

            # Periodically clean up dominated points to keep the set manageable
            if len(all_X) > 100:
                F_arr = np.array(all_F)
                nd_mask = get_nd_points(F_arr)
                if np.sum(nd_mask) < len(nd_mask):
                    all_X = [all_X[i] for i in range(len(all_X)) if nd_mask[i]]
                    all_F = [all_F[i] for i in range(len(all_F)) if nd_mask[i]]

        return all_X
# EVOLVE-BLOCK-END
