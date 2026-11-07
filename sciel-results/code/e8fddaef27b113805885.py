"""Multi-objective optimizer on RE34: hypervolume-guided evolutionary search with local refinement."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Hypervolume-guided evolutionary multi-objective optimizer for 3-objective RE34."""

    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Reference point for hypervolume
        ref = np.array([1705.0, 12.0, 0.3])

        used = 0
        X = []  # list of decision vectors
        F = []  # list of objective vectors

        def evaluate(x):
            nonlocal used
            x = np.clip(x, lo, hi)
            f = np.asarray(problem(x), dtype=np.float64)
            used += 1
            return x, f

        def dominated_mask(F_arr):
            """Return boolean mask: True for points that are dominated by at least one other point."""
            n = len(F_arr)
            dom = np.zeros(n, dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dom[i] = True
                        break
            return dom

        def non_dominated_set(F_arr):
            """Return indices of non-dominated points."""
            n = len(F_arr)
            nd = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd[i]:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        nd[i] = False
                        break
            return np.where(nd)[0]

        def hypervolume_3d(points, ref_pt):
            """Exact 3D hypervolume for points strictly better than ref_pt in all objectives."""
            # Filter points that are strictly better than ref in all objectives
            pts = [p for p in points if np.all(p < ref_pt)]
            if not pts:
                return 0.0
            pts = np.array(pts)
            n = len(pts)
            if n == 1:
                return float(np.prod(ref_pt - pts[0]))
            if n == 2:
                p1, p2 = pts[0], pts[1]
                # Hypervolume of union of two boxes
                # Sort by first objective
                if p1[0] > p2[0]:
                    p1, p2 = p2, p1
                # p1[0] <= p2[0]
                # Box1: [p1[0], ref[0]] x [p1[1], ref[1]] x [p1[2], ref[2]]
                # Box2: [p2[0], ref[0]] x [p2[1], ref[1]] x [p2[2], ref[2]]
                # Overlap in x: [p2[0], ref[0]] (since p1[0] <= p2[0])
                ov_x = max(0.0, ref_pt[0] - p2[0])
                ov_y = max(0.0, min(ref_pt[1] - p1[1], ref_pt[1] - p2[1]))
                ov_z = max(0.0, min(ref_pt[2] - p1[2], ref_pt[2] - p2[2]))
                vol1 = (ref_pt[0] - p1[0]) * (ref_pt[1] - p1[1]) * (ref_pt[2] - p1[2])
                vol2 = (ref_pt[0] - p2[0]) * (ref_pt[1] - p2[1]) * (ref_pt[2] - p2[2])
                ov_vol = ov_x * ov_y * ov_z
                return float(vol1 + vol2 - ov_vol)
            # General case: sort by first objective, use sweep
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            # For 3D, we can use the inclusion-exclusion approach
            # Sort by x, then for each point, compute the 2D hypervolume of the "slice"
            total = 0.0
            prev_x = pts[0, 0]
            # Actually, let's use a simpler approach for small n
            # Since n is at most ~50 in our archive, we can use a direct method
            # Sort by first coordinate
            pts_sorted = pts[np.argsort(pts[:, 0])]
            total_hv = 0.0
            for i in range(len(pts_sorted)):
                p = pts_sorted[i]
                # The contribution of point i is the volume of the box from p to ref
                # minus the volume already covered by points with smaller x (better in obj 1)
                # that also cover this region in obj 2 and obj 3
                box_vol = (ref_pt[0] - p[0]) * (ref_pt[1] - p[1]) * (ref_pt[2] - p[2])
                # Subtract overlap with previously processed points
                overlap = 0.0
                for j in range(i):
                    q = pts_sorted[j]
                    # Overlap in x: from max(p[0], q[0]) to ref[0] = from p[0] to ref[0] since p[0] >= q[0]
                    ov_x = ref_pt[0] - p[0]
                    ov_y = max(0.0, min(ref_pt[1] - p[1], ref_pt[1] - q[1]))
                    ov_z = max(0.0, min(ref_pt[2] - p[2], ref_pt[2] - q[2]))
                    overlap += ov_x * ov_y * ov_z
                total_hv += box_vol - overlap
            return float(total_hv)

        def hv_contribution(F_arr, ref_pt):
            """Compute hypervolume contribution of each point in F_arr."""
            n = len(F_arr)
            contribs = np.zeros(n)
            # Only consider points better than ref in all objectives
            valid = np.all(F_arr < ref_pt, axis=1)
            if not np.any(valid):
                return contribs
            valid_idx = np.where(valid)[0]
            for i in valid_idx:
                # Hypervolume with point i vs without point i
                hv_with = hypervolume_3d(F_arr[valid_idx].tolist(), ref_pt)
                # Remove point i
                mask = np.ones(len(valid_idx), dtype=bool)
                mask[list(valid_idx).index(i)] = False
                if np.any(mask):
                    hv_without = hypervolume_3d(F_arr[valid_idx[mask]].tolist(), ref_pt)
                else:
                    hv_without = 0.0
                contribs[i] = hv_with - hv_without
            return contribs

        # Phase 1: Latin Hypercube Sampling for diverse initial population
        n_init = min(40, budget // 5)
        rng = np.random.default_rng(42)
        for i in range(n_init):
            x = np.zeros(dim)
            for d in range(dim):
                x[d] = lo[d] + (i + rng.random()) / n_init * span[d]
            x, f = evaluate(x)
            X.append(x)
            F.append(f)

        # Phase 2: Tchebycheff scalarization for targeted exploration
        n_tcheb = min(20, budget // 10)
        # Generate weight vectors
        for _ in range(n_tcheb):
            weights = rng.dirichlet(np.ones(n_obj))
            # Find a starting point: use the point that best matches this weight direction
            if F:
                F_arr = np.array(F)
                valid = np.all(F_arr < ref, axis=1)
                if np.any(valid):
                    # Normalize objectives roughly
                    f_min = F_arr[valid].min(axis=0)
                    f_max = F_arr[valid].max(axis=0)
                    f_norm = (F_arr[valid] - f_min) / (f_max - f_min + 1e-10)
                    scores = np.sum(f_norm * weights, axis=1)
                    best_idx = valid_idx[np.argmin(scores)] if False else None
                    # Just pick a random valid point as start
                    start_idx = valid_idx[rng.integers(len(valid_idx))]
                else:
                    start_idx = rng.integers(len(F))
            else:
                start_idx = 0

            x0 = X[start_idx].copy()

            # Local search using Nelder-Mead on Tchebycheff scalarization
            def tcheb_obj(x):
                f = np.asarray(problem(np.clip(x, lo, hi)), dtype=np.float64)
                return np.max(np.abs(weights * (f - ref)))  # Approximate

            # Actually, let's just do a simple local search with random perturbations
            best_x = x0.copy()
            best_f = F[start_idx].copy()
            step_size = 0.15 * span
            for _ in range(5):
                if time.time() > deadline:
                    break
                dx = rng.standard_normal(dim) * step_size
                x_new = np.clip(best_x + dx, lo, hi)
                f_new = np.asarray(problem(x_new), dtype=np.float64)
                used += 1
                if np.all(f_new <= best_f) and np.any(f_new < best_f):
                    best_x = x_new
                    best_f = f_new
                elif not np.any(np.all(np.array(F) <= f_new, axis=1) & np.any(np.array(F) < f_new, axis=1)):
                    # Non-dominated, add it
                    X.append(best_x)
                    F.append(best_f)
                    break
                step_size *= 0.8

            X.append(best_x)
            F.append(best_f)

        # Phase 3: Evolutionary loop with hypervolume-guided selection
        pop_size = min(50, max(20, budget // 4))

        # Build initial archive
        F_arr = np.array(F)
        X_arr = np.array(X)
        nd_idx = non_dominated_set(F_arr)
        if len(nd_idx) > pop_size:
            # Keep top pop_size by hypervolume contribution
            nd_F = F_arr[nd_idx]
            contribs = hv_contribution(nd_F, ref)
            top_idx = nd_idx[np.argsort(contribs)[::-1][:pop_size]]
            X_arr = X_arr[top_idx]
            F_arr = F_arr[top_idx]
        X = [x for x in X_arr]
        F = [f for f in F_arr]

        while used < budget:
            if time.time() > deadline:
                break

            F_arr = np.array(F)
            X_arr = np.array(X)

            # Get non-dominated set
            nd_idx = non_dominated_set(F_arr)
            if len(nd_idx) == 0:
                nd_idx = np.array([0])

            # Compute hypervolume contributions for selection
            if len(nd_idx) > 1:
                contribs = hv_contribution(F_arr[nd_idx], ref)
                # Select parent with highest contribution
                parent_local = np.argmax(contribs)
                parent_idx = nd_idx[parent_local]
            else:
                parent_idx = nd_idx[0]

            # Crossover: blend with another non-dominated point
            if len(nd_idx) > 1:
                other_idx = nd_idx[rng.integers(len(nd_idx))]
                if other_idx == parent_idx:
                    other_idx = nd_idx[(nd_idx == parent_idx).nonzero()[0][0] + 1] % len(nd_idx)
                alpha = rng.random()
                child_x = alpha * X_arr[parent_idx] + (1 - alpha) * X_arr[other_idx]
            else:
                child_x = X_arr[parent_idx].copy()

            # Mutation
            mutation_rate = 0.3
            for d in range(dim):
                if rng.random() < mutation_rate:
                    child_x[d] += rng.standard_normal() * 0.1 * span[d]
            child_x = np.clip(child_x, lo, hi)

            child_f = np.asarray(problem(child_x), dtype=np.float64)
            used += 1

            # Update archive
            # Check if child is dominated
            dominated = False
            for i in range(len(F)):
                if np.all(np.array(F[i]) <= child_f) and np.any(np.array(F[i]) < child_f):
                    dominated = True
                    break

            if not dominated:
                # Remove points dominated by child
                new_X = []
                new_F = []
                for i in range(len(X)):
                    if not (np.all(child_f <= np.array(F[i])) and np.any(child_f < np.array(F[i]))):
                        new_X.append(X[i])
                        new_F.append(F[i])
                new_X.append(child_x)
                new_F.append(child_f)
                X, F = new_X, new_F

            # If archive too large, prune by hypervolume contribution
            if len(F) > pop_size:
                F_arr = np.array(F)
                X_arr = np.array(X)
                nd_idx = non_dominated_set(F_arr)
                if len(nd_idx) > pop_size:
                    contribs = hv_contribution(F_arr[nd_idx], ref)
                    top_idx = nd_idx[np.argsort(contribs)[::-1][:pop_size]]
                    X = [X_arr[i] for i in top_idx]
                    F = [F_arr[i] for i in top_idx]

        return X
# EVOLVE-BLOCK-END
