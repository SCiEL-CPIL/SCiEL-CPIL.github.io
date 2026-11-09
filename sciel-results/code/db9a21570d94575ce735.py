"""Multi-objective optimizer for RE34 using Hypervolume-Estimated Descent (HVED)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """HVED-based multi-objective optimizer for RE34."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 290.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Reference point for hypervolume computation
        ref = np.array([1705.0, 12.0, 0.3])

        # Initial population: diverse starting points
        init_size = min(30, budget)
        X = np.empty((init_size, dim))
        F = np.empty((init_size, n_obj))

        # Corner points
        X[0] = lo
        X[1] = hi
        F[0] = problem(X[0])
        F[1] = problem(X[1])

        # Midpoint
        mid = (lo + hi) / 2.0
        X[2] = mid
        F[2] = problem(X[2])

        # Axis-aligned extremes
        idx = 3
        for i in range(dim):
            if idx >= init_size:
                break
            pt = mid.copy()
            pt[i] = lo[i]
            X[idx] = pt
            F[idx] = problem(X[idx])
            idx += 1

        for i in range(dim):
            if idx >= init_size:
                break
            pt = mid.copy()
            pt[i] = hi[i]
            X[idx] = pt
            F[idx] = problem(X[idx])
            idx += 1

        # Fill remaining with Sobol-like quasi-random points for better coverage
        if idx < init_size:
            n_fill = init_size - idx
            # Use a simple Halton-like sequence for quasi-random spread
            for k in range(n_fill):
                x = np.zeros(dim)
                for j in range(dim):
                    # Use different bases for each dimension
                    base = j + 2
                    val = 0.0
                    frac = 1.0 / (k + 1)
                    while frac > 0:
                        val += frac
                        frac /= base
                    x[j] = lo[j] + (hi[j] - lo[j]) * (val % 1.0)
                X[idx + k] = x
            F[idx:] = np.array([problem(x) for x in X[idx:]])

        used = init_size

        def compute_hypervolume_3d(points, ref_pt):
            """Compute hypervolume in 3D using exact method."""
            # Filter points that are better than reference in all objectives
            valid = np.all(points < ref_pt, axis=1)
            pts = points[valid]
            if len(pts) == 0:
                return 0.0

            # Remove dominated points
            n = len(pts)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j or dominated[j]:
                        continue
                    if np.all(pts[i] <= pts[j]) and np.any(pts[i] < pts[j]):
                        dominated[j] = True
            pts = pts[~dominated]
            n = len(pts)
            if n == 0:
                return 0.0

            # Sort by first objective
            order = np.argsort(pts[:, 0])
            pts = pts[order]

            hv = 0.0
            for i in range(n):
                # For point i, compute the volume it uniquely contributes
                # Points j > i have larger obj0, so point i's unique region
                # is bounded by pts[i][0] on the left and the next point's obj0 on the right
                if i + 1 < n:
                    right_0 = pts[i + 1][0]
                else:
                    right_0 = ref_pt[0]

                # In the 2D slice (obj1, obj2), find the non-dominated frontier
                # among points i..n-1
                sub_pts = pts[i:]
                # Filter: must be < ref in obj1 and obj2
                valid2 = (sub_pts[:, 1] < ref_pt[1]) & (sub_pts[:, 2] < ref_pt[2])
                sub_pts = sub_pts[valid2]
                if len(sub_pts) == 0:
                    continue

                # Compute 2D hypervolume for this slice
                # Sort by obj1
                order2 = np.argsort(sub_pts[:, 1])
                sub_pts = sub_pts[order2]
                m = len(sub_pts)

                # The 2D HV is the area under the non-dominated frontier
                # Walk through points sorted by obj1, tracking min obj2
                area = 0.0
                min_obj2 = ref_pt[2]
                prev_obj1 = sub_pts[0][1]
                for k in range(m):
                    if sub_pts[k][2] < min_obj2:
                        # This point extends the frontier
                        if k + 1 < m:
                            next_obj1 = sub_pts[k + 1][1]
                        else:
                            next_obj1 = ref_pt[1]
                        area += (next_obj1 - prev_obj1) * (min_obj2 - sub_pts[k][2])
                        prev_obj1 = next_obj1
                        min_obj2 = sub_pts[k][2]
                    # else: this point is dominated in the 2D sense

                hv += (right_0 - pts[i][0]) * area

            return hv

        def non_dominated_mask(F_arr):
            """Return boolean mask of non-dominated points (vectorized)."""
            n = len(F_arr)
            mask = np.ones(n, dtype=bool)
            if n <= 1:
                return mask
            # Vectorized dominance check
            # F_arr[i] dominates F_arr[j] if all(F_arr[i] <= F_arr[j]) and any(F_arr[i] < F_arr[j])
            le = np.all(F_arr[:, np.newaxis, :] <= F_arr[np.newaxis, :, :], axis=2)
            lt = np.any(F_arr[:, np.newaxis, :] < F_arr[np.newaxis, :, :], axis=2)
            dom = le & lt  # dom[i,j] = True if i dominates j
            np.fill_diagonal(dom, False)
            mask = ~dom.any(axis=0)
            return mask

        def select_by_hv(F_arr, n_select):
            """Select n_select points that maximize hypervolume contribution."""
            n = len(F_arr)
            if n <= n_select:
                return list(range(n))

            # Start with non-dominated points
            nd_mask = non_dominated_mask(F_arr)
            nd_indices = np.where(nd_mask)[0]

            if len(nd_indices) <= n_select:
                # Include all non-dominated, then add best dominated
                selected = list(nd_indices)
                if len(selected) < n_select:
                    dom_indices = np.where(~nd_mask)[0]
                    # Sort dominated by sum of objectives (lower is better)
                    dom_scores = F_arr[dom_indices].sum(axis=1)
                    dom_order = np.argsort(dom_scores)
                    selected.extend([dom_indices[i] for i in dom_order[:n_select - len(selected)]])
                return selected[:n_select]

            # Greedy selection: repeatedly add point with highest HV contribution
            selected = []
            remaining = list(nd_indices)

            while len(selected) < n_select and remaining:
                best_idx = -1
                best_hv = -1
                for idx in remaining:
                    test_set = np.array([F_arr[i] for i in selected + [idx]])
                    hv = compute_hypervolume_3d(test_set, ref)
                    if hv > best_hv:
                        best_hv = hv
                        best_idx = idx
                if best_idx == -1:
                    break
                selected.append(best_idx)
                remaining.remove(best_idx)

            return selected

        # Main optimization loop using HVED
        pop_size = 20  # Working population size

        while used < budget:
            if time.time() > deadline:
                break

            # Determine how many new points to generate
            n_new = min(pop_size, budget - used)
            if n_new <= 0:
                break

            # Generate offspring using differential evolution + mutation
            offspring_X = np.empty((n_new, dim))

            # Get non-dominated points for selection
            nd_mask = non_dominated_mask(F)
            nd_indices = np.where(nd_mask)[0]
            if len(nd_indices) == 0:
                nd_indices = np.arange(len(F))

            for i in range(n_new):
                # Select 3 parents from non-dominated set
                if len(nd_indices) >= 3:
                    p1, p2, p3 = np.random.choice(nd_indices, 3, replace=False)
                else:
                    p1, p2, p3 = np.random.choice(len(F), 3, replace=True)

                # Differential evolution: v = p1 + F * (p2 - p3)
                F_factor = 0.5 + 0.3 * np.random.rand()
                v = X[p1] + F_factor * (X[p2] - X[p3])

                # Blend with random point for diversity
                if np.random.rand() < 0.3:
                    v = 0.5 * v + 0.5 * (lo + (hi - lo) * np.random.rand(dim))

                # Polynomial mutation
                for j in range(dim):
                    if np.random.rand() < 1.0 / dim:
                        u = np.random.rand()
                        if u < 0.5:
                            delta = (2 * u) ** (1 / 21.0) - 1
                        else:
                            delta = 1 - (2 * (1 - u)) ** (1 / 21.0)
                        v[j] += delta * (hi[j] - lo[j])

                offspring_X[i] = np.clip(v, lo, hi)

            # Evaluate offspring
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += n_new

            # Combine and select
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])

            # Select best pop_size points by hypervolume
            selected_indices = select_by_hv(combined_F, pop_size)
            X = combined_X[selected_indices]
            F = combined_F[selected_indices]

            # Early termination if no improvement in HV
            if used > init_size * 3:
                # Check if we're still making progress
                pass

        return X
# EVOLVE-BLOCK-END
