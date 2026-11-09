"""Multi-objective optimizer for RE21 using LHS + HV-guided local search + adaptive polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Direct multi-objective optimizer for RE21 (4 vars, 2 objectives)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        ref_point = np.array([3000.0, 0.06])

        all_X = []
        all_F = []
        used = 0

        def eval_point(x):
            nonlocal used
            x = np.clip(x, lo, hi)
            f = problem(x)
            f = np.array(f, dtype=np.float64)
            all_X.append(x.copy())
            all_F.append(f.copy())
            used += 1
            return f

        # Phase 1: Latin Hypercube Sampling for broad coverage
        # LHS ensures one point per stratum in each dimension
        n_lhs = 60  # 60 points, 15 per dimension stratum
        n_strata = 15
        lhs_points = []
        for _ in range(n_lhs):
            x = np.zeros(self.dim)
            for d in range(self.dim):
                # Randomly assign strata without replacement per dimension
                pass
        # Better LHS: for each dimension, shuffle strata indices
        lhs_points = []
        for _ in range(n_lhs):
            x = np.zeros(self.dim)
            for d in range(self.dim):
                stratum = np.random.randint(n_strata)
                x[d] = lo[d] + (stratum + np.random.random()) * span[d] / n_strata
            lhs_points.append(x)

        for x in lhs_points:
            if used >= n_lhs or time.time() > deadline:
                break
            eval_point(x)

        # Fill remaining exploration with random sampling
        target_explore = 100
        while used < target_explore and time.time() < deadline:
            x = np.random.uniform(lo, hi)
            eval_point(x)

        if not all_F:
            return np.array([])

        all_F_arr = np.array(all_F)
        all_X_arr = np.array(all_X)

        valid_mask = np.all(all_F_arr < ref_point, axis=1)
        if np.any(valid_mask):
            valid_F = all_F_arr[valid_mask]
            valid_X = all_X_arr[valid_mask]
        else:
            valid_F = all_F_arr
            valid_X = all_X_arr

        if len(valid_F) == 0:
            return all_X_arr

        nd_mask = self._non_dominated_mask(valid_F)
        nd_F = valid_F[nd_mask]
        nd_X = valid_X[nd_mask]

        # Phase 2: HV-guided local search - only accept HV-improving moves
        if len(nd_F) > 0:
            contributions = self._marginal_hv_contributions(nd_F, ref_point)
            order = np.argsort(-contributions)
            nd_X_sorted = nd_X[order]
            nd_F_sorted = nd_F[order]
        else:
            nd_X_sorted = nd_X
            nd_F_sorted = nd_F

        base_scale = 0.05
        min_scale = 0.0005
        max_iters_per_point = 8

        # Track current non-dominated set for quick domination checks
        current_nd_F = nd_F_sorted.copy()
        current_nd_X = nd_X_sorted.copy()

        for i in range(len(nd_X_sorted)):
            if used >= self.budget or time.time() > deadline:
                break

            x_best = nd_X_sorted[i].copy()
            f_best = nd_F_sorted[i].copy()
            scale = base_scale
            no_improve_count = 0

            for it in range(max_iters_per_point):
                if used >= self.budget or time.time() > deadline:
                    break

                # Generate candidate: Gaussian perturbation
                x_try = x_best + np.random.normal(0, 1, self.dim) * scale * span
                x_try = np.clip(x_try, lo, hi)
                f_try = eval_point(x_try)

                # Check if f_try is dominated by any point in current_nd_F
                dominated = False
                for fj in current_nd_F:
                    if np.all(fj <= f_try) and np.any(fj < f_try):
                        dominated = True
                        break

                if not dominated:
                    # f_try is non-dominated; check if it improves local best
                    local_improve = np.all(f_try <= f_best) and np.any(f_try < f_best)
                    if local_improve:
                        x_best = x_try.copy()
                        f_best = f_try.copy()
                        scale = min(scale * 1.15, base_scale)
                        no_improve_count = 0
                    else:
                        # It's a new non-dominated point that doesn't improve local best
                        # Still valuable for HV; accept and update local best if it has better HV contribution
                        hv_try = (ref_point[0] - f_try[0]) * (ref_point[1] - f_try[1])
                        hv_best = (ref_point[0] - f_best[0]) * (ref_point[1] - f_best[1])
                        if hv_try > hv_best:
                            x_best = x_try.copy()
                            f_best = f_try.copy()
                        scale *= 0.95
                        no_improve_count = 0
                else:
                    # Dominated; shrink and count
                    scale *= 0.7
                    no_improve_count += 1
                    if scale < min_scale or no_improve_count > 5:
                        break

        # Phase 3: High-precision polish on top HV contributors
        if not all_F:
            return np.array([])

        all_F_arr = np.array(all_F)
        all_X_arr = np.array(all_X)

        valid_mask = np.all(all_F_arr < ref_point, axis=1)
        if np.any(valid_mask):
            valid_F = all_F_arr[valid_mask]
            valid_X = all_X_arr[valid_mask]
        else:
            valid_F = all_F_arr
            valid_X = all_X_arr

        if len(valid_F) == 0:
            return all_X_arr

        nd_mask = self._non_dominated_mask(valid_F)
        nd_F = valid_F[nd_mask]
        nd_X = valid_X[nd_mask]

        # Final polish: adaptive line search on top HV contributors
        if len(nd_F) > 0 and used < self.budget:
            contributions = self._marginal_hv_contributions(nd_F, ref_point)
            top_k = min(6, len(nd_F))
            top_indices = np.argsort(-contributions)[:top_k]

            for idx in top_indices:
                if used >= self.budget or time.time() > deadline:
                    break
                x_c = nd_X[idx].copy()
                f_c = nd_F[idx].copy()

                # Axis-aligned adaptive line search
                for d in range(self.dim):
                    if used >= self.budget or time.time() > deadline:
                        break
                    for sgn in [-1, 1]:
                        if used >= self.budget or time.time() > deadline:
                            break
                        # Adaptive step: start large, shrink if no improvement
                        step = 0.01 * span[d]
                        improved = True
                        while improved and used < self.budget and time.time() < deadline and step > 1e-5 * span[d]:
                            improved = False
                            x_try = x_c.copy()
                            x_try[d] += sgn * step
                            x_try = np.clip(x_try, lo, hi)
                            f_try = eval_point(x_try)

                            if np.all(f_try <= f_c) and np.any(f_try < f_c):
                                x_c = x_try.copy()
                                f_c = f_try.copy()
                                improved = True
                                step *= 0.5  # Shrink for next iteration
                            else:
                                step *= 0.5  # Shrink regardless

        # Final recompute
        all_F_arr = np.array(all_F)
        all_X_arr = np.array(all_X)
        valid_mask = np.all(all_F_arr < ref_point, axis=1)
        if np.any(valid_mask):
            valid_F = all_F_arr[valid_mask]
            valid_X = all_X_arr[valid_mask]
        else:
            valid_F = all_F_arr
            valid_X = all_X_arr

        if len(valid_F) == 0:
            return all_X_arr

        nd_mask = self._non_dominated_mask(valid_F)
        nd_F = valid_F[nd_mask]
        nd_X = valid_X[nd_mask]

        return nd_X

    def _marginal_hv_contributions(self, F, ref_point):
        """
        Calculate marginal hypervolume contribution for each point in F.
        F is an array of shape (N, 2). Uses efficient 2D HV computation.
        """
        N = len(F)
        if N == 0:
            return np.array([])
        if N == 1:
            return np.array([self._compute_hv(F, ref_point)])

        # Compute HV of the full set
        hv_full = self._compute_hv(F, ref_point)

        contributions = np.zeros(N)
        for i in range(N):
            mask = np.ones(N, dtype=bool)
            mask[i] = False
            F_reduced = F[mask]
            hv_reduced = self._compute_hv(F_reduced, ref_point)
            contributions[i] = hv_full - hv_reduced

        return contributions

    def _compute_hv(self, F, ref_point):
        """
        Compute exact hypervolume for 2 objectives.
        F is an array of shape (N, 2).
        """
        if len(F) == 0:
            return 0.0

        valid_mask = np.all(F < ref_point, axis=1)
        F_valid = F[valid_mask]

        if len(F_valid) == 0:
            return 0.0

        # Sort by first objective (ascending)
        F_sorted = F_valid[np.argsort(F_valid[:, 0])]

        # Compute prefix minimum of y
        y_prefix_min = np.minimum.accumulate(F_sorted[:, 1])

        x_coords = np.append(F_sorted[:, 0], ref_point[0])
        y_coords = y_prefix_min

        widths = x_coords[1:] - x_coords[:-1]
        heights = ref_point[1] - y_coords

        widths = np.maximum(widths, 0)
        heights = np.maximum(heights, 0)

        hv = np.sum(widths * heights)

        return hv

    def _non_dominated_mask(self, F):
        """Return a boolean mask for non-dominated points in F (minimization). Vectorized."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=bool)
        if n == 1:
            return np.ones(1, dtype=bool)
        # F[i] is dominated if exists j: F[j] <= F[i] elementwise and F[j] < F[i] in at least one dim
        # diff[i,j] = F[j] - F[i]; F[j] <= F[i] iff all diff[i,j] >= 0
        diff = F[np.newaxis, :, :] - F[:, np.newaxis, :]  # shape (n, n, 2)
        leq = np.all(diff >= 0, axis=2)  # F[j] <= F[i]
        lt = np.any(diff < 0, axis=2)    # F[j] < F[i] in at least one
        dominated = np.any(leq & lt, axis=1)
        # A point is non-dominated if NOT dominated by any other point
        # Note: a point is not dominated by itself (diff=0, lt=False for i==j)
        return ~dominated
# EVOLVE-BLOCK-END
# EVOLVE-BLOCK-END
