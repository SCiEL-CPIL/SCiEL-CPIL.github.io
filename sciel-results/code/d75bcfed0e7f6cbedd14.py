"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-objective optimizer: LHS init, hypervolume-guided search with adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _lhs(self, n, dim, lo, hi):
        """Latin hypercube sampling for good initial spread."""
        pts = np.zeros((n, dim))
        for d in range(dim):
            perm = np.random.permutation(n)
            u = (perm + np.random.random(n)) / n
            pts[:, d] = lo[d] + (hi[d] - lo[d]) * u
        return pts

    def _non_dominated(self, F):
        """Return array of non-dominated points."""
        n = len(F)
        if n == 0:
            return F
        keep = np.ones(n, dtype=bool)
        for i in range(n):
            if not keep[i]:
                continue
            for j in range(n):
                if i == j or not keep[j]:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    keep[i] = False
                    break
        return F[keep]

    def _hv_2d_in_box(self, cand, lo1, lo2, hi1, hi2):
        """2D HV of cand points within box [lo1,hi1]x[lo2,hi2], where cand points are lower-left corners."""
        if len(cand) == 0:
            return 0.0
        order = np.argsort(cand[:, 0])
        cand = cand[order]
        total_area = 0.0
        n = len(cand)
        xs = list(cand[:, 0]) + [hi1]
        min_y = hi2
        for k in range(n):
            if cand[k, 1] < min_y:
                min_y = cand[k, 1]
            seg_w = xs[k + 1] - xs[k]
            if seg_w > 0 and min_y < hi2:
                total_area += seg_w * (hi2 - min_y)
        return total_area

    def _hv_3d(self, f, ref):
        """Exact 3D hypervolume for a set of points vs ref."""
        valid = np.all(f < ref, axis=1)
        pts = f[valid]
        if len(pts) == 0:
            return 0.0
        pts = self._non_dominated(pts)
        order = np.argsort(pts[:, 0])
        pts = pts[order]
        total = 0.0
        n = len(pts)
        for i in range(n):
            next0 = pts[i + 1, 0] if i + 1 < n else ref[0]
            w0 = next0 - pts[i, 0]
            if w0 <= 0:
                continue
            later = pts[i + 1:]
            if len(later) == 0:
                area = (ref[1] - pts[i, 1]) * (ref[2] - pts[i, 2])
            else:
                cand = later[(later[:, 1] <= pts[i, 1]) & (later[:, 2] <= pts[i, 2])]
                if len(cand) == 0:
                    area = (ref[1] - pts[i, 1]) * (ref[2] - pts[i, 2])
                else:
                    area = self._hv_2d_in_box(cand, pts[i, 1], pts[i, 2], ref[1], ref[2])
            total += w0 * area
        return total

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 280.0
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([1705.0, 12.0, 0.3])

        # Initial population via LHS
        pop_size = min(25, max(10, self.budget // 8))
        X = self._lhs(pop_size, self.dim, lo, hi)
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track all evaluated points for final hypervolume
        all_F = [F.copy()]

        def non_dominated_idx(F):
            n = len(F)
            keep = np.ones(n, dtype=bool)
            for i in range(n):
                if not keep[i]:
                    continue
                for j in range(n):
                    if i == j or not keep[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        keep[i] = False
                        break
            return np.where(keep)[0]

        # Adaptive mutation scale
        mut_scale = 0.12
        min_scale = 0.03
        max_scale = 0.35

        # Parent selection: weighted by hypervolume contribution
        while used < self.budget:
            if time.time() > deadline:
                break

            nd_idx = non_dominated_idx(F)
            if len(nd_idx) == 0:
                nd_idx = np.arange(len(F))

            # Compute marginal HV contribution for each non-dominated point
            base_hv = self._hv_3d(F[nd_idx], ref)
            weights = np.zeros(len(nd_idx))
            for ii, idx in enumerate(nd_idx):
                others = np.delete(F[nd_idx], ii)
                hv_without = self._hv_3d(others, ref) if len(others) > 0 else 0.0
                weights[ii] = max(base_hv - hv_without, 1e-10)

            # Normalize to probabilities
            weights = weights / weights.sum()

            # Select parent
            parent_idx = nd_idx[rng.choice(len(nd_idx), p=weights)]
            parent = X[parent_idx]

            # Adaptive mutation: Gaussian with adaptive scale
            sigma = mut_scale * (hi - lo)
            child = parent + rng.standard_normal(self.dim) * sigma
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1
            all_F.append(fc.copy())

            # Replacement strategy
            nd_idx_all = non_dominated_idx(F)
            if len(nd_idx_all) < len(F):
                # Replace a dominated point
                dominated_idx = np.setdiff1d(np.arange(len(F)), nd_idx_all)
                worst = dominated_idx[np.argmax(np.sum(F[dominated_idx], axis=1))]
                X[worst] = child
                F[worst] = fc
            else:
                # All non-dominated; replace the one with lowest marginal HV contribution
                base_hv_all = self._hv_3d(F, ref)
                worst = 0
                worst_contrib = np.inf
                for i in range(len(F)):
                    others = np.delete(F, i)
                    hv_without = self._hv_3d(others, ref) if len(others) > 0 else 0.0
                    contrib = base_hv_all - hv_without
                    if contrib < worst_contrib:
                        worst_contrib = contrib
                        worst = i
                X[worst] = child
                F[worst] = fc

            # Adapt mutation scale
            new_hv = self._hv_3d(F, ref)
            old_hv = self._hv_3d(np.delete(F, worst), ref) if len(F) > 1 else 0.0
            if new_hv > old_hv:
                mut_scale = min(max_scale, mut_scale * 1.03)
            else:
                mut_scale = max(min_scale, mut_scale * 0.97)

        # Combine all evaluated points for final result
        all_points = np.vstack(all_F)
        return all_points
# EVOLVE-BLOCK-END
