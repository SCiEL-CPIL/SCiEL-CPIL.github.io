"""Multi-objective optimizer for RE34: all-evaluations-count hypervolume strategy."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-guided multi-objective search where every evaluation is recorded."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s kill
        budget = int(self.budget)
        dim = self.dim
        n_obj = self.n_obj

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(42)

        # Reference point for hypervolume
        ref = np.array([1705.0, 12.0, 0.3], dtype=float)

        # Store ALL evaluated points - every evaluation counts
        X_all = []
        F_all = []

        def add_point(x, f):
            X_all.append(np.asarray(x, dtype=float))
            F_all.append(np.asarray(f, dtype=float))

        # ---------- Hypervolume helpers ----------
        def hv2(sub, ref2):
            """2D hypervolume (minimization)."""
            if len(sub) == 0:
                return 0.0
            sub = sub[np.all(sub < ref2, axis=1)]
            if len(sub) == 0:
                return 0.0
            order = np.argsort(sub[:, 0])
            sub = sub[order]
            total = 0.0
            prev_x = -np.inf
            for i in range(len(sub)):
                x = sub[i, 0]
                if x >= ref2[0]:
                    break
                # width along first obj
                w0 = min(x, ref2[0]) - prev_x
                if w0 <= 0:
                    prev_x = x
                    continue
                # best (min) second obj among points from i onward
                ybest = np.min(sub[i:, 1])
                ybest = min(ybest, ref2[1])
                total += w0 * (ref2[1] - ybest)
                prev_x = x
            return total

        def hv3(points, ref3):
            """3D hypervolume (minimization)."""
            if not points:
                return 0.0
            P = np.asarray(points, dtype=float)
            P = P[np.all(P < ref3, axis=1)]
            if len(P) == 0:
                return 0.0
            order = np.argsort(P[:, 0])
            P = P[order]
            total = 0.0
            prev_x = -np.inf
            for i in range(len(P)):
                x = P[i, 0]
                if x >= ref3[0]:
                    break
                w0 = min(x, ref3[0]) - prev_x
                if w0 <= 0:
                    prev_x = x
                    continue
                sub = P[i:, 1:]
                total += w0 * hv2(sub, ref3[1:])
                prev_x = x
            return total

        def hv_contribution(f_new, F_existing):
            """Compute additional hypervolume contributed by f_new given existing points."""
            # HV with f_new added
            all_points = np.vstack([np.asarray(F_existing), f_new.reshape(1, -1)])
            hv_with = hv3(all_points, ref)
            hv_without = hv3(F_existing, ref)
            return max(0.0, hv_with - hv_without)

        def pareto_mask(F):
            """Boolean mask: True where point is NOT dominated."""
            n = len(F)
            keep = np.ones(n, dtype=bool)
            for i in range(n):
                fi = F[i]
                # Check if any other point dominates fi
                better = np.all(F <= fi, axis=1) & np.any(F < fi, axis=1)
                better[i] = False
                if np.any(better):
                    keep[i] = False
            return keep

        # ---------- Main search ----------
        # Initial diverse spread
        n_init = min(10, budget)
        for _ in range(n_init):
            x = lo + (hi - lo) * rng.random(dim)
            x = np.clip(x, lo, hi)
            f = problem(x)
            add_point(x, f)

        used = len(F_all)
        scale = 0.3  # adaptive perturbation scale

        while used < budget:
            if time.time() > deadline:
                break

            F_arr = np.array(F_all)
            n_pts = len(F_arr)

            # Identify non-dominated points
            nd_mask = pareto_mask(F_arr)
            nd_indices = np.where(nd_mask)[0]

            if len(nd_indices) == 0:
                nd_indices = np.arange(n_pts)

            # Select parent: weight by hypervolume contribution potential
            # Prefer points with good HV contribution and low objective sum
            nd_F = F_arr[nd_indices]
            sums = np.sum(nd_F, axis=1)
            # Normalize sums for weighting
            s_min, s_max = sums.min(), sums.max()
            if s_max - s_min > 1e-9:
                norm_sums = (sums - s_min) / (s_max - s_min)
            else:
                norm_sums = np.zeros(len(sums))
            # Lower sum = higher weight
            weights = 1.0 - norm_sums
            weights = np.maximum(weights, 0.1)  # avoid zero weight
            weights /= weights.sum()

            # Select parent
            pidx = nd_indices[rng.choice(len(nd_indices), p=weights)]

            # Adaptive mutation
            if rng.random() < 0.1:
                step = scale * 1.5  # occasional larger jump
            else:
                step = scale

            delta = step * (hi - lo) * rng.standard_normal(dim)
            child = X_all[pidx] + delta
            child = np.clip(child, lo, hi)

            # Evaluate
            fc = problem(child)
            add_point(child, fc)
            used += 1

            # Adapt scale: shrink to refine, but maintain some exploration
            scale = max(0.05, scale * 0.995)

            # Periodic exploration boost
            if used % 20 == 0 and scale < 0.15:
                scale = 0.15

        return X_all
# EVOLVE-BLOCK-END
