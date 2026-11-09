"""Multi-objective optimizer for RE34: hypervolume-guided search with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-guided multi-objective search with adaptive perturbation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0          # safety margin before 300s kill
        budget = int(self.budget)
        dim = self.dim
        n_obj = self.n_obj

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(42)

        # Reference point for hypervolume
        ref = np.array([1705.0, 12.0, 0.3], dtype=float)

        # Store ALL evaluated points
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
                w0 = min(x, ref2[0]) - prev_x
                if w0 <= 0:
                    prev_x = x
                    continue
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

        def pareto_mask(F):
            """Boolean mask: True where point is NOT dominated."""
            n = len(F)
            keep = np.ones(n, dtype=bool)
            for i in range(n):
                fi = F[i]
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
            nd_F = F_arr[nd_indices]
            sums = np.sum(nd_F, axis=1)
            s_min, s_max = sums.min(), sums.max()
            if s_max - s_min > 1e-9:
                norm_sums = (sums - s_min) / (s_max - s_min)
            else:
                norm_sums = np.zeros(len(sums))
            weights = 1.0 - norm_sums
            weights = np.maximum(weights, 0.1)
            weights /= weights.sum()

            pidx = nd_indices[rng.choice(len(nd_indices), p=weights)]

            # Adaptive mutation
            if rng.random() < 0.1:
                step = scale * 1.5
            else:
                step = scale

            delta = step * (hi - lo) * rng.standard_normal(dim)
            child = X_all[pidx] + delta
            child = np.clip(child, lo, hi)

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
