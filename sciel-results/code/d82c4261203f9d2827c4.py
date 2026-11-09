"""Multi-objective optimizer for RE34 using HV-guided search with proper 3D HV."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using HV-guided search with adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _hv_3d(self, F, ref):
        """Compute exact hypervolume for 3D points strictly below ref."""
        valid = np.all(F < ref, axis=1)
        if not np.any(valid):
            return 0.0
        Fv = F[valid]
        n = len(Fv)
        if n == 0:
            return 0.0

        # Sort by f1 ascending
        order = np.argsort(Fv[:, 0])
        Fv = Fv[order]
        n = len(Fv)

        # Remove dominated points (keep Pareto front)
        # After sorting by f1, a point is dominated if there's an earlier point
        # with f2 <= current f2 AND f3 <= current f3
        keep = np.ones(n, dtype=bool)
        for i in range(n):
            if not keep[i]:
                continue
            for j in range(i + 1, n):
                if Fv[j, 1] >= Fv[i, 1] and Fv[j, 2] >= Fv[i, 2]:
                    keep[j] = False
        Fv = Fv[keep]
        n = len(Fv)
        if n == 0:
            return 0.0
        if n == 1:
            return float(np.prod(np.maximum(0, ref - Fv[0])))

        # Sort by f1 ascending
        Fv = Fv[np.argsort(Fv[:, 0])]
        n = len(Fv)

        # For 3D HV, use the standard algorithm:
        # Sort by f1. For each point i, compute the contribution:
        # (f1_next - f1_i) * 2D_HV of the "skyline" in (f2, f3) from points 0..i
        # where f1_next = min(f1[i+1], ref[0]) if i+1 < n else ref[0]

        hv = 0.0
        # We maintain the set of (f2, f3) pairs that form the skyline
        # Process points in order of increasing f1
        # For each point, we need to compute the 2D HV of the current skyline

        # Build skyline incrementally
        # skyline is a list of (f2, f3) pairs, sorted by f2 ascending
        # where f3 is strictly decreasing (skyline property)
        skyline = []  # list of [f2, f3]

        for i in range(n):
            f1_i = Fv[i, 0]
            f1_next = Fv[i + 1, 0] if i + 1 < n else ref[0]
            f1_next = min(f1_next, ref[0])

            if f1_next <= f1_i:
                continue

            # Add current point to skyline
            f2_i = Fv[i, 1]
            f3_i = Fv[i, 2]

            # Remove points from skyline that are dominated by (f2_i, f3_i)
            # A skyline point (a, b) is dominated if a >= f2_i and b >= f3_i
            new_skyline = []
            for pt in skyline:
                if not (pt[0] >= f2_i and pt[1] >= f3_i):
                    new_skyline.append(pt)
            new_skyline.append([f2_i, f3_i])
            # Remove points dominated by the new point (already done above)
            # Sort by f2 ascending, and ensure f3 is strictly decreasing
            new_skyline.sort(key=lambda x: x[0])
            # Now remove any point that has f3 >= previous point's f3
            final_skyline = []
            for pt in new_skyline:
                if not final_skyline or pt[1] < final_skyline[-1][1]:
                    final_skyline.append(pt)
            skyline = final_skyline

            # Compute 2D HV of skyline with reference (ref[1], ref[2])
            # The skyline points are sorted by f2 ascending, f3 strictly decreasing
            # 2D HV = sum over skyline points of (f2_next - f2_i) * (ref[2] - f3_i)
            # where f2_next = min(next point's f2, ref[1]) or ref[1] for last
            hv_2d = 0.0
            for j in range(len(skyline)):
                f2_j = skyline[j][0]
                f3_j = skyline[j][1]
                if f2_j >= ref[1] or f3_j >= ref[2]:
                    continue
                f2_next = skyline[j + 1][0] if j + 1 < len(skyline) else ref[1]
                f2_next = min(f2_next, ref[1])
                if f2_next <= f2_j:
                    continue
                hv_2d += (f2_next - f2_j) * (ref[2] - f3_j)

            hv += (f1_next - f1_i) * hv_2d

        return float(hv)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        ref = np.array([1705.0, 12.0, 0.3])

        # Initial random sampling
        n_init = min(30, self.budget // 5)
        X_list = []
        F_list = []

        for i in range(n_init):
            x = lo + span * rng.random(self.dim)
            f = np.array(problem(x))
            X_list.append(x)
            F_list.append(f)

        used = n_init
        step_size = 0.2
        best_hv = 0.0
        hv_stagnation = 0
        last_hv_update = 0

        while used < self.budget:
            if time.time() > deadline:
                break

            F_arr = np.array(F_list)
            n = len(F_arr)

            # Compute HV-guided parent selection
            valid = np.all(F_arr < ref, axis=1)
            if np.sum(valid) == 0:
                parent_idx = rng.integers(n)
            else:
                Fv = F_arr[valid]
                valid_indices = np.where(valid)[0]
                nv = len(Fv)

                if nv == 1:
                    contrib = np.zeros(n)
                    contrib[valid_indices[0]] = np.prod(np.maximum(0, ref - Fv[0]))
                else:
                    # Compute approximate contribution for each point
                    # Use box volume as a proxy, weighted by novelty
                    box_vols = np.prod(np.maximum(0, ref - Fv), axis=1)

                    # Compute novelty: distance to nearest neighbor in objective space
                    k = min(2, nv - 1)
                    novelty = np.zeros(nv)
                    for i in range(nv):
                        dists = np.linalg.norm(Fv - Fv[i], axis=1)
                        dists[i] = np.inf
                        if k < len(dists):
                            nearest = np.partition(dists, k)[k]
                        else:
                            nearest = np.max(dists)
                        novelty[i] = 1.0 / (1.0 + nearest * 0.1)

                    contrib = np.zeros(n)
                    contrib[valid_indices] = box_vols * novelty

                if np.max(contrib) > 0:
                    top_k = min(5, int(np.sum(contrib > 0)))
                    if top_k > 0:
                        valid_idx = np.where(contrib > 0)[0]
                        top_idx = valid_idx[np.argsort(contrib[valid_idx])][-top_k:]
                        w = contrib[top_idx]
                        w = w / w.sum()
                        parent_idx = rng.choice(top_idx, p=w)
                    else:
                        parent_idx = rng.integers(n)
                else:
                    parent_idx = rng.integers(n)

            parent = X_list[parent_idx]
            child = parent + step_size * span * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            fc = np.array(problem(child))
            X_list.append(child)
            F_list.append(fc)
            used += 1

            # Adapt step size based on whether new point is non-dominated
            if np.all(fc < ref):
                old_F = np.array(F_list[:-1])
                dominated = np.any(np.all(old_F <= fc, axis=1) & np.any(old_F < fc, axis=1))
                if not dominated:
                    step_size = max(0.05, step_size * 0.93)
                else:
                    step_size = min(1.0, step_size * 1.07)
            else:
                step_size = min(1.0, step_size * 1.12)

            # Update HV periodically (every 10 evaluations or at end)
            if used % 10 == 0 or used == self.budget:
                F_new = np.array(F_list)
                current_hv = self._hv_3d(F_new, ref)

                if current_hv > best_hv + 1e-6:
                    best_hv = current_hv
                    hv_stagnation = 0
                else:
                    hv_stagnation += 1

            # Random restart if HV stagnates
            if hv_stagnation > 5 and used < self.budget - 5:
                x = lo + span * rng.random(self.dim)
                f = np.array(problem(x))
                X_list.append(x)
                F_list.append(f)
                used += 1
                hv_stagnation = 0
                step_size = 0.3

        return X_list
# EVOLVE-BLOCK-END
