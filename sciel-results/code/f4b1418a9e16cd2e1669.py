"""MOEA/D-style multi-objective optimizer for RE37 with adaptive refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using MOEA/D with adaptive local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref_point = np.array([1.1, 1.3, 1.1])

        used = 0

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def hv_2d(pts, ref_y, ref_z):
            if len(pts) == 0:
                return 0.0
            pts = np.array(pts)
            mask = (pts[:, 0] < ref_y) & (pts[:, 1] < ref_z)
            pts = pts[mask]
            if len(pts) == 0:
                return 0.0
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            n_pts = len(pts)
            hv2 = 0.0
            for i in range(n_pts):
                y_next = pts[i + 1, 0] if i + 1 < n_pts else ref_y
                if i + 1 < n_pts:
                    min_z_remaining = np.min(pts[i + 1:, 1])
                else:
                    min_z_remaining = ref_z
                if min_z_remaining > pts[i, 1]:
                    w_y = y_next - pts[i, 0]
                    w_z = min_z_remaining - pts[i, 1]
                    if w_y > 0 and w_z > 0:
                        hv2 += w_y * w_z
            return hv2

        def hv_3d(points, ref):
            if len(points) == 0:
                return 0.0
            P = np.array(points)
            mask = np.all(P < ref, axis=1)
            P = P[mask]
            if len(P) == 0:
                return 0.0
            order = np.argsort(P[:, 0])
            P_sorted = P[order]
            n_pts = len(P_sorted)
            hv = 0.0
            for i in range(n_pts):
                f0_min = P_sorted[i, 0]
                f0_max = P_sorted[i + 1, 0] if i + 1 < n_pts else ref[0]
                if f0_max <= f0_min:
                    continue
                mask_slice = (P_sorted[:, 0] >= f0_min) & (P_sorted[:, 0] < f0_max)
                slice_pts = P_sorted[mask_slice, 1:3]
                hv_slice = hv_2d(slice_pts, ref[1], ref[2])
                hv += hv_slice * (f0_max - f0_min)
            return hv

        def marginal_hv(point, archive_F):
            if not np.all(point < ref_point):
                return 0.0
            if len(archive_F) == 0:
                return hv_3d([point], ref_point)
            for f in archive_F:
                if dominates(f, point):
                    return 0.0
            hv_before = hv_3d(archive_F, ref_point)
            archive_new = list(archive_F) + [point]
            hv_after = hv_3d(archive_new, ref_point)
            return hv_after - hv_before

        def update_archive(x, f, archive_X, archive_F):
            new_archive_X = []
            new_archive_F = []
            for ax, af in zip(archive_X, archive_F):
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive_X.append(ax)
                    new_archive_F.append(af)
            dominated = False
            for af in new_archive_F:
                if dominates(af, f):
                    dominated = True
                    break
            if not dominated:
                new_archive_X.append(x)
                new_archive_F.append(f)
            return new_archive_X, new_archive_F

        # --- Phase 1: Latin Hypercube Sampling for initial coverage ---
        archive_X = []
        archive_F = []

        n_lhs = 40
        for i in range(n_lhs):
            if used >= budget:
                break
            if time.time() > deadline - 30:
                break
            x = np.array([(rng.random()) for _ in range(dim)])
            x = lo + x * span
            try:
                f = problem(x)
                f = np.array(f, dtype=float)
            except Exception:
                continue
            mhv = marginal_hv(f, archive_F)
            if mhv > 1e-10:
                archive_X, archive_F = update_archive(x, f, archive_X, archive_F)
                used += 1

        # --- Phase 2: MOEA/D-style weight-based local search ---
        # Generate weight vectors on the simplex
        n_weights = 15
        weight_vectors = []
        for i in range(n_obj):
            w = np.zeros(n_obj)
            w[i] = 1.0
            weight_vectors.append(w)
        for i in range(n_obj):
            for j in range(n_obj):
                if i < j:
                    w = np.zeros(n_obj)
                    w[i] = 0.5
                    w[j] = 0.5
                    weight_vectors.append(w)
        while len(weight_vectors) < n_weights:
            w = rng.random(n_obj)
            w = w / w.sum()
            weight_vectors.append(w)

        # For each weight vector, do a local search from the best archive point
        for w in weight_vectors:
            if used >= budget:
                break
            if time.time() > deadline - 20:
                break

            # Find the archive point closest to the weight direction
            if len(archive_F) > 0:
                # Normalize objectives to [0,1] for distance calculation
                f_min = np.min(archive_F, axis=0)
                f_max = np.max(archive_F, axis=0)
                f_range = f_max - f_min
                f_range[f_range < 1e-10] = 1.0
                f_norm = (np.array(archive_F) - f_min) / f_range
                distances = np.sum((f_norm - w) ** 2, axis=1)
                best_idx = int(np.argmin(distances))
                start_x = archive_X[best_idx].copy()
            else:
                start_x = lo + span * 0.5

            # Nelder-Mead local search with Tchebycheff scalarization
            def tchebycheff_scalar(x):
                try:
                    f_val = problem(x)
                    f_val = np.array(f_val, dtype=float)
                    # Tchebycheff: max_i w_i * |f_i - z_i^*|
                    # Use current best as z*
                    if len(archive_F) > 0:
                        z_star = np.min(archive_F, axis=0)
                    else:
                        z_star = np.array([0.5, 0.5, 0.5])
                    return np.max(w * np.abs(f_val - z_star))
                except Exception:
                    return 1e10

            # Simple gradient-free local search
            step_size = 0.1
            current_x = start_x.copy()
            try:
                current_f = problem(current_x)
                current_f = np.array(current_f, dtype=float)
                current_score = tchebycheff_scalar(current_x)
            except Exception:
                continue

            improved = False
            for step_iter in range(30):
                if used >= budget:
                    break
                if time.time() > deadline - 5:
                    break

                # Try perturbations in each dimension
                best_neighbor_x = None
                best_neighbor_score = current_score

                for d in range(dim):
                    for direction in [-1, 1]:
                        neighbor_x = current_x.copy()
                        neighbor_x[d] += direction * step_size
                        neighbor_x = np.clip(neighbor_x, lo, hi)

                        try:
                            neighbor_score = tchebycheff_scalar(neighbor_x)
                        except Exception:
                            continue

                        if neighbor_score < best_neighbor_score - 1e-8:
                            best_neighbor_score = neighbor_score
                            best_neighbor_x = neighbor_x

                if best_neighbor_x is not None:
                    current_x = best_neighbor_x
                    current_score = best_neighbor_score
                    improved = True
                    step_size *= 0.8
                else:
                    step_size *= 0.5
                    if step_size < 1e-4:
                        break

            # Evaluate the final point
            try:
                f_opt = problem(current_x)
                f_opt = np.array(f_opt, dtype=float)
            except Exception:
                continue

            mhv = marginal_hv(f_opt, archive_F)
            if mhv > 1e-10:
                archive_X, archive_F = update_archive(current_x, f_opt, archive_X, archive_F)
                used += 1

        # --- Phase 3: Adaptive multi-objective hill polish ---
        # Accept sideways non-dominated moves to spread the front
        if len(archive_F) > 0:
            polish_step = 0.02
            n_polish = min(40, budget - used)
            for _ in range(n_polish):
                if used >= budget:
                    break
                if time.time() > deadline - 2:
                    break

                if len(archive_X) > 0:
                    # Select archive point with best marginal HV
                    hv_contribs = []
                    for i, f in enumerate(archive_F):
                        archive_without = [archive_F[j] for j in range(len(archive_F)) if j != i]
                        hv_without = hv_3d(archive_without, ref_point)
                        hv_with = hv_3d(archive_F, ref_point)
                        hv_contribs.append(hv_with - hv_without)
                    best_idx = int(np.argmax(hv_contribs))
                    parent_x = archive_X[best_idx]
                else:
                    parent_x = lo + span * 0.5

                noise = polish_step * span * rng.standard_normal(dim)
                child_x = np.clip(parent_x + noise, lo, hi)
                try:
                    child_f = problem(child_x)
                    child_f = np.array(child_f, dtype=float)
                except Exception:
                    continue

                mhv = marginal_hv(child_f, archive_F)
                if mhv > 1e-10:
                    archive_X, archive_F = update_archive(child_x, child_f, archive_X, archive_F)
                    used += 1

        # --- Phase 4: Final micro-refinement ---
        if used < budget and time.time() < deadline - 1:
            polish_step = 0.005
            n_polish = min(20, budget - used)
            for _ in range(n_polish):
                if time.time() > deadline - 0.5:
                    break
                if used >= budget:
                    break

                if len(archive_X) > 0:
                    parent_x = archive_X[rng.integers(len(archive_X))]
                else:
                    parent_x = lo + span * 0.5

                noise = polish_step * span * rng.standard_normal(dim)
                child_x = np.clip(parent_x + noise, lo, hi)
                try:
                    child_f = problem(child_x)
                    child_f = np.array(child_f, dtype=float)
                except Exception:
                    continue

                mhv = marginal_hv(child_f, archive_F)
                if mhv > 1e-10:
                    archive_X, archive_F = update_archive(child_x, child_f, archive_X, archive_F)
                    used += 1

        if archive_X:
            return np.array(archive_X)
        else:
            return np.array([lo + span * 0.5])
# EVOLVE-BLOCK-END
