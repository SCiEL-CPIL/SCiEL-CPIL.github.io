"""Combined optimizer: HV-guided structured exploration with Nelder-Mead refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer using structured exploration + Nelder-Mead polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref_point = np.array([1.1, 1.3, 1.1])

        used = 0
        archive_X = []
        archive_F = []

        # --- Helper: dominance ---
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        # --- Helper: 2D hypervolume (for slices in 3D) ---
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

        # --- Helper: 3D hypervolume ---
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

        # --- Helper: marginal HV contribution of a new point ---
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

        # --- Helper: update archive with new point ---
        def update_archive(x, f):
            nonlocal archive_X, archive_F
            # Remove points dominated by new point
            new_archive_X = []
            new_archive_F = []
            for ax, af in zip(archive_X, archive_F):
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive_X.append(ax)
                    new_archive_F.append(af)
            # Add new point if not dominated by existing
            dominated = False
            for af in new_archive_F:
                if dominates(af, f):
                    dominated = True
                    break
            if not dominated:
                new_archive_X.append(x)
                new_archive_F.append(f)
            archive_X = new_archive_X
            archive_F = new_archive_F

        # --- Phase 1: Structured weight-grid exploration ---
        # Use a set of weight vectors to explore the objective space systematically
        # Generate weight vectors on a simplex
        n_weights = 20
        weight_vectors = []
        # Include extreme points
        for i in range(n_obj):
            w = np.zeros(n_obj)
            w[i] = 1.0
            weight_vectors.append(w)
        # Include midpoints
        for i in range(n_obj):
            for j in range(n_obj):
                if i < j:
                    w = np.zeros(n_obj)
                    w[i] = 0.5
                    w[j] = 0.5
                    weight_vectors.append(w)
        # Fill remaining with random weights
        while len(weight_vectors) < n_weights:
            w = rng.random(n_obj)
            w = w / w.sum()
            weight_vectors.append(w)

        # For each weight vector, generate a candidate point
        # We'll evaluate candidates greedily by marginal HV
        candidates = []
        for w in weight_vectors:
            # Generate multiple candidates per weight vector
            for _ in range(3):
                # Start from a random point and project toward the weight direction
                x = rng.random(dim)
                # Blend with a point that emphasizes the weighted objectives
                # Since we don't know the objective landscape, use structured sampling
                x_candidate = lo + x * span
                candidates.append(x_candidate)

        # Add some Latin Hypercube samples for diversity
        n_lhs = 30
        for d in range(dim):
            perm = rng.permutation(n_lhs)
            for i in range(n_lhs):
                if len(candidates) >= n_lhs + n_weights * 3:
                    break
        # Simpler: just add LHS points
        for i in range(n_lhs):
            x = np.array([(perm_d + rng.random()) / n_lhs for perm_d in [rng.permutation(n_lhs)[i] for _ in range(dim)]])
            x = lo + x * span
            candidates.append(x)

        # Shuffle candidates
        rng.shuffle(candidates)

        # Evaluate candidates greedily by marginal HV
        for x in candidates:
            if used >= budget:
                break
            if time.time() > deadline - 30:
                break

            try:
                f = problem(x)
            except Exception:
                continue

            f = np.array(f, dtype=float)

            # Only add if it contributes to HV
            mhv = marginal_hv(f, archive_F)
            if mhv > 1e-10:
                update_archive(x, f)
                used += 1

        # --- Phase 2: Nelder-Mead refinement around archive points ---
        # Select top archive points by marginal HV contribution
        if len(archive_F) > 0:
            # Compute marginal HV for each archive point (remove and re-add)
            hv_contribs = []
            for i, f in enumerate(archive_F):
                # Compute HV without this point
                archive_without = [archive_F[j] for j in range(len(archive_F)) if j != i]
                hv_without = hv_3d(archive_without, ref_point)
                hv_with = hv_3d(archive_F, ref_point)
                hv_contribs.append(hv_with - hv_without)

            # Sort by HV contribution (descending)
            order = np.argsort(hv_contribs)[::-1]
            n_refine = min(5, len(archive_F))

            for idx in order[:n_refine]:
                if used >= budget:
                    break
                if time.time() > deadline - 10:
                    break

                center_x = archive_X[idx]
                center_f = archive_F[idx]

                # Nelder-Mead on a scalarized objective
                # Use a weighted sum with weights based on the objective values
                # to find a point that improves the front locally

                # Try several weight vectors for local refinement
                local_weights = [
                    np.array([1.0, 0.0, 0.0]),
                    np.array([0.0, 1.0, 0.0]),
                    np.array([0.0, 0.0, 1.0]),
                    np.array([0.5, 0.5, 0.0]),
                    np.array([0.5, 0.0, 0.5]),
                    np.array([0.0, 0.5, 0.5]),
                    np.array([1.0/3, 1.0/3, 1.0/3]),
                ]

                for lw in local_weights:
                    if used >= budget:
                        break
                    if time.time() > deadline - 5:
                        break

                    # Scalarized objective
                    def scalarized(x):
                        try:
                            f_val = problem(x)
                            f_val = np.array(f_val, dtype=float)
                            return np.dot(lw, f_val)
                        except Exception:
                            return 1e10

                    # Nelder-Mead optimization
                    try:
                        result = minimize(
                            scalarized,
                            center_x,
                            method='Nelder-Mead',
                            options={
                                'maxiter': 50,
                                'xatol': 1e-4,
                                'fatol': 1e-6,
                                'adaptive': True
                            }
                        )
                    except Exception:
                        continue

                    x_opt = np.clip(result.x, lo, hi)
                    try:
                        f_opt = problem(x_opt)
                        f_opt = np.array(f_opt, dtype=float)
                    except Exception:
                        continue

                    # Check if this point improves the archive
                    mhv = marginal_hv(f_opt, archive_F)
                    if mhv > 1e-10:
                        update_archive(x_opt, f_opt)
                        used += 1

        # --- Phase 3: Final micro-refinement pass ---
        if used < budget and time.time() < deadline - 2:
            polish_step = 0.01
            n_polish = min(20, budget - used)
            for _ in range(n_polish):
                if time.time() > deadline - 1:
                    break
                if used >= budget:
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
                    update_archive(child_x, child_f)
                    used += 1

        # Return the archive points
        if archive_X:
            return np.array(archive_X)
        else:
            return np.array([lo + span * 0.5])
# EVOLVE-BLOCK-END
