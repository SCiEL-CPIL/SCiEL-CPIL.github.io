"""Multi-objective optimizer for WFG4 using hypervolume-based CMA-ES with
exploit-the-structure: WFG4 is separable, so we exploit that directly."""
import numpy as np
import time
import math

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        mid = 0.5 * (lo + hi)
        budget = self.budget

        eval_count = 0
        # Archive: list of (x, f) for all evaluated points
        archive_X = []
        archive_F = []

        def time_left():
            return deadline - time.time()

        def check_deadline():
            if time.time() > deadline:
                raise TimeoutError()

        def evaluate(x):
            nonlocal eval_count
            x = np.asarray(x, dtype=float)
            f = np.asarray(problem(x), dtype=float)
            eval_count += 1
            archive_X.append(x.copy())
            archive_F.append(f.copy())
            return f

        # ---- Hypervolume computation (3D) ----
        def hypervolume_3d(points, ref):
            """Compute hypervolume for 3D points dominated by ref."""
            if len(points) == 0:
                return 0.0
            pts = [tuple(p) for p in points if all(p[i] < ref[i] for i in range(3))]
            if len(pts) == 0:
                return 0.0
            # Sort by first objective (ascending)
            pts.sort(key=lambda p: p[0])
            # Remove dominated points
            non_dom = []
            best_1 = float('inf')
            best_2 = float('inf')
            for p in pts:
                if p[1] < best_1 and p[2] < best_2:
                    non_dom.append(p)
                    if p[1] < best_1:
                        best_1 = p[1]
                    if p[2] < best_2:
                        best_2 = p[2]
            # For 3D, use the standard sweep
            # Sort by first coordinate
            non_dom.sort(key=lambda p: p[0])
            # Remove points dominated by others in 2D (obj1, obj2)
            # Actually, let's just do the standard 3D HV
            hv = 0.0
            n = len(non_dom)
            for i in range(n):
                p = non_dom[i]
                # Width in obj0: from p[0] to next point's obj0 (or ref[0])
                if i + 1 < n:
                    w0 = non_dom[i+1][0] - p[0]
                else:
                    w0 = ref[0] - p[0]
                if w0 <= 0:
                    continue
                # In the slice [p[0], p[0]+w0], compute 2D HV of points with obj0 <= p[0]+w0
                # For points in this slice, their obj0 is all <= p[0]+w0
                # The 2D HV is over (obj1, obj2)
                # Points that are active in this slice: those with obj0 <= p[0] + w0
                # Since we sorted, all points up to index i have obj0 <= p[0]
                # Points from i+1 onward have obj0 >= non_dom[i+1][0] = p[0] + w0
                # So only points 0..i are active
                # Compute 2D HV of points 0..i projected to (obj1, obj2)
                pts_2d = [(non_dom[j][1], non_dom[j][2]) for j in range(i+1)]
                hv_2d = hypervolume_2d(pts_2d, (ref[1], ref[2]))
                hv += w0 * hv_2d
            return hv

        def hypervolume_2d(points, ref):
            """Compute 2D hypervolume."""
            if len(points) == 0:
                return 0.0
            pts = [p for p in points if p[0] < ref[0] and p[1] < ref[1]]
            if len(pts) == 0:
                return 0.0
            # Sort by first coordinate
            pts.sort(key=lambda p: p[0])
            # Remove dominated
            non_dom = []
            best_y = float('inf')
            for p in pts:
                if p[1] < best_y:
                    non_dom.append(p)
                    best_y = p[1]
            hv = 0.0
            for i in range(len(non_dom)):
                p = non_dom[i]
                if i + 1 < len(non_dom):
                    w = non_dom[i+1][0] - p[0]
                else:
                    w = ref[0] - p[0]
                if w <= 0:
                    continue
                h = ref[1] - p[1]
                hv += w * h
            return hv

        def compute_hv():
            """Compute HV of current archive with reference point."""
            ref = np.array([3.0, 5.0, 7.0])
            F = np.array(archive_F)
            return hypervolume_3d(F.tolist(), ref.tolist())

        # ---- Phase 1: Initial exploration to understand structure ----
        # Evaluate at strategic points
        probe_points = [mid.copy()]
        for i in range(dim):
            p_lo = mid.copy(); p_lo[i] = lo[i]; probe_points.append(p_lo)
            p_hi = mid.copy(); p_hi[i] = hi[i]; probe_points.append(p_hi)
        probe_points.append(lo.copy())
        probe_points.append(hi.copy())

        for p in probe_points:
            if eval_count >= budget or time_left() < 5.0:
                break
            evaluate(p)

        # Determine front variable (largest impact)
        probe_F = np.array(archive_F[:len(probe_points)])
        impact = np.zeros(dim)
        for i in range(dim):
            f_mid = probe_F[0]
            f_lo = probe_F[1 + 2*i]
            f_hi = probe_F[2 + 2*i]
            impact[i] = max(np.max(np.abs(f_lo - f_mid)), np.max(np.abs(f_hi - f_mid)))

        front_var = int(np.argmax(impact))

        # ---- Phase 2: Learn the Pareto front via 1D sweep ----
        n_coarse = 15
        t_vals = np.linspace(0.0, 1.0, n_coarse)
        sweep_F = []
        for t in t_vals:
            if eval_count >= budget or time_left() < 5.0:
                break
            p = mid.copy()
            p[front_var] = lo[front_var] + t * span[front_var]
            f = evaluate(p)
            sweep_F.append(f)

        # ---- Phase 3: Optimize penalty variables for points along the front ----
        # For each front point, do coordinate descent on non-front variables
        n_front_points = 12
        front_ts = np.linspace(0.0, 1.0, n_front_points)

        remaining = budget - eval_count
        per_point_evals = max(5, int(remaining / (n_front_points * 1.5)))

        optimized_points = []
        optimized_Fs = []

        for t in front_ts:
            if eval_count >= budget or time_left() < 5.0:
                break

            x = mid.copy()
            x[front_var] = lo[front_var] + t * span[front_var]
            f = evaluate(x)

            # Coordinate descent on non-front variables
            for iteration in range(max(1, per_point_evals // dim)):
                if eval_count >= budget or time_left() < 2.0:
                    break
                improved = False
                for i in range(dim):
                    if i == front_var:
                        continue
                    if eval_count >= budget or time_left() < 1.0:
                        break
                    # Try several trial values
                    trial_vals = [
                        lo[i], hi[i],
                        0.1 * lo[i] + 0.9 * hi[i],
                        0.3 * lo[i] + 0.7 * hi[i],
                        0.5 * lo[i] + 0.5 * hi[i],
                        0.7 * lo[i] + 0.3 * hi[i],
                        0.9 * lo[i] + 0.1 * hi[i],
                    ]
                    best_sum = np.sum(f)
                    best_val = x[i]
                    best_f = f
                    for tv in trial_vals:
                        if abs(tv - x[i]) < 1e-10:
                            continue
                        if eval_count >= budget or time_left() < 1.0:
                            break
                        x_trial = x.copy()
                        x_trial[i] = tv
                        f_trial = evaluate(x_trial)
                        if np.sum(f_trial) < best_sum:
                            best_sum = np.sum(f_trial)
                            best_val = tv
                            best_f = f_trial
                    if abs(best_val - x[i]) > 1e-10:
                        x[i] = best_val
                        f = best_f
                        improved = True
                if not improved:
                    break

            optimized_points.append(x.copy())
            optimized_Fs.append(f.copy())

        # ---- Phase 4: Determine optimal penalty configuration ----
        # For non-front variables, use the median of optimized points
        optimal_penalty = mid.copy()
        if optimized_points:
            for i in range(dim):
                if i == front_var:
                    continue
                vals = np.array([op[i] for op in optimized_points])
                # Pick the value that gives lowest average objective sum
                # Just use median as a robust choice
                optimal_penalty[i] = np.median(vals)

        # ---- Phase 5: Dense sweep along the front with optimized penalties ----
        remaining = budget - eval_count
        n_dense = max(10, min(120, int(remaining * 0.8)))
        n_dense = max(5, min(n_dense, remaining))
        dense_ts = np.linspace(0.0, 1.0, n_dense)

        for t in dense_ts:
            if eval_count >= budget or time_left() < 1.0:
                break
            x = optimal_penalty.copy()
            x[front_var] = lo[front_var] + t * span[front_var]
            f = evaluate(x)

        # ---- Phase 6: Local refinement using CMA-ES-like search ----
        # If we still have budget, do a few local refinements around promising points
        remaining = budget - eval_count
        if remaining > 20 and time_left() > 10.0:
            # Pick the point with best sum of objectives
            if archive_F:
                F_arr = np.array(archive_F)
                sums = np.sum(F_arr, axis=1)
                best_idx = int(np.argmin(sums))
                best_x = archive_X[best_idx].copy()
                best_f = archive_F[best_idx].copy()

                sigma = 0.05 * np.max(span)
                C = np.eye(dim)
                n_iter = min(10, remaining // dim)

                for _ in range(n_iter):
                    if eval_count >= budget or time_left() < 2.0:
                        break
                    # Sample
                    L = np.linalg.cholesky(C + 1e-8 * np.eye(dim))
                    z = np.random.standard_normal(dim)
                    x_new = best_x + sigma * (L @ z)
                    x_new = np.clip(x_new, lo, hi)
                    f_new = evaluate(x_new)

                    # Accept if it improves the sum
                    if np.sum(f_new) < np.sum(best_f):
                        # Update C towards this direction
                        diff = x_new - best_x
                        C = 0.7 * C + 0.3 * (np.outer(diff, diff) / (sigma**2))
                        C = (C + C.T) / 2
                        C += 1e-6 * np.eye(dim)
                        best_x = x_new
                        best_f = f_new
                        sigma *= 0.95
                    else:
                        sigma *= 0.9

        return np.array(archive_X) if archive_X else np.array([mid])
