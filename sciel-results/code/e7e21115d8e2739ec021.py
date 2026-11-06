import time
import math
import numpy as np
from scipy.optimize import differential_evolution, minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # Objective function for 8 variables representing 4 complex numbers
    # x = [x1, y1, x2, y2, x3, y3, x4, y4]
    # Points are (xk, yk) and (xk, -yk)
    def get_points(x):
        x1, y1, x2, y2, x3, y3, x4, y4 = x
        return np.array([
            [x1, y1],
            [x1, -y1],
            [x2, y2],
            [x2, -y2],
            [x3, y3],
            [x3, -y3],
            [x4, y4],
            [x4, -y4]
        ])

    def compute_score(pts):
        """
        Compute the actual score: Delta / n^n.
        Delta is product over ordered pairs of distances after scaling to diameter 2.
        """
        n = 8
        pts = np.asarray(pts)
        if not np.all(np.isfinite(pts)):
            return 0.0

        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        iu = np.triu_indices(n, k=1)
        d = dists[iu]

        # Check for coincident points
        if np.min(d) < 1e-12:
            return 0.0

        D = np.max(d)
        if D < 1e-12:
            return 0.0

        # Scale distances to diameter 2
        log_d_sum = np.sum(np.log(d))
        log_prod_unordered_scaled = log_d_sum + 28.0 * math.log(2.0) - 28.0 * math.log(D)
        log_score = 2.0 * log_prod_unordered_scaled - 8.0 * math.log(8.0)

        return math.exp(log_score)

    def objective(x):
        """
        Minimize negative log score for optimization.
        Returns a large penalty for invalid configurations.
        """
        pts = get_points(x)
        if not np.all(np.isfinite(pts)):
            return 1e100

        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        iu = np.triu_indices(8, k=1)
        d = dists[iu]

        if np.min(d) < 1e-9:
            return 1e100

        D = np.max(d)
        if D < 1e-9:
            return 1e100

        log_d_sum = np.sum(np.log(d))
        log_score = 2.0 * (log_d_sum + 28.0 * math.log(2.0) - 28.0 * math.log(D)) - 8.0 * math.log(8.0)

        return -log_score

    def two_square_objective(params):
        """
        Objective for two concentric squares, both rotated 45 degrees.
        params = [log_r1, log_r2] where r1, r2 > 0
        Points: (±r1/√2, ±r1/√2) and (±r2/√2, ±r2/√2)
        """
        log_r1, log_r2 = params
        r1 = math.exp(log_r1)
        r2 = math.exp(log_r2)

        # Ensure r1 != r2 and both positive
        if r1 < 1e-10 or r2 < 1e-10 or abs(r1 - r2) < 1e-10:
            return 1e100

        s1 = r1 / math.sqrt(2.0)
        s2 = r2 / math.sqrt(2.0)

        pts = np.array([
            [s1, s1],
            [s1, -s1],
            [-s1, s1],
            [-s1, -s1],
            [s2, s2],
            [s2, -s2],
            [-s2, s2],
            [-s2, -s2]
        ])

        if not np.all(np.isfinite(pts)):
            return 1e100

        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        iu = np.triu_indices(8, k=1)
        d = dists[iu]

        if np.min(d) < 1e-9:
            return 1e100

        D = np.max(d)
        if D < 1e-9:
            return 1e100

        log_d_sum = np.sum(np.log(d))
        log_score = 2.0 * (log_d_sum + 28.0 * math.log(2.0) - 28.0 * math.log(D)) - 8.0 * math.log(8.0)

        return -log_score

    bounds = [(-10, 10)] * 8

    best_val = np.inf
    best_x = None
    best_score = 0.0
    best_points = None

    def update_best_from_x(x, val):
        nonlocal best_val, best_x, best_score, best_points
        if val < best_val:
            best_val = val
            best_x = x.copy()
            pts = get_points(best_x)
            s = compute_score(pts)
            if s > best_score:
                best_score = s
                best_points = pts.copy()

    def update_best_from_points(pts, score):
        nonlocal best_score, best_points
        if score > best_score:
            best_score = score
            best_points = pts.copy()

    # === PHASE 1: Targeted two-concentric-squares optimization ===
    # This is the known high-performing topology (score ~1.235)
    # Parameters: log(r1), log(r2) with r1 < r2 (WLOG)
    # Regular octagon corresponds to r1 = r2 = 1 (degenerate), so we search r1/r2 in (0, 1)

    # Grid search over ratio r1/r2 to find good starting point
    best_ts_val = np.inf
    best_ts_params = None

    # Try a range of ratios
    for ratio in np.logspace(-3, 0, 50):  # r1/r2 from 0.001 to 1
        for r2 in np.logspace(-2, 2, 20):  # r2 from 0.01 to 100
            r1 = ratio * r2
            if r1 < 1e-6 or r2 < 1e-6:
                continue
            params = np.array([math.log(r1), math.log(r2)])
            val = two_square_objective(params)
            if val < best_ts_val:
                best_ts_val = val
                best_ts_params = params.copy()

    # Refine with local optimization from best grid point
    if best_ts_params is not None:
        try:
            res_ts = minimize(
                two_square_objective,
                best_ts_params,
                method='L-BFGS-B',
                bounds=[(-10, 10), (-10, 10)],
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res_ts.fun < best_ts_val:
                best_ts_val = res_ts.fun
                best_ts_params = res_ts.x.copy()
        except Exception:
            pass

    # Try multiple starting points for the two-square optimization
    np.random.seed(123)
    for _ in range(20):
        if time.time() > deadline - 60.0:
            break
        r1 = np.exp(np.random.uniform(-3, 3))
        r2 = np.exp(np.random.uniform(-3, 3))
        if abs(r1 - r2) < 0.01:
            continue
        params = np.array([math.log(r1), math.log(r2)])
        try:
            res_ts = minimize(
                two_square_objective,
                params,
                method='L-BFGS-B',
                bounds=[(-10, 10), (-10, 10)],
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-11}
            )
            if res_ts.fun < best_ts_val:
                best_ts_val = res_ts.fun
                best_ts_params = res_ts.x.copy()
        except Exception:
            pass

    # Evaluate the best two-square configuration
    if best_ts_params is not None and best_ts_val < 1e99:
        log_r1, log_r2 = best_ts_params
        r1 = math.exp(log_r1)
        r2 = math.exp(log_r2)
        s1 = r1 / math.sqrt(2.0)
        s2 = r2 / math.sqrt(2.0)
        ts_pts = np.array([
            [s1, s1], [s1, -s1], [-s1, s1], [-s1, -s1],
            [s2, s2], [s2, -s2], [-s2, s2], [-s2, -s2]
        ])
        ts_score = compute_score(ts_pts)
        update_best_from_points(ts_pts, ts_score)

        # Also try with r1 and r2 swapped (should be same by symmetry, but verify)
        # The configuration is symmetric under swapping, so skip

    # === PHASE 2: General 8-parameter search ===
    # Initial Guess: Regular Octagon
    angles = np.deg2rad([22.5, 67.5, 112.5, 157.5])
    init_x = []
    for a in angles:
        init_x.append(math.cos(a))
        init_x.append(math.sin(a))
    init_x = np.array(init_x)

    val = objective(init_x)
    update_best_from_x(init_x, val)

    # Differential Evolution
    if time.time() < deadline - 40.0:
        try:
            res = differential_evolution(
                objective,
                bounds,
                seed=42,
                maxiter=100,
                popsize=20,
                tol=1e-8,
                polish=True
            )
            if res.fun < best_val:
                update_best_from_x(res.x, res.fun)
        except Exception:
            pass

    # Local refinement from multiple starts
    start_points = []
    if best_x is not None:
        start_points.append(best_x)

    np.random.seed(42)
    for _ in range(8):
        if time.time() > deadline - 20.0:
            break
        if best_x is not None:
            x_perturb = best_x + 0.3 * np.random.randn(8)
        else:
            x_perturb = np.random.uniform(-5, 5, 8)
        x_perturb = np.clip(x_perturb, [b[0] for b in bounds], [b[1] for b in bounds])
        start_points.append(x_perturb)

    # Add two-square-derived starting points for the general search
    if best_ts_params is not None and best_ts_val < 1e99:
        log_r1, log_r2 = best_ts_params
        r1 = math.exp(log_r1)
        r2 = math.exp(log_r2)
        s1 = r1 / math.sqrt(2.0)
        s2 = r2 / math.sqrt(2.0)
        # Map to the 8-parameter representation:
        # Points: (s1,s1),(s1,-s1),(-s1,s1),(-s1,-s1),(s2,s2),(s2,-s2),(-s2,s2),(-s2,-s2)
        # In our parameterization: (x1,y1),(x1,-y1),(x2,y2),(x2,-y2),(x3,y3),(x3,-y3),(x4,y4),(x4,-y4)
        # So: x1=s1,y1=s1; x2=-s1,y2=s1; x3=s2,y3=s2; x4=-s2,y4=s2
        ts_x = np.array([s1, s1, -s1, s1, s2, s2, -s2, s2])
        start_points.append(ts_x)

        # Perturbed two-square starts
        for scale in [0.9, 1.1]:
            ts_x_pert = ts_x * scale
            start_points.append(ts_x_pert)

    for x0 in start_points:
        if time.time() > deadline - 10.0:
            break
        try:
            res_local = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-13, 'gtol': 1e-9}
            )
            if np.all(np.isfinite(res_local.x)):
                if res_local.fun < best_val:
                    update_best_from_x(res_local.x, res_local.fun)
        except Exception:
            pass

    # === PHASE 3: Final high-precision polish ===
    if time.time() < deadline - 5.0 and best_points is not None:
        # Polish the two-square solution with higher precision
        if best_ts_params is not None and best_ts_val < 1e99:
            try:
                res_final_ts = minimize(
                    two_square_objective,
                    best_ts_params,
                    method='L-BFGS-B',
                    bounds=[(-10, 10), (-10, 10)],
                    options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-13}
                )
                if res_final_ts.fun < best_ts_val:
                    log_r1, log_r2 = res_final_ts.x
                    r1 = math.exp(log_r1)
                    r2 = math.exp(log_r2)
                    s1 = r1 / math.sqrt(2.0)
                    s2 = r2 / math.sqrt(2.0)
                    ts_pts_final = np.array([
                        [s1, s1], [s1, -s1], [-s1, s1], [-s1, -s1],
                        [s2, s2], [s2, -s2], [-s2, s2], [-s2, -s2]
                    ])
                    ts_score_final = compute_score(ts_pts_final)
                    update_best_from_points(ts_pts_final, ts_score_final)
            except Exception:
                pass

        # Polish the general solution
        if best_x is not None:
            try:
                res_final = minimize(
                    objective,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-11}
                )
                if np.all(np.isfinite(res_final.x)) and res_final.fun < best_val:
                    update_best_from_x(res_final.x, res_final.fun)
            except Exception:
                pass

    # Construct final points
    if best_points is not None:
        pts = best_points
    elif best_x is not None:
        pts = get_points(best_x)
    else:
        # Fallback
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        pts = np.column_stack([np.cos(angles), np.sin(angles)])

    # Verify distinctness
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    iu = np.triu_indices(8, k=1)
    d = dists[iu]

    if np.min(d) < 1e-6 or not np.all(np.isfinite(pts)):
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        pts = np.column_stack([np.cos(angles), np.sin(angles)])

    max_coord = np.max(np.abs(pts))
    if max_coord > 1e6:
        pts = pts / max_coord * 1e5

    return pts.tolist()
