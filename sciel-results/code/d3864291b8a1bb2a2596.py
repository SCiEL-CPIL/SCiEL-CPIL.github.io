import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    n = 8
    pair_indices = np.array([(i, j) for i in range(n) for j in range(i + 1, n)])
    idx_i = pair_indices[:, 0]
    idx_j = pair_indices[:, 1]
    
    def calculate_score(points):
        diff = points[idx_i, :] - points[idx_j, :]
        dists = np.sqrt(np.sum(diff**2, axis=1))
        if np.any(dists < 1e-10):
            return -np.inf
        D = np.max(dists)
        if D < 1e-10:
            return -np.inf
        sum_log_d = np.sum(np.log(dists))
        log_scaled_sum = sum_log_d + 28.0 * np.log(2.0) - 28.0 * np.log(D)
        log_score = 2.0 * log_scaled_sum - 8.0 * np.log(8.0)
        return log_score

    def build_two_squares(R1, R2, theta, dx, dy):
        """Build 8 points from two squares with radii R1,R2, rotation theta, offset (dx,dy) for second square."""
        angles = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        p1 = np.column_stack([R1 * np.cos(angles), R1 * np.sin(angles)])
        p2 = np.column_stack([R2 * np.cos(angles + theta), R2 * np.sin(angles + theta)])
        p2 = p2 + np.array([dx, dy])
        return np.vstack([p1, p2])

    def objective_5d(x):
        if time.time() > deadline:
            return 1e10
        R1, R2, theta, dx, dy = x
        if R1 < 0.05 or R2 < 0.05:
            return 1e10
        pts = build_two_squares(R1, R2, theta, dx, dy)
        s = calculate_score(pts)
        if np.isinf(s) or np.isnan(s):
            return 1e10
        return -s

    best_points = None
    best_score = -np.inf
    
    # 1. Regular Octagon
    angles_oct = np.linspace(0, 2*np.pi, 8, endpoint=False)
    pts_oct = np.column_stack([np.cos(angles_oct), np.sin(angles_oct)])
    s_oct = calculate_score(pts_oct)
    if s_oct > best_score:
        best_score = s_oct
        best_points = pts_oct

    # 2. DE on 5D two-squares manifold
    bounds_5d = [
        (0.05, 3.0),   # R1
        (0.05, 3.0),   # R2
        (0.0, np.pi/4),# theta
        (-1.0, 1.0),   # dx
        (-1.0, 1.0)    # dy
    ]
    
    candidates = []
    
    try:
        res = differential_evolution(
            objective_5d,
            bounds_5d,
            maxiter=300,
            popsize=30,
            tol=1e-10,
            seed=42,
            init='sobol',
            workers=1,
            polish=False
        )
        if res.fun < 1e9:
            x_opt = res.x
            pts_cand = build_two_squares(*x_opt)
            s_cand = calculate_score(pts_cand)
            candidates.append((s_cand, pts_cand))
            if s_cand > best_score:
                best_score = s_cand
                best_points = pts_cand
    except Exception:
        pass

    # 3. Try multiple symmetric starting points with Nelder-Mead on 16 params
    if time.time() < deadline - 20.0:
        def obj_16(y):
            if time.time() > deadline:
                return 1e10
            pts = y.reshape(8, 2)
            s = calculate_score(pts)
            if np.isinf(s) or np.isnan(s):
                return 1e10
            return -s

        # Starting points: regular octagon, and various two-squares configs
        start_pts_list = [pts_oct.copy()]
        
        # Add some hand-crafted two-squares starting points
        for R1, R2, theta, dx, dy in [
            (1.0, 0.7, np.pi/8, 0.0, 0.0),
            (1.0, 0.5, np.pi/4, 0.0, 0.0),
            (1.0, 0.8, np.pi/6, 0.0, 0.0),
            (1.0, 0.6, np.pi/8, 0.1, 0.0),
            (1.0, 0.9, np.pi/12, 0.0, 0.1),
            (1.0, 0.4, np.pi/4, 0.0, 0.0),
            (1.0, 1.0, np.pi/8, 0.0, 0.0),
            (1.0, 0.3, np.pi/4, 0.0, 0.0),
        ]:
            start_pts_list.append(build_two_squares(R1, R2, theta, dx, dy))

        # Also add the DE result if available
        if candidates:
            start_pts_list.append(candidates[0][1].copy())

        for sp in start_pts_list:
            if time.time() > deadline - 5.0:
                break
            try:
                res_nm = minimize(
                    obj_16,
                    sp.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                )
                s_nm = -res_nm.fun
                if s_nm > best_score and not np.isinf(s_nm) and not np.isnan(s_nm):
                    best_score = s_nm
                    best_points = res_nm.x.reshape(8, 2)
                    candidates.append((s_nm, best_points.copy()))
            except Exception:
                pass

    # 4. If we have good candidates, do a final Powell polish on the best
    if candidates and time.time() < deadline - 3.0:
        candidates.sort(key=lambda t: -t[0])
        best_cand = candidates[0][1]
        try:
            def obj_16b(y):
                if time.time() > deadline:
                    return 1e10
                pts = y.reshape(8, 2)
                s = calculate_score(pts)
                if np.isinf(s) or np.isnan(s):
                    return 1e10
                return -s
            res_pw = minimize(
                obj_16b,
                best_cand.flatten(),
                method='Powell',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            s_pw = -res_pw.fun
            if s_pw > best_score and not np.isinf(s_pw) and not np.isnan(s_pw):
                best_score = s_pw
                best_points = res_pw.x.reshape(8, 2)
        except Exception:
            pass

    if best_points is None:
        angles_oct = np.linspace(0, 2*np.pi, 8, endpoint=False)
        best_points = np.column_stack([np.cos(angles_oct), np.sin(angles_oct)])
        
    # Verify distinctness
    diffs = best_points[:, np.newaxis, :] - best_points[:, :, np.newaxis]
    d = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(d, 1.0)
    if np.min(d) < 1e-6:
        # Fallback to octagon
        angles_oct = np.linspace(0, 2*np.pi, 8, endpoint=False)
        best_points = np.column_stack([np.cos(angles_oct), np.sin(angles_oct)])

    # Scale to diameter 2
    D = np.max(d)
    if D > 0:
        best_points = best_points * (2.0 / D)
        
    return [[float(p[0]), float(p[1])] for p in best_points]
