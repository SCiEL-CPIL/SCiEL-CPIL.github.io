import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n = 8
    
    # Precompute pair indices for 28 unordered pairs
    pair_indices = np.array([(i, j) for i in range(n) for j in range(i + 1, n)])
    idx_i = pair_indices[:, 0]
    idx_j = pair_indices[:, 1]
    
    def calculate_score(points):
        """
        points: 8x2 array
        Returns: log of the scaled product of distances.
        Score = Delta / 8^8, where Delta is product of ordered pair distances.
        Delta = (prod_{i<j} d_ij)^2.
        log(Score) = 2 * sum_{i<j} log(d_ij_scaled) - 8 * log(8).
        """
        # Vectorized distance calculation for 28 pairs
        diff = points[idx_i, :] - points[idx_j, :]
        dists_sq = np.sum(diff**2, axis=1)
        if np.any(dists_sq < 1e-24):
            return -np.inf
        dists = np.sqrt(dists_sq)
        D = np.max(dists)
        if D < 1e-12:
            return -np.inf
        # Scale to diameter 2: d_scaled = d * (2/D)
        # log(d_scaled) = log(d) + log(2) - log(D)
        sum_log_d_scaled = np.sum(np.log(dists)) + 28.0 * (np.log(2.0) - np.log(D))
        log_score = 2.0 * sum_log_d_scaled - 8.0 * np.log(8.0)
        return log_score
    
    def build_two_squares_3d(R1, R2, theta):
        """Two concentric squares, centered at origin."""
        angles = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        p1 = np.column_stack([R1 * np.cos(angles), R1 * np.sin(angles)])
        p2 = np.column_stack([R2 * np.cos(angles + theta), R2 * np.sin(angles + theta)])
        return np.vstack([p1, p2])
    
    def build_two_squares_5d(R1, R2, theta, dx, dy):
        """Two squares, second one offset by (dx, dy)."""
        angles = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        p1 = np.column_stack([R1 * np.cos(angles), R1 * np.sin(angles)])
        p2 = np.column_stack([R2 * np.cos(angles + theta) + dx, R2 * np.sin(angles + theta) + dy])
        return np.vstack([p1, p2])
    
    def build_central_symmetric(x):
        """
        x: 8-dim array representing 4 points (x0, y0, x1, y1, x2, y2, x3, y3)
        Returns: 8x2 array of points [p0, p1, p2, p3, -p0, -p1, -p2, -p3]
        """
        p = x.reshape(4, 2)
        return np.vstack([p, -p])
    
    def objective_3d(params):
        """Objective for 3D optimization (R1, R2, theta)."""
        if time.time() > deadline - 5.0:
            return 1e10
        R1, R2, theta = params
        if R1 < 0.05 or R2 < 0.05:
            return 1e10
        pts = build_two_squares_3d(R1, R2, theta)
        s = calculate_score(pts)
        if np.isinf(s) or np.isnan(s):
            return 1e10
        return -s
    
    def objective_5d(params):
        """Objective for 5D optimization (R1, R2, theta, dx, dy)."""
        if time.time() > deadline - 5.0:
            return 1e10
        R1, R2, theta, dx, dy = params
        if R1 < 0.05 or R2 < 0.05:
            return 1e10
        pts = build_two_squares_5d(R1, R2, theta, dx, dy)
        s = calculate_score(pts)
        if np.isinf(s) or np.isnan(s):
            return 1e10
        return -s
    
    def objective_symmetric(x):
        """Objective for 8D optimization (4 points x 2 coords) with central symmetry."""
        if time.time() > deadline - 5.0:
            return 1e10
        pts = build_central_symmetric(x)
        s = calculate_score(pts)
        if np.isinf(s) or np.isnan(s):
            return 1e10
        return -s
    
    def objective_16(x):
        """Objective for 16D optimization (8 points x 2 coords)."""
        if time.time() > deadline - 3.0:
            return 1e10
        pts = x.reshape(8, 2)
        s = calculate_score(pts)
        if np.isinf(s) or np.isnan(s):
            return 1e10
        return -s
    
    best_score = -np.inf
    best_points = None
    
    # 1. Regular Octagon Baseline
    angles_oct = np.linspace(0, 2*np.pi, 8, endpoint=False)
    pts_oct = np.column_stack([np.cos(angles_oct), np.sin(angles_oct)])
    s_oct = calculate_score(pts_oct)
    if s_oct > best_score:
        best_score = s_oct
        best_points = pts_oct.copy()
    
    candidates = [(s_oct, pts_oct.copy())]
    
    # 2. Global Search on 3D Two-Square Manifold (concentric)
    bounds_3d = [
        (0.1, 3.0),   # R1
        (0.1, 3.0),   # R2
        (0.0, np.pi/4)# theta
    ]
    
    try:
        res_de_3d = differential_evolution(
            objective_3d,
            bounds_3d,
            maxiter=300,
            popsize=40,
            tol=1e-12,
            seed=42,
            init='sobol',
            workers=1,
            polish=False
        )
        if res_de_3d.fun < 1e9:
            x_opt = res_de_3d.x
            pts_cand = build_two_squares_3d(*x_opt)
            s_cand = calculate_score(pts_cand)
            candidates.append((s_cand, pts_cand))
            if s_cand > best_score:
                best_score = s_cand
                best_points = pts_cand.copy()
    except Exception:
        pass
    
    # 3. Global Search on 5D Two-Square Manifold (with offset)
    bounds_5d = [
        (0.1, 3.0),   # R1
        (0.1, 3.0),   # R2
        (0.0, np.pi/4),# theta
        (-2.0, 2.0),  # dx
        (-2.0, 2.0),  # dy
    ]
    
    try:
        res_de_5d = differential_evolution(
            objective_5d,
            bounds_5d,
            maxiter=200,
            popsize=30,
            tol=1e-12,
            seed=123,
            init='sobol',
            workers=1,
            polish=False
        )
        if res_de_5d.fun < 1e9:
            x_opt = res_de_5d.x
            pts_cand = build_two_squares_5d(*x_opt)
            s_cand = calculate_score(pts_cand)
            candidates.append((s_cand, pts_cand))
            if s_cand > best_score:
                best_score = s_cand
                best_points = pts_cand.copy()
    except Exception:
        pass
    
    # 3.5 Global Search on 8D Central Symmetric Manifold
    bounds_sym = [(-2.0, 2.0)] * 8
    
    try:
        res_de_sym = differential_evolution(
            objective_symmetric,
            bounds_sym,
            maxiter=200,
            popsize=40,
            tol=1e-12,
            seed=999,
            init='sobol',
            workers=1,
            polish=False
        )
        if res_de_sym.fun < 1e9:
            x_opt = res_de_sym.x
            pts_cand = build_central_symmetric(x_opt)
            s_cand = calculate_score(pts_cand)
            candidates.append((s_cand, pts_cand))
            if s_cand > best_score:
                best_score = s_cand
                best_points = pts_cand.copy()
    except Exception:
        pass
    
    # 4. Local Refinement on 16D Space
    candidates.sort(key=lambda t: -t[0])
    
    # Refine top 3 candidates
    for s_init, pts_init in candidates[:3]:
        if time.time() > deadline - 20.0:
            break
        
        x_init = pts_init.flatten()
        
        # Step 4a: Powell Method
        try:
            res_powell = minimize(
                objective_16,
                x_init,
                method='Powell',
                options={'maxiter': 800, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            x_best = res_powell.x
            s_powell = -res_powell.fun
            if np.isfinite(s_powell) and s_powell > -1e9:
                if s_powell > best_score:
                    best_score = s_powell
                    best_points = x_best.reshape(8, 2).copy()
        except Exception:
            x_best = x_init
        
        # Step 4b: Nelder-Mead Polish
        if time.time() < deadline - 10.0:
            try:
                res_nm = minimize(
                    objective_16,
                    x_best,
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
                )
                x_nm = res_nm.x
                s_nm = -res_nm.fun
                if np.isfinite(s_nm) and s_nm > best_score:
                    best_score = s_nm
                    best_points = x_nm.reshape(8, 2).copy()
            except Exception:
                pass
    
    # 5. Final Safety Polish
    if best_points is not None and time.time() < deadline - 3.0:
        try:
            res_final = minimize(
                objective_16,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
            )
            s_final = -res_final.fun
            if np.isfinite(s_final) and s_final > best_score:
                best_score = s_final
                best_points = res_final.x.reshape(8, 2).copy()
        except Exception:
            pass
    
    if best_points is None:
        best_points = pts_oct
    
    # Verify distinctness
    diffs = best_points[:, np.newaxis, :] - best_points[:, :, np.newaxis]
    d = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(d, 1.0)
    if np.min(d) < 1e-6:
        best_points = pts_oct
    
    # Scale to diameter 2
    D = np.max(d)
    if D > 0:
        best_points = best_points * (2.0 / D)
        
    return [[float(p[0]), float(p[1])] for p in best_points]
