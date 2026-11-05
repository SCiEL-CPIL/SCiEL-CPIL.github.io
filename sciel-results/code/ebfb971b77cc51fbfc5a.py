import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    n = 8

    def exact_score(points):
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists = np.sqrt(dists_sq)
        mask = ~np.eye(n, dtype=bool)
        off_diag_dists = dists[mask]
        if len(off_diag_dists) == 0:
            return -np.inf
        diameter = np.max(off_diag_dists)
        if diameter < 1e-15:
            return -np.inf
        scaled_dists = off_diag_dists * (2.0 / diameter)
        scaled_dists = np.maximum(scaled_dists, 1e-15)
        log_prod = np.sum(np.log(scaled_dists))
        return log_prod - n * np.log(n)

    def log_product_smooth(points, beta):
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists = np.sqrt(dists_sq)
        mask = ~np.eye(n, dtype=bool)
        off_diag_dists = dists[mask]
        if len(off_diag_dists) == 0:
            return np.inf
        scaled = off_diag_dists / beta
        m = np.max(scaled)
        log_sum_exp = m + np.log(np.sum(np.exp(scaled - m)))
        diameter = beta * log_sum_exp
        if diameter < 1e-15:
            return np.inf
        scaled_dists = off_diag_dists * (2.0 / diameter)
        scaled_dists = np.maximum(scaled_dists, 1e-15)
        log_prod = np.sum(np.log(scaled_dists))
        return n * np.log(n) - log_prod

    # Two-square parameterization:
    # Square 1: radius r1, rotation a1, 4 points at angles a1 + k*pi/2
    # Square 2: radius r2, rotation a2, 4 points at angles a2 + k*pi/2
    # Parameters: [r1, a1, r2, a2] -> 4D
    def two_square_points(params):
        r1, a1, r2, a2 = params
        k = np.arange(4)
        a1_full = a1 + k * np.pi / 2
        a2_full = a2 + k * np.pi / 2
        sq1 = np.column_stack([np.cos(a1_full) * r1, np.sin(a1_full) * r1])
        sq2 = np.column_stack([np.cos(a2_full) * r2, np.sin(a2_full) * r2])
        return np.vstack([sq1, sq2])

    def two_square_obj(params, beta):
        pts = two_square_points(params)
        return log_product_smooth(pts, beta)

    # Baseline: regular octagon
    angles = 2 * np.pi * np.arange(n) / n
    reg_oct = np.column_stack([np.cos(angles), np.sin(angles)])
    best_score_val = exact_score(reg_oct)
    best_points = reg_oct.copy()

    def try_update(points):
        nonlocal best_score_val, best_points
        if np.any(np.isnan(points)):
            return
        d = np.sqrt(np.sum((points[:, np.newaxis, :] - points[np.newaxis, :, :])**2, axis=2))
        np.fill_diagonal(d, np.inf)
        if np.min(d) < 1e-8:
            return
        s = exact_score(points)
        if s > best_score_val:
            best_score_val = s
            best_points = points.copy()

    # Phase 1: Dense grid on two-square manifold (4D)
    # Parameters: [r1, a1, r2, a2]
    # By symmetry, we can fix a1=0 and search a2 in [0, pi/2]
    # r1, r2 in [0.3, 1.5]
    # Regular octagon corresponds to r1=r2=1, a1=0, a2=pi/8
    
    grid_starts = []
    # Regular octagon in two-square form: r1=r2=1, a1=0, a2=pi/8
    grid_starts.append([1.0, 0.0, 1.0, np.pi/8])
    
    # Dense grid
    r1_vals = np.linspace(0.4, 1.4, 8)
    r2_vals = np.linspace(0.4, 1.4, 8)
    a2_vals = np.linspace(0.0, np.pi/2, 12)
    
    for r1 in r1_vals:
        for r2 in r2_vals:
            for a2 in a2_vals:
                grid_starts.append([r1, 0.0, r2, a2])
    
    # Also try with a1 != 0 (breaks some symmetry)
    for a1 in [0.0, np.pi/16, np.pi/8, np.pi/4]:
        for r1 in [0.6, 0.8, 1.0, 1.2]:
            for r2 in [0.6, 0.8, 1.0, 1.2]:
                for a2 in [0.0, np.pi/16, np.pi/8, np.pi/4, 3*np.pi/16]:
                    grid_starts.append([r1, a1, r2, a2])
    
    # Phase 1: Optimize each grid start on 4D manifold
    betas_phase1 = [50.0, 100.0]
    
    for gs in grid_starts:
        if time.time() > deadline - 40:
            break
        for beta in betas_phase1:
            if time.time() > deadline - 40:
                break
            try:
                res = minimize(
                    lambda x, b=beta: two_square_obj(x, b),
                    gs,
                    method='L-BFGS-B',
                    bounds=[(0.1, 5.0), (-np.pi, np.pi), (0.1, 5.0), (-np.pi, np.pi)],
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.fun < np.inf:
                    candidate = two_square_points(res.x)
                    try_update(candidate)
            except Exception:
                pass
    
    # Phase 2: Refine best on 4D manifold with higher beta
    if time.time() < deadline - 25:
        # Find the best two-square params by re-optimizing from best_points
        # Project best_points onto two-square manifold approximately
        # Try multiple projections
        proj_starts = []
        # Regular octagon params
        proj_starts.append([1.0, 0.0, 1.0, np.pi/8])
        # Try to fit: split 8 points into two groups of 4
        # Use the current best points
        for split_seed in range(5):
            np.random.seed(split_seed)
            idx = np.random.permutation(n)
            g1 = idx[:4]
            g2 = idx[4:]
            # Fit circles to each group
            try:
                from scipy.optimize import least_squares
                def fit_circle(points):
                    x, y = points[:, 0], points[:, 1]
                    def residuals(c):
                        cx, cy, r = c
                        return np.sqrt((x-cx)**2 + (y-cy)**2) - r
                    c0 = [np.mean(x), np.mean(y), np.mean(np.sqrt((x-np.mean(x))**2 + (y-np.mean(y))**2))]
                    res_c = least_squares(residuals, c0, max_nfev=100)
                    return res_c.x
                c1 = fit_circle(best_points[g1])
                c2 = fit_circle(best_points[g2])
                # Convert to polar around origin
                r1 = np.sqrt(c1[0]**2 + c1[1]**2) if abs(c1[2]) > 0.01 else c1[2]
                r2 = np.sqrt(c2[0]**2 + c2[1]**2) if abs(c2[2]) > 0.01 else c2[2]
                a1 = np.arctan2(c1[1], c1[0]) if abs(c1[2]) > 0.01 else 0.0
                a2 = np.arctan2(c2[1], c2[0]) if abs(c2[2]) > 0.01 else 0.0
                # Use mean radius
                r1 = np.mean(np.sqrt((best_points[g1,0]-c1[0])**2 + (best_points[g1,1]-c1[1])**2))
                r2 = np.mean(np.sqrt((best_points[g2,0]-c2[0])**2 + (best_points[g2,1]-c2[1])**2))
                if r1 > 0.1 and r2 > 0.1:
                    proj_starts.append([r1, a1, r2, a2])
            except Exception:
                pass
        
        for ps in proj_starts:
            if time.time() > deadline - 15:
                break
            for beta_val in [200.0, 500.0, 1000.0, 2000.0]:
                if time.time() > deadline - 10:
                    break
                try:
                    res = minimize(
                        lambda x, b=beta_val: two_square_obj(x, b),
                        ps,
                        method='L-BFGS-B',
                        bounds=[(0.1, 5.0), (-np.pi, np.pi), (0.1, 5.0), (-np.pi, np.pi)],
                        options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
                    )
                    if res.fun < np.inf:
                        candidate = two_square_points(res.x)
                        try_update(candidate)
                except Exception:
                    pass
    
    # Phase 3: Full 16D refinement from best symmetric point
    if time.time() < deadline - 20:
        for beta_val in [500.0, 2000.0, 5000.0, 10000.0]:
            if time.time() > deadline - 10:
                break
            try:
                res = minimize(
                    lambda x: log_product_smooth(x.reshape(n, 2), beta_val),
                    best_points.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                if res.fun < np.inf:
                    candidate = res.x.reshape(n, 2)
                    try_update(candidate)
            except Exception:
                pass
    
    # Phase 4: Jitter and re-optimize in 16D to escape local minima
    if time.time() < deadline - 15:
        np.random.seed(42)
        for _ in range(15):
            if time.time() > deadline - 8:
                break
            for jitter_scale in [0.02, 0.05, 0.1]:
                if time.time() > deadline - 8:
                    break
                jitter = np.random.normal(0, jitter_scale, (n, 2))
                jittered = best_points + jitter
                try:
                    res = minimize(
                        lambda x: log_product_smooth(x.reshape(n, 2), 2000.0),
                        jittered.flatten(),
                        method='L-BFGS-B',
                        options={'maxiter': 400, 'ftol': 1e-13, 'gtol': 1e-9}
                    )
                    if res.fun < np.inf:
                        candidate = res.x.reshape(n, 2)
                        try_update(candidate)
                except Exception:
                    pass
    
    # Phase 5: Ultra-precise final polish
    if time.time() < deadline - 5:
        for beta_val in [50000.0, 100000.0, 500000.0]:
            if time.time() > deadline - 3:
                break
            try:
                final_res = minimize(
                    lambda x: log_product_smooth(x.reshape(n, 2), beta_val),
                    best_points.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14}
                )
                if final_res.fun < np.inf:
                    candidate = final_res.x.reshape(n, 2)
                    try_update(candidate)
            except Exception:
                pass
    
    # Phase 6: Final two-square refinement with very high beta
    if time.time() < deadline - 3:
        # Re-extract best two-square params
        best_ts_params = [1.0, 0.0, 1.0, np.pi/8]
        # Try to fit from best_points
        try:
            from scipy.optimize import least_squares
            # Try splitting into two groups of 4
            best_fit_score = -np.inf
            for split_seed in range(20):
                np.random.seed(split_seed)
                idx = np.random.permutation(n)
                g1 = idx[:4]
                g2 = idx[4:]
                try:
                    def fit_circle(points):
                        x, y = points[:, 0], points[:, 1]
                        def residuals(c):
                            cx, cy, r = c
                            return np.sqrt((x-cx)**2 + (y-cy)**2) - r
                        c0 = [np.mean(x), np.mean(y), np.mean(np.sqrt((x-np.mean(x))**2 + (y-np.mean(y))**2))]
                        res_c = least_squares(residuals, c0, max_nfev=100)
                        return res_c.x
                    c1 = fit_circle(best_points[g1])
                    c2 = fit_circle(best_points[g2])
                    r1 = np.mean(np.sqrt((best_points[g1,0]-c1[0])**2 + (best_points[g1,1]-c1[1])**2))
                    r2 = np.mean(np.sqrt((best_points[g2,0]-c2[0])**2 + (best_points[g2,1]-c2[1])**2))
                    a1 = np.arctan2(c1[1], c1[0])
                    a2 = np.arctan2(c2[1], c2[0])
                    if r1 > 0.1 and r2 > 0.1:
                        s = exact_score(two_square_points([r1, a1, r2, a2]))
                        if s > best_fit_score:
                            best_fit_score = s
                            best_ts_params = [r1, a1, r2, a2]
                except Exception:
                    pass
        except Exception:
            pass
        
        for beta_val in [10000.0, 50000.0]:
            if time.time() > deadline - 2:
                break
            try:
                res = minimize(
                    lambda x, b=beta_val: two_square_obj(x, b),
                    best_ts_params,
                    method='L-BFGS-B',
                    bounds=[(0.1, 5.0), (-np.pi, np.pi), (0.1, 5.0), (-np.pi, np.pi)],
                    options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                if res.fun < np.inf:
                    candidate = two_square_points(res.x)
                    try_update(candidate)
            except Exception:
                pass
    
    # Final validation
    dists = np.sqrt(np.sum((best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :])**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-10:
        best_points = reg_oct

    max_coord = np.max(np.abs(best_points))
    if max_coord > 1e6:
        best_points = best_points / max_coord * 1e5

    return [[float(p[0]), float(p[1])] for p in best_points]
