import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # 2s safety margin
    
    n = 8
    
    def exact_log_score(points):
        """
        Calculate the exact log-score: 
        log( (prod_{i!=j} d'_ij) / n^n )
        where d'_ij = d_ij * (2 / diameter)
        """
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
        log_score = log_prod - n * np.log(n)
        return log_score

    def exact_negative_log_score(x):
        """
        Exact negative log-score for optimization (minimization).
        x is flattened points (16 floats).
        """
        pts = x.reshape(n, 2)
        if np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
            return 1e10
        s = exact_log_score(pts)
        if not np.isfinite(s):
            return 1e10
        return -s

    best_state = {'score': -np.inf, 'points': None}
    
    def try_update(points, best_state):
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            return best_state
        
        # Check distinct points
        d = np.sqrt(np.sum((points[:, np.newaxis, :] - points[np.newaxis, :, :])**2, axis=2))
        np.fill_diagonal(d, np.inf)
        if np.min(d) < 1e-8:
            return best_state
            
        s = exact_log_score(points)
        if np.isfinite(s) and s > best_state['score']:
            return {'score': s, 'points': points.copy()}
        return best_state

    # Phase 1: Search in the "Two Concentric Squares" family
    # Points: 
    # Square 1: angles 0, pi/2, pi, 3pi/2, radius r1
    # Square 2: angles theta, theta+pi/2, theta+pi, theta+3pi/2, radius r2
    
    def two_squares_params_to_points(params):
        r1, r2, theta = params
        # Ensure radii are positive
        r1 = abs(r1)
        r2 = abs(r2)
        
        a1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        a2 = a1 + theta
        
        pts1 = np.column_stack([r1 * np.cos(a1), r1 * np.sin(a1)])
        pts2 = np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)])
        
        return np.vstack([pts1, pts2])

    def two_squares_objective(params):
        pts = two_squares_params_to_points(params)
        s = exact_log_score(pts)
        if not np.isfinite(s):
            return 1e10
        return -s

    # Bounds for r1, r2, theta
    # We can scale the whole set, so absolute values of r1, r2 don't matter, only ratio and theta.
    # But DE works better with fixed bounds. Let's fix r1=1 and optimize r2/r1 and theta?
    # Or just optimize r1, r2 in [0.1, 10] and theta in [0, pi/2].
    # Due to symmetry, theta in [0, pi/2] is sufficient (rotating both squares is same, 
    # swapping squares is same, reflecting is same).
    
    bounds_ts = [(0.1, 10.0), (0.1, 10.0), (0.0, np.pi/2)]
    
    try:
        de_ts = differential_evolution(
            two_squares_objective,
            bounds_ts,
            seed=42,
            maxiter=200,
            popsize=20,
            tol=1e-8,
            polish=True,
            polish_options={'method': 'Nelder-Mead', 'options': {'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}}
        )
        if de_ts.fun < 1e9:
            candidate_pts = two_squares_params_to_points(de_ts.x)
            best_state = try_update(candidate_pts, best_state)
    except Exception:
        pass

    # Phase 2: Refine the best point from Phase 1 using full 16-dim Nelder-Mead
    if best_state['points'] is not None:
        init_pts = best_state['points'].flatten()
    else:
        # Fallback: Regular Octagon
        angles = 2 * np.pi * np.arange(n) / n
        init_pts = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
        
    # Run multiple local optimizations from the best point and its perturbations
    start_points = [init_pts]
    
    # Add some perturbations
    for _ in range(10):
        perturb = np.random.normal(0, 0.1, (n, 2))
        start_points.append((best_state['points'] + perturb).flatten() if best_state['points'] is not None else init_pts + np.random.normal(0, 0.1, 16))
        
    # Also include the regular octagon as a baseline
    angles = 2 * np.pi * np.arange(n) / n
    reg_oct = np.column_stack([np.cos(angles), np.sin(angles)])
    start_points.append(reg_oct.flatten())
    
    # Run Nelder-Mead on each
    nm_deadline = deadline - 5.0
    for init in start_points:
        if time.time() > nm_deadline:
            break
        try:
            res = minimize(
                exact_negative_log_score,
                init,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-12,
                    'fatol': 1e-14,
                    'adaptive': True
                }
            )
            if res.fun < 1e9:
                candidate = res.x.reshape(n, 2)
                best_state = try_update(candidate, best_state)
        except Exception:
            pass

    # Phase 3: If we have time, try a general DE on 16 dims initialized near the best found
    if time.time() < deadline - 20 and best_state['points'] is not None:
        # Create bounds around the best point
        center = best_state['points'].flatten()
        # The scale of the points is roughly 1 (diameter 2). 
        # A box of +/- 2 around the center should cover good variations.
        bounds_gen = [(center[i] - 2.0, center[i] + 2.0) for i in range(2*n)]
        
        try:
            de_gen = differential_evolution(
                exact_negative_log_score,
                bounds_gen,
                seed=123,
                maxiter=50,
                popsize=10,
                tol=1e-8,
                polish=True,
                polish_options={'method': 'Nelder-Mead', 'options': {'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}}
            )
            if de_gen.fun < 1e9:
                candidate = de_gen.x.reshape(n, 2)
                best_state = try_update(candidate, best_state)
        except Exception:
            pass

    # Final check and return
    if best_state['points'] is None:
        best_state = {'score': 0, 'points': reg_oct}
        
    pts = best_state['points']
    
    # Validation
    dists = np.sqrt(np.sum((pts[:, np.newaxis, :] - pts[np.newaxis, :, :])**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-10:
        pts = reg_oct

    max_coord = np.max(np.abs(pts))
    if max_coord > 1e6:
        pts = pts / max_coord * 1e5

    return [[float(p[0]), float(p[1])] for p in pts]
