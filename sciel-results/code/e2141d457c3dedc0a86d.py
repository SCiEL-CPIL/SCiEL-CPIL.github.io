import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
import time

N = 200
_TIME_LIMIT = 145  # Leave 5s margin for 150s kill

def _score_from_points(pts):
    """Calculate the maximum radius for a set of points."""
    if len(pts) < 2:
        return 0.0
    
    # Check bounds
    if np.any(pts[:, 0] < -0.5) or np.any(pts[:, 0] > 0.5) or \
       np.any(pts[:, 1] < -0.5) or np.any(pts[:, 1] > 0.5):
        return 0.0

    # Min distance to wall
    wall_dist = np.min([
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1]),
    ])
    
    # Min pairwise distance
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D)
    
    r = min(wall_dist, min_pair / 2.0)
    return r if r > 0 else 0.0

def _eval_hex_params(params, rows, cols):
    """
    Evaluate radius for a hexagonal-like grid.
    params: [dx, dy, shift, x0, y0]
    """
    dx, dy, shift, x0, y0 = params
    
    if dx <= 0 or dy <= 0:
        return 0.0

    pts = []
    for i in range(rows):
        y = y0 + i * dy
        s = (shift * dx) if (i % 2 == 1) else 0.0
        for j in range(cols):
            x = x0 + j * dx + s
            pts.append([x, y])
            
    pts = np.array(pts)
    
    if len(pts) > N:
        dists = np.sum(pts**2, axis=1)
        indices = np.argsort(dists)[:N]
        pts = pts[indices]
    elif len(pts) < N:
        return 0.0

    return _score_from_points(pts)

def _neg_eval_hex_params(params, rows, cols):
    return -_eval_hex_params(params, rows, cols)

def _local_refine(pts, time_left):
    """
    Perform a local optimization on the points using a repulsion-based objective.
    We minimize the sum of squared violations of the minimum distance constraint.
    """
    if time_left < 10:
        return pts
    
    def objective(x):
        p = x.reshape(-1, 2)
        # Bounds penalty
        if np.any(p[:, 0] < -0.5) or np.any(p[:, 0] > 0.5) or \
           np.any(p[:, 1] < -0.5) or np.any(p[:, 1] > 0.5):
            return 1e10
        
        # Pairwise distances
        D = cdist(p, p)
        np.fill_diagonal(D, np.inf)
        
        # Wall distances
        wall_d = np.min([
            np.min(p[:, 0]) + 0.5,
            0.5 - np.max(p[:, 0]),
            np.min(p[:, 1]) + 0.5,
            0.5 - np.max(p[:, 1]),
        ])
        
        # Minimize sum of soft violations.
        # If we assume the target radius is r, we want d_ij >= 2r and wall_d >= r.
        # Since we don't know r, let's just maximize the minimum.
        # A smooth proxy: minimize sum(1/d_ij^2) + sum(1/wall_d_i^2)
        
        eps = 1e-6
        pair_term = np.sum(1.0 / (D + eps)**2)
        
        # Wall terms
        wall_x_left = p[:, 0] + 0.5
        wall_x_right = 0.5 - p[:, 0]
        wall_y_left = p[:, 1] + 0.5
        wall_y_right = 0.5 - p[:, 1]
        
        wall_term = np.sum(1.0 / (wall_x_left + eps)**2) + \
                    np.sum(1.0 / (wall_x_right + eps)**2) + \
                    np.sum(1.0 / (wall_y_left + eps)**2) + \
                    np.sum(1.0 / (wall_y_right + eps)**2)
        
        return pair_term + wall_term
    
    x0 = pts.flatten()
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 200, 'ftol': 1e-10})
        new_pts = res.x.reshape(-1, 2)
        
        r_old = _score_from_points(pts)
        r_new = _score_from_points(new_pts)
        
        if r_new > r_old:
            return new_pts
        else:
            return pts
            
    except Exception:
        return pts

def construct():
    t0 = time.time()
    best_r = 0.0
    best_pts = None
    
    grid_configs = [
        (14, 15), (15, 14),
        (13, 16), (16, 13),
        (12, 17), (17, 12),
        (10, 20), (20, 10),
        (11, 19), (19, 11),
        (15, 15),
    ]
    
    bounds = [
        (0.03, 0.06),
        (0.03, 0.06),
        (0.0, 1.0),
        (-0.1, 0.1),
        (-0.1, 0.1),
    ]
    
    for rows, cols in grid_configs:
        if time.time() - t0 > _TIME_LIMIT:
            break
            
        dx0 = 1.0 / cols
        dy0 = 1.0 / rows
        x00 = 0.0
        y00 = 0.0
        shift0 = 0.5
        
        x0 = np.array([dx0, dy0, shift0, x00, y00])
        
        try:
            result = differential_evolution(
                _neg_eval_hex_params, 
                bounds, 
                args=(rows, cols),
                seed=42,
                maxiter=100,
                tol=1e-8,
                popsize=10,
                polish=True
            )
            
            r = -result.fun
            if r > best_r:
                best_r = r
                dx, dy, shift, x0, y0 = result.x
                pts = []
                for i in range(rows):
                    y = y0 + i * dy
                    s = (shift * dx) if (i % 2 == 1) else 0.0
                    for j in range(cols):
                        x = x0 + j * dx + s
                        pts.append([x, y])
                pts = np.array(pts)
                if len(pts) > N:
                    dists = np.sum(pts**2, axis=1)
                    indices = np.argsort(dists)[:N]
                    pts = pts[indices]
                
                if len(pts) == N:
                    if _score_from_points(pts) > 0:
                        best_pts = pts
                        
        except Exception:
            pass
            
    if time.time() - t0 < _TIME_LIMIT - 10:
        for rows, cols in grid_configs[:4]:
            if time.time() - t0 > _TIME_LIMIT:
                break
            for k in range(5):
                if time.time() - t0 > _TIME_LIMIT:
                    break
                dx0 = np.random.uniform(0.04, 0.06)
                dy0 = np.random.uniform(0.04, 0.06)
                shift0 = np.random.uniform(0, 1)
                x00 = np.random.uniform(-0.1, 0.1)
                y00 = np.random.uniform(-0.1, 0.1)
                x0 = np.array([dx0, dy0, shift0, x00, y00])
                
                try:
                    res = minimize(_neg_eval_hex_params, x0, args=(rows, cols), method='Nelder-Mead', 
                                   options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
                    r = -res.fun
                    if r > best_r:
                        best_r = r
                        dx, dy, shift, x0, y0 = res.x
                        pts = []
                        for i in range(rows):
                            y = y0 + i * dy
                            s = (shift * dx) if (i % 2 == 1) else 0.0
                            for j in range(cols):
                                x = x0 + j * dx + s
                                pts.append([x, y])
                        pts = np.array(pts)
                        if len(pts) > N:
                            dists = np.sum(pts**2, axis=1)
                            indices = np.argsort(dists)[:N]
                            pts = pts[indices]
                        if len(pts) == N and _score_from_points(pts) > 0:
                            best_pts = pts
                except Exception:
                    pass

    if best_pts is None:
        rows, cols = 14, 15
        dx = 1.0 / cols
        dy = 1.0 / rows
        xs = -0.5 + (np.arange(cols) + 0.5) * dx
        ys = -0.5 + (np.arange(rows) + 0.5) * dy
        grid = np.array([(x, y) for y in ys for x in xs])
        best_pts = grid[:N]
        
    # Refine the best solution found so far
    if time.time() - t0 < _TIME_LIMIT - 5:
        best_pts = _local_refine(best_pts, _TIME_LIMIT - (time.time() - t0))
        
    return best_pts
