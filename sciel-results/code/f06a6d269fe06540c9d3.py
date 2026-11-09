import numpy as np
import time
import scipy.spatial.distance as spd
from scipy.optimize import minimize

def calculate_score(points):
    """Calculate the metric R = (d_min / d_max)^2"""
    if len(points) < 2:
        return 0.0
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0:
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio

def soft_min_max_ratio(points, beta):
    """
    Compute a smooth surrogate for (d_min / d_max)^2.
    Returns log(R) for optimization stability.
    """
    dists = spd.pdist(points)
    if len(dists) == 0:
        return -1e6
    
    x_max = dists / beta
    m_max = np.max(x_max)
    log_sum_exp_max = m_max + np.log(np.sum(np.exp(x_max - m_max)))
    d_max_soft = beta * log_sum_exp_max
    
    x_min = -dists / beta
    m_min = np.max(x_min)
    log_sum_exp_min = m_min + np.log(np.sum(np.exp(x_min - m_min)))
    d_min_soft = -beta * log_sum_exp_min
    
    if d_max_soft <= 0 or d_min_soft <= 0:
        return -1e6
        
    log_ratio = np.log(d_min_soft) - np.log(d_max_soft)
    return 2 * log_ratio

def points_from_params(params):
    n = 16
    pts = np.zeros((n, 2))
    pts[0] = [0.0, 0.0]
    pts[1] = [1.0, 0.0]
    rest = params.reshape(14, 2)
    pts[2:] = rest
    return pts

def points_to_params(pts):
    """
    Convert a 16x2 point array to the 28-dim parameter vector
    assuming pts[0] and pts[1] are the fixed points.
    """
    p0 = pts[0].copy()
    p1 = pts[1].copy()
    
    pts_trans = pts - p0
    d = np.linalg.norm(p1)
    
    if d < 1e-9:
        # Fallback: pick farthest point as P1
        dists = np.linalg.norm(pts_trans, axis=1)
        idx = np.argmax(dists)
        if idx == 0:
            return None
        p1_new = pts_trans[idx]
        d = np.linalg.norm(p1_new)
        if d < 1e-9:
            return None
        # Rotate p1_new to x-axis
        angle = np.arctan2(p1_new[1], p1_new[0])
        rot = np.array([[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]])
        pts_rot = pts_trans @ rot.T
        pts_scaled = pts_rot / d
        # Now pts_rot[idx] is (1,0).
        # We need to reorder points so that idx is at position 1.
        # This changes the identity of points, but for init it's fine.
        # Move pts_rot[idx] to index 1.
        # This is complex. Let's just return None for this case in init generation.
        return None

    angle = np.arctan2(p1[1], p1[0])
    rot = np.array([[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]])
    pts_rot = pts_trans @ rot.T
    pts_scaled = pts_rot / d
    
    # Check if pts_scaled[1] is close to (1,0)
    if not np.allclose(pts_scaled[1], [1.0, 0.0]):
        return None
        
    params = pts_scaled[2:].flatten()
    return params

def min_max_dist_dim2_16() -> np.ndarray:
    n = 16
    d = 2
    
    global_deadline = time.time() + 620 # Safety margin
    
    def objective(params, beta):
        try:
            pts = points_from_params(params)
            if not np.all(np.isfinite(pts)):
                return 1e6
            return -soft_min_max_ratio(pts, beta)
        except Exception:
            return 1e6

    def objective_exact(params):
        try:
            pts = points_from_params(params)
            if not np.all(np.isfinite(pts)):
                return 1e6
            score = calculate_score(pts)
            return -score
        except Exception:
            return 1e6

    best_params = None
    best_score = 0.0
    
    rng = np.random.RandomState(42)
    
    inits = []
    
    # Random inits
    for _ in range(5):
        inits.append(rng.randn(28) * 2.0)
        
    # Hexagonal init
    hex_pts = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i + 0.5 * j
            y = np.sqrt(3) / 2 * j
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts)
    hex_pts -= np.mean(hex_pts, axis=0)
    d_center = np.linalg.norm(hex_pts, axis=1)
    idx = np.argsort(d_center)[:16]
    hex_init = hex_pts[idx]
    hex_init -= np.mean(hex_init, axis=0)
    
    # Try to convert hex_init to params
    # We need to pick two points to be P0 and P1.
    # Let's pick the two closest points in hex_init as P0 and P1.
    dists_hex = spd.pdist(hex_init)
    # Find min distance pair
    min_idx = np.argmin(dists_hex)
    # Convert linear index to (i, j)
    # 16 points -> 120 pairs
    # This is tedious. Let's just pick P0=hex_init[0], P1=hex_init[1] and hope they are close.
    # Or better, sort by x, then y.
    hex_sorted = hex_init[np.lexsort((hex_init[:, 1], hex_init[:, 0]))]
    p0_hex = hex_sorted[0]
    p1_hex = hex_sorted[1]
    
    # Construct a candidate with P0, P1 fixed
    cand = hex_sorted.copy()
    cand[0] = p0_hex
    cand[1] = p1_hex
    # Translate and rotate
    p0 = cand[0]
    p1 = cand[1]
    cand_trans = cand - p0
    d = np.linalg.norm(p1)
    if d > 1e-9:
        angle = np.arctan2(p1[1], p1[0])
        rot = np.array([[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]])
        cand_rot = cand_trans @ rot.T
        cand_scaled = cand_rot / d
        params_hex = cand_scaled[2:].flatten()
        inits.append(params_hex)

    # Ring init
    angles = np.linspace(0, 2*np.pi, 14, endpoint=False)
    r = 2.0
    circle_init = np.column_stack((r*np.cos(angles), r*np.sin(angles))).flatten()
    inits.append(circle_init)

    start_time = time.time()
    
    betas = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    
    current_params = None
    
    for beta in betas:
        if time.time() > global_deadline - 100:
            break
            
        best_local_params = None
        best_local_score = 1e6 # Minimize negative log ratio
        
        for init in inits:
            if time.time() > global_deadline - 100:
                break
                
            try:
                res = minimize(
                    objective, init, args=(beta,), 
                    method='L-BFGS-B', 
                    options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-5}
                )
                if res.fun < best_local_score:
                    best_local_score = res.fun
                    best_local_params = res.x
            except Exception:
                pass
        
        if best_local_params is not None:
            current_params = best_local_params
            pts = points_from_params(current_params)
            score = calculate_score(pts)
            if score > best_score:
                best_score = score
                best_params = current_params.copy()
            
            inits.insert(0, best_local_params)
            if len(inits) > 4:
                inits = inits[:4]

    if time.time() < global_deadline - 60:
        if best_params is not None:
            start_polish = best_params
        else:
            start_polish = inits[0]
            
        try:
            res = minimize(
                objective_exact, start_polish, 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'fatol': 1e-10, 'xatol': 1e-10}
            )
            if -res.fun > best_score:
                best_score = -res.fun
                best_params = res.x
        except Exception:
            pass
        
        if time.time() < global_deadline - 30 and best_params is not None:
            try:
                res = minimize(
                    objective_exact, best_params, 
                    method='Nelder-Mead', 
                    options={'maxiter': 200, 'fatol': 1e-12, 'xatol': 1e-12}
                )
                if -res.fun > best_score:
                    best_score = -res.fun
                    best_params = res.x
            except Exception:
                pass

    if best_params is None:
        best_params = rng.randn(28)
        
    final_points = points_from_params(best_params)
    
    if not np.all(np.isfinite(final_points)):
        final_points = np.zeros((n, d))
        
    return final_points
