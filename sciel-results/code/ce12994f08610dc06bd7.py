# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = 600.0  # safety margin before 630s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _vertex_score(p, i):
    """Compute the per-vertex score for vertex i of polygon p."""
    n = len(p)
    d = np.linalg.norm(p[i] - p, axis=1)
    d_sorted = np.sort(d)
    d_sorted = d_sorted[1:]  # remove self-distance (index 0 after sort may be self)
    # Actually, d[0] is distance to self = 0. Remove it.
    d_sorted = np.sort(d[1:])  # distances to other 19 vertices
    if len(d_sorted) != 19:
        return 0.0
    if np.any(d_sorted < 0.05):
        return 0.0
    diffs = np.diff(d_sorted)
    k = int(np.argmin(diffs))
    D = (d_sorted[k] + d_sorted[k+1]) / 2.0
    if D < 0.1:
        return 0.0
    if k == 0:
        four = d_sorted[0:4]
    elif k == 17:
        four = d_sorted[15:19]
    else:
        four = d_sorted[k-1:k+3]
    ratios = np.maximum(four / D, D / four)
    raw = np.mean(ratios)
    return min(1.0, 1.0 / raw)

def _min_vertex_score(p):
    """Minimum vertex score over all vertices."""
    n = len(p)
    scores = np.array([_vertex_score(p, i) for i in range(n)])
    return float(np.min(scores))

def _is_valid(p):
    """Check if polygon is strictly convex with no repeated points."""
    n = len(p)
    if n < 3:
        return False
    # No repeated points
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(p[i] - p[j]) < 1e-9:
                return False
    # Check convexity: all cross products same sign
    cross_products = []
    for i in range(n):
        p0 = p[i]
        p1 = p[(i+1) % n]
        p2 = p[(i+2) % n]
        e1 = p1 - p0
        e2 = p2 - p1
        cp = e1[0]*e2[1] - e1[1]*e2[0]
        cross_products.append(cp)
    cross_products = np.array(cross_products)
    if np.any(cross_products == 0):
        return False
    if np.all(cross_products > 0) or np.all(cross_products < 0):
        return True
    return False

def _normalize(p):
    """Normalize polygon: subtract centroid, divide by mean distance."""
    centroid = np.mean(p, axis=0)
    p = p - centroid
    dists = np.linalg.norm(p, axis=1)
    mean_dist = np.mean(dists)
    if mean_dist <= 0.01:
        return p
    return p / mean_dist

def _smooth_min_vertex_score(p, beta=50.0):
    """Smooth approximation of min vertex score using log-sum-exp."""
    n = len(p)
    scores = np.array([_vertex_score(p, i) for i in range(n)])
    # Soft min: -log(sum(exp(-beta*s)))/beta
    neg_beta_scores = -beta * scores
    # Clip for numerical stability
    neg_beta_scores = np.clip(neg_beta_scores, -500, 500)
    log_sum = np.log(np.sum(np.exp(neg_beta_scores)))
    soft_min = -log_sum / beta
    return soft_min

def _objective(params, beta=50.0):
    """Objective to minimize: -smooth_min_vertex_score."""
    n = 20
    angles = params[:n]
    radii = params[n:]
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    p = np.stack([x, y], axis=1)
    p = _normalize(p)
    if not _is_valid(p):
        return 1000.0
    return -_smooth_min_vertex_score(p, beta)

def _regular_polygon_20():
    n = 20
    angles = np.linspace(0.0, 2.0*np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.stack([x, y], axis=1)

def _search():
    n = 20
    best_score = 0.0
    best_p = _regular_polygon_20()
    
    # Start from regular polygon
    angles0 = np.linspace(0.0, 2.0*np.pi, n, endpoint=False)
    radii0 = np.ones(n)
    params0 = np.concatenate([angles0, radii0])
    
    # Multiple restarts with perturbations
    np.random.seed(42)
    n_restarts = 30
    max_iters_per_restart = 300
    
    for restart in range(n_restarts):
        if _time_left() < 30.0:
            break
        
        if restart == 0:
            params = params0.copy()
        else:
            # Perturb angles and radii
            angle_pert = np.random.normal(0, 0.1, n)
            # Keep angles sorted to maintain convexity
            angles_pert = angles0 + angle_pert
            # Re-sort angles to maintain cyclic order
            angles_pert = np.sort(angles_pert)
            radii_pert = np.random.uniform(0.7, 1.3, n)
            params = np.concatenate([angles_pert, radii_pert])
        
        # Try several beta values
        for beta in [20.0, 50.0, 100.0]:
            if _time_left() < 10.0:
                break
            
            try:
                result = minimize(
                    _objective,
                    params,
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': max_iters_per_restart, 'maxfev': max_iters_per_restart*2}
                )
                
                if result.success or result.fun < 0:
                    params_opt = result.x
                    angles_opt = params_opt[:n]
                    radii_opt = params_opt[n:]
                    x = radii_opt * np.cos(angles_opt)
                    y = radii_opt * np.sin(angles_opt)
                    p = np.stack([x, y], axis=1)
                    p = _normalize(p)
                    
                    if _is_valid(p):
                        score = _min_vertex_score(p)
                        if score > best_score:
                            best_score = score
                            best_p = p.copy()
            except Exception:
                continue
        
        # Also try Nelder-Mead for local refinement
        if _time_left() < 10.0:
            break
        
        try:
            result2 = minimize(
                _objective,
                params,
                args=(50.0,),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            if result2.fun < 0:
                params_opt = result2.x
                angles_opt = params_opt[:n]
                radii_opt = params_opt[n:]
                x = radii_opt * np.cos(angles_opt)
                y = radii_opt * np.sin(angles_opt)
                p = np.stack([x, y], axis=1)
                p = _normalize(p)
                if _is_valid(p):
                    score = _min_vertex_score(p)
                    if score > best_score:
                        best_score = score
                        best_p = p.copy()
        except Exception:
            continue
    
    return best_p, best_score

def _polish(p):
    """High-precision polish pass using coordinate descent."""
    n = len(p)
    p = p.copy()
    best_score = _min_vertex_score(p)
    
    # Try small perturbations of each vertex
    for iteration in range(200):
        if _time_left() < 5.0:
            break
        
        improved = False
        for i in range(n):
            for dx in [-0.01, 0.01, 0.005, -0.005, 0.002, -0.002]:
                for dy in [-0.01, 0.01, 0.005, -0.005, 0.002, -0.002]:
                    p_new = p.copy()
                    p_new[i] = p_new[i] + np.array([dx, dy])
                    p_new = _normalize(p_new)
                    if _is_valid(p_new):
                        score = _min_vertex_score(p_new)
                        if score > best_score + 1e-8:
                            p = p_new
                            best_score = score
                            improved = True
                            break
                if improved:
                    break
        if not improved:
            break
    
    return p, best_score

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices that tries to maximize the minimum
    per-vertex equidistance score. Returns a (20, 2) array of finite vertex
    coordinates in cyclic order.
    """
    # Search for best polygon
    p, score = _search()
    
    # Polish
    if _time_left() > 10.0:
        p, score = _polish(p)
    
    # Final normalization check
    p = _normalize(p)
    
    return p

# EVOLVE-BLOCK-END
