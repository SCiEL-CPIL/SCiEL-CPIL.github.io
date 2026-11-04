import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize, minimize_scalar
import time
import itertools

N = 14

# Precompute triple indices for N=14
TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)
])

def _calculate_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: (N, 2) array
    """
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    
    # Calculate Minimum Triangle Area
    # points shape (N, 2)
    # TRIPLE_INDICES shape (M, 3)
    
    # Extract points for all triples
    # This creates a (M, 3, 2) array
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]
    
    # Compute cross product (pj - pi) x (pk - pi)
    # (xj - xi)(yk - yi) - (yj - yi)(xk - xi)
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    
    min_area = 0.5 * np.min(np.abs(cross))
    
    return min_area / hull_area

def _soft_min_score(points, temp=0.05):
    """
    Differentiable approximation of min triangle area / hull area.
    Uses soft-min on triangle areas, normalized by hull area.
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    
    areas = 0.5 * np.abs(cross)
    
    # Soft min: -log(sum(exp(-areas/temp))) * temp
    # To avoid overflow, subtract max area
    max_area = np.max(areas)
    log_sum = np.log(np.sum(np.exp(-(areas - max_area) / temp)))
    soft_min_area = max_area - temp * log_sum
    
    return soft_min_area / hull_area

def _fourier_points(params):
    """
    Generate points from Fourier series parameters.
    params: [a0, b0, a1, b1, ..., aK, bK]
    r(k) = 1 + sum_{k=1}^{K} (a_k cos(2*pi*k*n/N) + b_k sin(2*pi*k*n/N))
    theta(n) = 2*pi*n/N + small perturbation? 
    For now, let's just vary radius to keep points in angular order, 
    which helps maintain convexity and avoids crossing.
    Actually, varying theta breaks the simple radial sort.
    Let's stick to radial distortion first: r_n = 1 + sum_{k=1}^K (a_k cos(...) + b_k sin(...))
    This ensures points are still in angular order if distortion is small.
    """
    K = (len(params) - 2) // 2
    a0 = params[0]
    b0 = params[1] # b0 is usually 0 or unused for radius, maybe use for global shift? No, r must be positive.
    # Let's use params[0] as base radius offset? No, let's keep base at 1.
    # params: [a1, b1, ..., aK, bK]
    
    n = np.arange(N)
    angles = 2 * np.pi * n / N
    r = np.ones(N)
    
    for k in range(1, K+1):
        a_k = params[2*(k-1)]
        b_k = params[2*(k-1)+1]
        r += a_k * np.cos(2 * np.pi * k * n / N) + b_k * np.sin(2 * np.pi * k * n / N)
        
    # Ensure r > 0.5 to maintain convexity and reasonable shape
    r = np.maximum(r, 0.5)
    
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    
    return np.column_stack((x, y))

def _neg_soft_fourier(params, temp=0.05):
    """
    Objective for optimization: negative soft-min score.
    """
    points = _fourier_points(params)
    score = _soft_min_score(points, temp)
    return -score

def _neg_hard_fourier(params):
    """
    Objective for final polish: negative hard score.
    """
    points = _fourier_points(params)
    score = _calculate_score(points)
    return -score

def _optimize_fourier(time_limit):
    """
    Optimize the Fourier radius distortion parameters.
    Uses DE for global search, then L-BFGS-B or Powell for local polish.
    """
    start_time = time.time()
    
    # Number of Fourier modes. K=3 means 6 parameters. K=5 means 10 parameters.
    # Let's try K=3 first (6 params) as it's low dimensional and captures asymmetry.
    K = 3
    n_params = 2 * K
    
    # Bounds: coefficients should be small to maintain convexity.
    # If sum |a_k| + |b_k| < 0.5, r is between 0.5 and 1.5.
    bounds = [(-0.4, 0.4)] * n_params
    
    # DE for global search
    # Use soft-min for differentiability and smoothness during DE
    best_x = None
    best_score = -np.inf
    
    try:
        res = differential_evolution(
            lambda p: _neg_soft_fourier(p, temp=0.05),
            bounds,
            seed=42,
            maxiter=50,
            popsize=15,
            tol=1e-5,
            polish=False,
            mutation=(0.3, 1.0),
            recombination=0.7
        )
        if time.time() - start_time < time_limit * 0.6:
            best_x = res.x
            best_score = -res.fun
            
            # Local polish with Powell (doesn't need gradients, robust)
            def obj_powell(p):
                return _neg_hard_fourier(p)
                
            res2 = minimize(
                obj_powell,
                best_x,
                method='Powell',
                options={'maxiter': 100, 'ftol': 1e-8, 'xtol': 1e-8}
            )
            
            if -res2.fun > best_score:
                best_score = -res2.fun
                best_x = res2.x
                
    except Exception:
        pass
    
    # Generate points from best parameters
    if best_x is not None:
        points = _fourier_points(best_x)
        final_score = _calculate_score(points)
        return final_score, points
    else:
        # Fallback: regular polygon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        points = np.column_stack((np.cos(angles), np.sin(angles)))
        return _calculate_score(points), points

def _optimize_radial_pairs(time_limit):
    """
    Optimize the radial pair configuration (7-fold symmetry).
    This is kept as a baseline/comparison.
    """
    start_time = time.time()
    
    best_score = 0.0
    best_params = (0.5, 0.0)
    
    # 1. Coarse grid search for r and phi
    r_vals = np.linspace(0.1, 0.99, 20)
    phi_vals = np.linspace(0, 2*np.pi/7, 10, endpoint=False)
    
    for r in r_vals:
        if time.time() - start_time > time_limit * 0.5:
            break
        for phi in phi_vals:
            angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
            x_out = np.cos(angles)
            y_out = np.sin(angles)
            angles_in = angles + phi
            x_in = r * np.cos(angles_in)
            y_in = r * np.sin(angles_in)
            points = np.vstack([
                np.column_stack((x_out, y_out)),
                np.column_stack((x_in, y_in))
            ])
            score = _calculate_score(points)
            if score > best_score:
                best_score = score
                best_params = (r, phi)
                
    # 2. Local polish
    if time.time() - start_time < time_limit * 0.8:
        x0 = np.array(best_params)
        
        def obj(p):
            r, phi = p
            if r < 0.01 or r > 0.99:
                return 100.0
            angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
            x_out = np.cos(angles)
            y_out = np.sin(angles)
            angles_in = angles + phi
            x_in = r * np.cos(angles_in)
            y_in = r * np.sin(angles_in)
            points = np.vstack([
                np.column_stack((x_out, y_out)),
                np.column_stack((x_in, y_in))
            ])
            return -_calculate_score(points)
            
        try:
            res = minimize(
                obj, 
                x0, 
                method='Nelder-Mead',
                options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 500}
            )
            if -res.fun > best_score:
                best_score = -res.fun
                best_params = res.x
        except Exception:
            pass
            
    # Generate points
    r, phi = best_params
    angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
    x_out = np.cos(angles)
    y_out = np.sin(angles)
    angles_in = angles + phi
    x_in = r * np.cos(angles_in)
    y_in = r * np.sin(angles_in)
    points = np.vstack([
        np.column_stack((x_out, y_out)),
        np.column_stack((x_in, y_in))
    ])
    
    return best_score, points

def _coordinate_polish(points, time_limit):
    """
    Final polish of the point coordinates to maximize min triangle area / hull area.
    """
    start_time = time.time()
    
    def obj(p):
        pts = p.reshape(-1, 2)
        return -_calculate_score(pts)
        
    x0 = points.flatten()
    
    try:
        res = minimize(
            obj,
            x0,
            method='Nelder-Mead',
            options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 1000}
        )
        
        if time.time() - start_time < time_limit:
            if -res.fun > -obj(x0):
                return res.x.reshape(-1, 2)
    except Exception:
        pass
        
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_score = 0.0
    best_points = None
    
    # Strategy 1: Fourier Radius Distortion
    # This allows asymmetric distortions of the regular 14-gon, which might be optimal.
    score1, points1 = _optimize_fourier(50.0)
    if score1 > best_score:
        best_score = score1
        best_points = points1
        
    # Strategy 2: Symmetric Radial Pairs (7-fold symmetry)
    # This is a specific family that might be competitive.
    if time.time() < deadline - 40:
        score2, points2 = _optimize_radial_pairs(30.0)
        if score2 > best_score:
            best_score = score2
            best_points = points2

    # Final Coordinate Polish
    if best_points is not None and time.time() < deadline - 5:
        best_points = _coordinate_polish(best_points, deadline - time.time() - 2)
        final_score = _calculate_score(best_points)
        if final_score > best_score:
            best_score = final_score

    # Fallback
    if best_points is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_points
