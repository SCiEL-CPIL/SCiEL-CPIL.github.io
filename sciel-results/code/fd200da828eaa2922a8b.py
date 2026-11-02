import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

N = 14

def _compute_min_triangle_area(points):
    """
    Compute the minimum area of any triangle formed by triples of points.
    points: (N, 2) array
    Returns: float
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Vectorized computation of all triangle areas
    # i < j < k
    areas = []
    # For N=14, 364 triangles, loop is fine, but we can do better
    # Let's use a simple loop for clarity and correctness, it's fast enough
    # 364 * (few ops) is negligible compared to optimization overhead
    min_area = float('inf')
    
    # Pre-compute cross products? 
    # Area = 0.5 * | (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i) |
    
    # To speed up, we can iterate
    for i in range(n - 2):
        xi, yi = points[i]
        for j in range(i + 1, n - 1):
            xj, yj = points[j]
            dx_ij = xj - xi
            dy_ij = yj - yi
            for k in range(j + 1, n):
                xk, yk = points[k]
                dx_ik = xk - xi
                dy_ik = yk - yi
                # Cross product magnitude
                cross = dx_ij * dy_ik - dy_ij * dx_ik
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
                    
    return min_area

def _compute_hull_area(points):
    """
    Compute the area of the convex hull of the points.
    points: (N, 2) array
    Returns: float
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        # If points are collinear or degenerate, area is 0
        return 0.0

def _score(points):
    """
    Calculate the score: min_triangle_area / hull_area
    """
    points = np.asarray(points, dtype=float)
    if points.shape != (N, 2) or np.any(~np.isfinite(points)):
        return 0.0
    
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
        
    min_area = _compute_min_triangle_area(points)
    if min_area <= 0.0:
        return 0.0
        
    return min_area / hull_area

def _params_to_points(params):
    """
    Convert parameters (log_radii, angles) to points.
    params: array of length 2*N
    First N: log(r_i)
    Next N: theta_i
    """
    log_r = params[:N]
    theta = params[N:]
    
    r = np.exp(log_r)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    return np.column_stack((x, y))

def _objective(params):
    """
    Objective function to minimize (negative score).
    """
    points = _params_to_points(params)
    score = _score(points)
    if score <= 1e-12:
        return 1e10
    return -score

def _generate_initial_params():
    """
    Generate diverse initial parameter sets.
    Returns a list of param arrays.
    """
    inits = []
    
    # 1. Regular 14-gon
    # r = 1, theta = 2*pi*k/14
    r_reg = np.ones(N)
    theta_reg = 2 * np.pi * np.arange(N) / N
    params_reg = np.concatenate((np.log(r_reg), theta_reg))
    inits.append(params_reg)
    
    # 2. Ellipses
    # x = a * cos(t), y = b * sin(t)
    # r(t) = sqrt(a^2 cos^2(t) + b^2 sin^2(t))
    for aspect in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
        a = 1.0
        b = aspect
        theta_ell = 2 * np.pi * np.arange(N) / N
        r_ell = np.sqrt(a**2 * np.cos(theta_ell)**2 + b**2 * np.sin(theta_ell)**2)
        params_ell = np.concatenate((np.log(r_ell), theta_ell))
        inits.append(params_ell)
        
    # 3. Random perturbations of regular polygon
    rng = np.random.default_rng(42)
    for _ in range(20):
        r_rand = 1.0 + rng.normal(0, 0.1, N)
        r_rand = np.clip(r_rand, 0.5, 2.0)
        theta_rand = theta_reg + rng.normal(0, 0.1, N)
        # Ensure sorted angles for consistent indexing? 
        # The parameterization doesn't strictly require sorted angles for the geometry,
        # but the "convex hull" calculation handles any order.
        # However, for the "min triangle" to be meaningful in the context of convex position,
        # we assume the points are vertices of the convex hull.
        # If angles are not sorted, the polygon might self-intersect.
        # Let's sort angles to maintain convex order.
        order = np.argsort(theta_rand)
        theta_sorted = theta_rand[order]
        r_sorted = r_rand[order]
        params_rand = np.concatenate((np.log(r_sorted), theta_sorted))
        inits.append(params_rand)
        
    # 4. Random convex polygons
    for _ in range(20):
        theta_rand = np.sort(rng.uniform(0, 2*np.pi, N))
        r_rand = rng.uniform(0.5, 1.5, N)
        params_rand = np.concatenate((np.log(r_rand), theta_rand))
        inits.append(params_rand)
        
    return inits

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # 5s safety margin
    
    best_score = 0.0
    best_params = None
    best_points = None
    
    inits = _generate_initial_params()
    
    for init_params in inits:
        if time.time() > deadline - 10:
            break
            
        # Try Nelder-Mead
        try:
            res = minimize(
                _objective, 
                init_params, 
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            if res.success or np.isfinite(res.fun):
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_params = res.x.copy()
                    best_points = _params_to_points(res.x)
                    
        except Exception:
            continue
            
        # If we have a good candidate, try a refinement with Powell
        if best_score > 0 and time.time() < deadline - 5:
            try:
                res2 = minimize(
                    _objective,
                    best_params,
                    method='Powell',
                    options={
                        'maxiter': 500,
                        'xtol': 1e-12,
                        'ftol': 1e-14
                    }
                )
                score2 = -res2.fun
                if score2 > best_score:
                    best_score = score2
                    best_params = res2.x.copy()
                    best_points = _params_to_points(res2.x)
            except Exception:
                pass
                
    # Final fallback if nothing found
    if best_points is None:
        # Return regular polygon
        r_reg = np.ones(N)
        theta_reg = 2 * np.pi * np.arange(N) / N
        best_points = _params_to_points(np.concatenate((np.log(r_reg), theta_reg)))
        
    # Normalize to unit area
    hull_area = _compute_hull_area(best_points)
    if hull_area > 1e-12:
        scale = np.sqrt(1.0 / hull_area)
        best_points = best_points * scale
    else:
        # Should not happen
        best_points = best_points * 1.0
        
    return best_points
