"""Heilbronn convex regions: maximize min triangle area / hull area for 13 points."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
import itertools

N = 13
TIME_LIMIT = 115.0  # Safety margin before 120s kill

def _compute_score(points):
    """Compute min triangle area / convex hull area."""
    points = np.asarray(points)
    n = len(points)
    if n < 3:
        return 0.0
    # Convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-12:
            return 0.0
    except Exception:
        return 0.0
    # Min triangle area
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = points[i]
                b = points[j]
                c = points[k]
                area = 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1]))
                if area < min_area:
                    min_area = area
    return min_area / hull_area

def _smooth_min_triangle_area(points, beta=50.0):
    """
    Smooth approximation of min triangle area using log-sum-exp.
    We maximize min area, so we minimize -log(sum(exp(-beta * area)))
    which approximates -min(area) for large beta.
    """
    points = np.asarray(points)
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = points[i]
                b = points[j]
                c = points[k]
                area = 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1]))
                areas.append(area)
    areas = np.array(areas)
    if len(areas) == 0:
        return 0.0
    # log-sum-exp trick for numerical stability
    m = np.min(areas)
    log_sum = m + np.log(np.sum(np.exp(areas - m)))
    # We want to minimize the negative of the smooth min
    # smooth_min ≈ -log_sum_exp(-beta * areas) / beta
    # = -(1/beta) * log(sum(exp(-beta * area)))
    neg_beta_areas = -beta * areas
    m2 = np.min(neg_beta_areas)
    log_sum_exp = m2 + np.log(np.sum(np.exp(neg_beta_areas - m2)))
    smooth_min = -log_sum_exp / beta
    return smooth_min

def _hull_area(points):
    """Compute convex hull area."""
    points = np.asarray(points)
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1.0

def _objective(params, beta=50.0):
    """
    Objective function for minimization.
    params: 2*N parameters - angles and radii for N points
    We want to maximize min_triangle_area / hull_area
    So minimize -min_triangle_area / hull_area
    """
    angles = params[:N]
    radii = params[N:]
    # Ensure positive radii
    radii = np.maximum(radii, 0.01)
    # Compute points
    points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    # Sort by angle to ensure convex polygon
    order = np.argsort(angles)
    points = points[order]
    
    ha = _hull_area(points)
    if ha < 1e-12:
        return 1e6
    
    smooth_min = _smooth_min_triangle_area(points, beta=beta)
    score = smooth_min / ha
    return -score  # Negate for minimization

def _objective_with_grad(params, beta=50.0):
    """Same as objective but could add gradient later."""
    return _objective(params, beta=beta)

def construct():
    start_time = time.time()
    best_score = 0.0
    best_points = None
    
    # Initial guess: regular 13-gon
    angles_init = np.linspace(0, 2*np.pi, N, endpoint=False)
    radii_init = np.ones(N)
    params_init = np.concatenate([angles_init, radii_init])
    
    # Try multiple initializations and optimizations
    attempts = 0
    max_attempts = 8
    
    for attempt in range(max_attempts):
        if time.time() - start_time > TIME_LIMIT - 5.0:
            break
        
        if attempt == 0:
            params = params_init.copy()
        else:
            # Perturb the initial guess
            rng = np.random.RandomState(attempt * 12345 + 67890)
            angles_pert = angles_init + rng.randn(N) * 0.1
            radii_pert = 1.0 + rng.randn(N) * 0.1
            radii_pert = np.maximum(radii_pert, 0.5)
            params = np.concatenate([angles_pert, radii_pert])
        
        # Use L-BFGS-B for optimization
        try:
            result = minimize(
                _objective_with_grad,
                params,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            if time.time() - start_time > TIME_LIMIT - 3.0:
                break
            
            params_opt = result.x
            angles_opt = params_opt[:N]
            radii_opt = np.maximum(params_opt[N:], 0.01)
            points_opt = np.column_stack([radii_opt * np.cos(angles_opt), radii_opt * np.sin(angles_opt)])
            order = np.argsort(angles_opt)
            points_opt = points_opt[order]
            
            score_opt = _compute_score(points_opt)
            if score_opt > best_score:
                best_score = score_opt
                best_points = points_opt.copy()
                
        except Exception as e:
            pass
    
    # If no good result, fall back to regular polygon
    if best_points is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Final check: ensure it's valid
    final_score = _compute_score(best_points)
    if final_score <= 0:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points

# EVOLVE-BLOCK-START
# (construct function defined above)
# EVOLVE-BLOCK-END
