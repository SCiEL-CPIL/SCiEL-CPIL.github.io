import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize_scalar, minimize
import time

N = 14
N_OUT = 7
N_IN = 7

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
        hull_area = hull.volume  # In 2D, volume is area
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0

    # Calculate Minimum Triangle Area
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]

    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])

    min_area = 0.5 * np.min(np.abs(cross))

    return min_area / hull_area

def _generate_points_77(r, theta):
    """Generate points for 7-7 ring configuration."""
    angles_out = np.linspace(0, 2 * np.pi, N_OUT, endpoint=False)
    x_out = np.cos(angles_out)
    y_out = np.sin(angles_out)

    angles_in = np.linspace(0, 2 * np.pi, N_IN, endpoint=False) + theta
    x_in = r * np.cos(angles_in)
    y_in = r * np.sin(angles_in)

    return np.vstack([
        np.column_stack((x_out, y_out)),
        np.column_stack((x_in, y_in))
    ])

def _score_77(r, theta):
    """Score for 7-7 ring configuration."""
    if r < 0 or r > 1:
        return 0.0
    points = _generate_points_77(r, theta)
    return _calculate_score(points)

def _score_77_neg(r, theta):
    """Negative score for minimization."""
    return -_score_77(r, theta)

def _optimize_77_analytic_r(theta, time_deadline):
    """
    For a fixed theta, find the optimal r by 1D search.
    """
    start_time = time.time()
    
    def neg_score_r(r):
        return -_score_77(r, theta)
    
    # Search r in [0.1, 0.95]
    result = minimize_scalar(
        neg_score_r,
        bounds=(0.1, 0.95),
        method='bounded',
        options={'xatol': 1e-8, 'maxiter': 100}
    )
    
    if result.success:
        best_r = result.x
        best_score = -result.fun
    else:
        # Fallback to midpoint
        best_r = 0.5
        best_score = _score_77(best_r, theta)
    
    return best_r, best_score

def _optimize_77_1d_theta(time_deadline):
    """
    Optimize over theta in [0, 2*pi/7] using 1D search.
    For each theta, optimize r analytically.
    """
    start_time = time.time()
    
    def neg_score_theta(theta):
        if time.time() > time_deadline:
            return 0.0
        r, score = _optimize_77_analytic_r(theta, time_deadline)
        return -score
    
    # The symmetry period is 2*pi/7 for 7-7 configuration
    theta_max = 2 * np.pi / 7
    
    result = minimize_scalar(
        neg_score_theta,
        bounds=(0, theta_max),
        method='bounded',
        options={'xatol': 1e-8, 'maxiter': 50}
    )
    
    if result.success:
        best_theta = result.x
        r, best_score = _optimize_77_analytic_r(best_theta, time_deadline)
    else:
        # Try a few heuristic points
        best_score = 0.0
        best_theta = 0.0
        best_r = 0.5
        for theta_try in [0.0, theta_max/2, theta_max/3, 2*theta_max/3]:
            if time.time() > time_deadline:
                break
            r_try, score_try = _optimize_77_analytic_r(theta_try, time_deadline)
            if score_try > best_score:
                best_score = score_try
                best_theta = theta_try
                best_r = r_try
    
    return best_r, best_theta, best_score

def _soft_min_score(points, beta=50.0):
    """
    Soft-min surrogate for the minimum triangle area.
    """
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    
    areas = 0.5 * np.abs(cross)
    
    # Soft-min: -1/beta * log(sum(exp(-beta * areas)))
    max_area = np.max(areas)
    shifted = -beta * (areas - max_area)
    log_sum = np.log(np.sum(np.exp(shifted))) + beta * max_area
    soft_min = max_area - log_sum / beta
    
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    
    return soft_min / hull_area

def _soft_score_77(params, beta=50.0):
    """Soft-min surrogate objective for 7-7 configuration."""
    r, theta = params
    if r < 0 or r > 1:
        return 0.0
    points = _generate_points_77(r, theta)
    return _soft_min_score(points, beta)

def _polish_77(r, theta, time_deadline):
    """
    High-precision polish using L-BFGS-B on the soft-min surrogate.
    """
    start_time = time.time()
    
    # Progressive refinement with increasing beta
    betas = [20.0, 50.0, 100.0, 200.0, 500.0]
    current_params = np.array([r, theta])
    theta_max = 2 * np.pi / 7
    bounds = [(0.05, 0.999), (0.0, theta_max)]
    
    for beta in betas:
        if time.time() > time_deadline:
            break
        
        def neg_soft(params):
            return -_soft_score_77(params, beta)
        
        try:
            res = minimize(
                neg_soft,
                current_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'ftol': 1e-12, 'gtol': 1e-10, 'maxiter': 500}
            )
            if res.success:
                current_params = res.x
        except Exception:
            pass
    
    # Final evaluation with exact score
    final_r, final_theta = current_params
    final_score = _score_77(final_r, final_theta)
    
    return final_r, final_theta, final_score

def _optimize_ring(n_out, n_in, time_limit):
    """
    Optimize a specific ring configuration.
    For 7-7, use the analytic r + 1D theta approach.
    For others, use a simpler search.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    if n_out == 7 and n_in == 7:
        # Use the specialized 7-7 optimization
        r, theta, score = _optimize_77_1d_theta(deadline)
        
        # Polish if time remains
        if time.time() < deadline - 2:
            r, theta, score = _polish_77(r, theta, deadline)
        
        return score, (r, theta)
    else:
        # For other configurations, do a simple grid search
        best_score = 0.0
        best_params = None
        
        r_vals = np.linspace(0.3, 0.9, 10)
        g = np.gcd(n_out, n_in)
        theta_max = 2 * np.pi / g
        theta_vals = np.linspace(0, theta_max, 10, endpoint=False)
        
        for r in r_vals:
            if time.time() > deadline:
                break
            for theta in theta_vals:
                if time.time() > deadline:
                    break
                score = _score_77(r, theta)  # This won't work for non-7-7
                # Need a more general approach
                angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
                x_out = np.cos(angles_out)
                y_out = np.sin(angles_out)
                angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + theta
                x_in = r * np.cos(angles_in)
                y_in = r * np.sin(angles_in)
                points = np.vstack([
                    np.column_stack((x_out, y_out)),
                    np.column_stack((x_in, y_in))
                ])
                score = _calculate_score(points)
                if score > best_score:
                    best_score = score
                    best_params = (r, theta)
        
        return best_score, best_params

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_overall_score = 0.0
    best_overall_points = None
    
    # Primary candidate: 7-7 configuration
    if time.time() < deadline - 5:
        t_limit_77 = min(80.0, deadline - start_time - 5)
        score, params = _optimize_ring(7, 7, t_limit_77)
        
        if score > best_overall_score:
            best_overall_score = score
            if params is not None:
                r, theta = params
                best_overall_points = _generate_points_77(r, theta)
    
    # Secondary candidates: other ring configurations
    if time.time() < deadline - 5:
        other_candidates = [(8, 6), (6, 8), (9, 5), (5, 9)]
        
        for n_out, n_in in other_candidates:
            if time.time() > deadline - 5:
                break
            
            t_limit = min(10.0, deadline - start_time - 5)
            score, params = _optimize_ring(n_out, n_in, t_limit)
            
            if score > best_overall_score:
                best_overall_score = score
                if params is not None:
                    r, theta = params
                    angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
                    x_out = np.cos(angles_out)
                    y_out = np.sin(angles_out)
                    angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + theta
                    x_in = r * np.cos(angles_in)
                    y_in = r * np.sin(angles_in)
                    best_overall_points = np.vstack([
                        np.column_stack((x_out, y_out)),
                        np.column_stack((x_in, y_in))
                    ])
    
    # Fallback if no valid points found
    if best_overall_points is None:
        # Return 7-7 with default parameters
        r, theta = 0.5, 0.0
        best_overall_points = _generate_points_77(r, theta)
    
    return best_overall_points
