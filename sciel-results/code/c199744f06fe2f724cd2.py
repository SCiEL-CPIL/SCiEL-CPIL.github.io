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

def _score_ring_params(n_out, n_in, params):
    """
    Objective function for differential evolution.
    params: [r, theta]
    """
    r, theta = params
    if r < 0 or r > 0.999:
        return 0.0
        
    # Generate outer points
    angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
    x_out = np.cos(angles_out)
    y_out = np.sin(angles_out)
    
    # Generate inner points
    # Rotate inner points by theta
    angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + theta
    x_in = r * np.cos(angles_in)
    y_in = r * np.sin(angles_in)
    
    points = np.vstack([
        np.column_stack((x_out, y_out)),
        np.column_stack((x_in, y_in))
    ])
    
    return _calculate_score(points)

def _optimize_ring(n_out, n_in, time_limit, start_time=0, deadline=0):
    """
    Optimize a specific ring configuration.
    """
    if start_time == 0:
        start_time = time.time()
    if deadline == 0:
        deadline = start_time + time_limit

    g = np.gcd(n_out, n_in)
    theta_max = 2 * np.pi / g
    
    bounds = [(0.0, 0.999), (0.0, theta_max)]
    
    best_score = 0.0
    best_params = None
    
    # Use Differential Evolution
    # We want to be efficient.
    # If time_limit is large, we can use more iterations.
    
    # Let's try to be a bit more aggressive with popsize if time permits
    # But we need to be careful with the total time budget.
    # For 15s, maybe 20-30 iterations?
    
    popsize = 10
    maxiter = 30
    
    # If we have more time, increase popsize and maxiter
    if time_limit > 30:
        popsize = 15
        maxiter = 50
    elif time_limit > 15:
        popsize = 12
        maxiter = 40

    try:
        res = differential_evolution(
            lambda p: -_score_ring_params(n_out, n_in, p),
            bounds,
            seed=42,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-5,
            polish=True,
            # We don't have a strict time callback for DE in older scipy,
            # but we can rely on the maxiter/popsize limits being conservative enough
            # or we can implement a timeout wrapper if needed.
            # Given the constraint, let's assume 30 iters * 10 pop * eval time < time_limit
        )
        
        if res.success:
            best_score = -res.fun
            best_params = res.x
    except Exception as e:
        pass
    
    # If we have time, do a local polish with multiple starts
    if time.time() < deadline - 5.0:
        starts = []
        # Add some heuristic starts
        # r around 0.5, 0.7, 0.9
        # theta 0, theta_max/2
        for r_start in [0.3, 0.5, 0.7, 0.9, 0.95]:
            for theta_start in [0.0, theta_max/2, theta_max/3, 2*theta_max/3, theta_max/4, 3*theta_max/4]:
                starts.append([r_start, theta_start])
                
        for start in starts:
            if time.time() > deadline - 5.0:
                break
            try:
                res_nm = minimize(
                    lambda p: -_score_ring_params(n_out, n_in, p),
                    start,
                    method='Nelder-Mead',
                    options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 200}
                )
                if -res_nm.fun > best_score:
                    best_score = -res_nm.fun
                    best_params = res_nm.x
            except Exception:
                pass
                
    if best_params is None:
        return 0.0, None
        
    return best_score, best_params

def _score_2p_params(params, n_out, n_in):
    """
    Objective for 2-parameter optimization: r and theta.
    This is the same as _score_ring_params but kept for clarity or potential future use.
    """
    return _score_ring_params(n_out, n_in, params)

def _optimize_2p(n_out, n_in, time_limit, start_time, deadline):
    """
    Optimize the 2-parameter ring configuration (r, theta).
    """
    g = np.gcd(n_out, n_in)
    theta_max = 2 * np.pi / g
    
    bounds = [(0.0, 0.999), (0.0, theta_max)]
    
    best_score = 0.0
    best_params = None
    
    # Use Differential Evolution
    # We want to be efficient.
    
    # Let's try to be a bit more aggressive with popsize if time permits
    popsize = 10
    maxiter = 30
    
    if time_limit > 30:
        popsize = 15
        maxiter = 50
    elif time_limit > 15:
        popsize = 12
        maxiter = 40

    try:
        res = differential_evolution(
            lambda p: -_score_2p_params(p, n_out, n_in),
            bounds,
            seed=42,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-5,
            polish=True
        )
        
        if res.success:
            best_score = -res.fun
            best_params = res.x
    except Exception as e:
        pass
    
    # If we have time, do a local polish with multiple starts
    if time.time() < deadline - 5.0:
        starts = []
        # Add some heuristic starts
        for r_start in [0.3, 0.5, 0.7, 0.9, 0.95]:
            for theta_start in [0.0, theta_max/2, theta_max/3, 2*theta_max/3]:
                starts.append([r_start, theta_start])
                
        for start in starts:
            if time.time() > deadline - 5.0:
                break
            try:
                res_nm = minimize(
                    lambda p: -_score_2p_params(p, n_out, n_in),
                    start,
                    method='Nelder-Mead',
                    options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 200}
                )
                if -res_nm.fun > best_score:
                    best_score = -res_nm.fun
                    best_params = res_nm.x
            except Exception:
                pass
                
    if best_params is None:
        return 0.0, None
        
    return best_score, best_params

def _get_points_2p(n_out, n_in, r, theta):
    """
    Generate points for the 2-parameter ring configuration.
    """
    angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
    x_out = np.cos(angles_out)
    y_out = np.sin(angles_out)
    
    angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + theta
    x_in = r * np.cos(angles_in)
    y_in = r * np.sin(angles_in)
    
    return np.vstack([
        np.column_stack((x_out, y_out)),
        np.column_stack((x_in, y_in))
    ])

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Candidate configurations (n_out, n_in)
    # We assume outer points are on the hull.
    # n_out + n_in = 14
    # n_out >= n_in is not strictly required, but if n_in > n_out, 
    # the inner points might form the hull if r is large enough.
    # However, our model fixes outer points on unit circle and inner on r.
    # If r is close to 1, inner points are close to outer.
    # Let's test all splits.
    
    candidates = [
        (7, 7),
        (8, 6),
        (6, 8),
        (9, 5),
        (5, 9),
        (10, 4),
        (4, 10),
        (11, 3),
        (3, 11),
        (12, 2),
        (2, 12),
        (13, 1),
        (1, 13)
    ]
    
    # Prioritize likely candidates
    # 7-7 is very symmetric.
    # 8-6 and 6-8 are next.
    
    best_overall_score = 0.0
    best_overall_points = None
    
    # Allocate time per candidate
    # We have ~110s.
    # Let's give 7-7 more time, others less.
    
    # Sort candidates by likelihood
    # 7-7 is most likely.
    
    for n_out, n_in in candidates:
        if time.time() > deadline - 10: # Reserve 10s for final assembly
            break
            
        # Time budget for this candidate
        # Decrease time for less likely candidates
        if (n_out, n_in) == (7, 7):
            t_limit = 40.0
        elif (n_out, n_in) in [(8, 6), (6, 8)]:
            t_limit = 20.0
        elif (n_out, n_in) in [(9, 5), (5, 9)]:
            t_limit = 15.0
        else:
            t_limit = 5.0
            
        # Use the 2p optimizer
        score, params = _optimize_2p(n_out, n_in, t_limit, start_time, deadline)
        
        if score > best_overall_score:
            best_overall_score = score
            if params is not None:
                r, theta = params
                # Generate points
                best_overall_points = _get_points_2p(n_out, n_in, r, theta)
                
    # Fallback if no valid points found
    if best_overall_points is None:
        # Return a regular 14-gon? Or just the 7-7 default?
        # Let's just return the 7-7 with r=0.5, theta=0
        n_out, n_in = 7, 7
        r, theta = 0.5, 0.0
        best_overall_points = _get_points_2p(n_out, n_in, r, theta)
        
    return best_overall_points
