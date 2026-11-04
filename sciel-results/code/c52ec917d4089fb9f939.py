import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
import time

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

def _soft_min_score(points, beta):
    """
    Smooth surrogate: -soft_min of triangle areas / hull area.
    Returns a smooth value that approximates -min_area/hull_area.
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
    
    # Soft-min via log-sum-exp: soft_min(a) ≈ (1/beta) * log(sum(exp(-beta * a)))
    # This is smooth and approximates min(a) as beta -> inf
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    soft_min_val = (1.0 / beta) * (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled))))
    
    return soft_min_val / hull_area

def _get_ring_points(n_out, n_in, r, theta):
    """
    Generate points for the two-ring configuration.
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

def _score_ring_params(n_out, n_in, params):
    """
    Objective function for ring parameters.
    params: [r, theta]
    """
    r, theta = params
    if r < 0 or r > 0.999:
        return 0.0
        
    points = _get_ring_points(n_out, n_in, r, theta)
    return _calculate_score(points)

def _soft_ring_score(n_out, n_in, params, beta):
    """
    Soft objective for ring parameters.
    """
    r, theta = params
    if r < 0 or r > 0.999:
        return 0.0
        
    points = _get_ring_points(n_out, n_in, r, theta)
    return _soft_min_score(points, beta)

def _optimize_ring(n_out, n_in, time_limit, deadline):
    """
    Optimize a specific ring configuration.
    """
    g = np.gcd(n_out, n_in)
    theta_max = 2 * np.pi / g
    
    bounds = [(0.0, 0.999), (0.0, theta_max)]
    
    best_score = 0.0
    best_params = None
    best_points = None
    
    # Phase 1: Differential Evolution on soft objective to find a good region
    # Use a moderate beta for smoothness
    beta_de = 50.0
    
    popsize = 15
    maxiter = 20
    
    try:
        res = differential_evolution(
            lambda p: _soft_ring_score(n_out, n_in, p, beta_de),
            bounds,
            seed=42,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-5,
            polish=False # We will do our own polish
        )
        
        if res.success:
            best_params = res.x
            best_points = _get_ring_points(n_out, n_in, best_params[0], best_params[1])
            best_score = _calculate_score(best_points)
    except Exception:
        pass
    
    # Phase 2: Local polish on 2D parameters (r, theta) using exact score
    # This is fast and can refine the global search result
    if best_params is not None and time.time() < deadline - 10.0:
        try:
            res_nm = minimize(
                lambda p: -_score_ring_params(n_out, n_in, p),
                best_params,
                method='Nelder-Mead',
                options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 200}
            )
            if -res_nm.fun > best_score:
                best_score = -res_nm.fun
                best_params = res_nm.x
                best_points = _get_ring_points(n_out, n_in, best_params[0], best_params[1])
        except Exception:
            pass
            
    # Phase 3: Coordinate-level polish
    # This allows the points to break the ring structure if it improves the score
    if best_points is not None and time.time() < deadline - 5.0:
        best_points, best_score = _coordinate_polish(best_points, deadline)
        
    if best_points is None:
        return 0.0, None
        
    return best_score, best_points

def _coordinate_polish(points, deadline):
    """
    Polish all 28 coordinates directly using L-BFGS-B with annealed soft-min,
    followed by a final Nelder-Mead pass on the exact objective.
    """
    if time.time() > deadline - 3.0:
        return points, _calculate_score(points)
    
    best_points = points.copy()
    best_score = _calculate_score(points)
    
    # Annealed beta sequence: start moderate, increase for sharper min
    # These values are tuned to balance smoothness and approximation accuracy
    beta_sequence = [100.0, 300.0, 1000.0, 3000.0]
    
    for beta in beta_sequence:
        if time.time() > deadline - 5.0:
            break
        
        def neg_soft_score(x, b=beta):
            pts = x.reshape(-1, 2)
            return _soft_min_score(pts, b)
        
        try:
            # Increase maxiter for lower beta (smoother), decrease for higher beta (sharper)
            maxiter = 300 if beta <= 300.0 else 150
            res = minimize(
                neg_soft_score,
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            polished = res.x.reshape(-1, 2)
            polished_score = _calculate_score(polished)
            
            if polished_score > best_score:
                best_points = polished
                best_score = polished_score
                
        except Exception:
            pass
            
    # Final polish with Nelder-Mead on exact score to handle non-smoothness
    # This is crucial for squeezing out the last bits of performance
    if time.time() < deadline - 3.0:
        try:
            def neg_exact_score(x):
                pts = x.reshape(-1, 2)
                return -_calculate_score(pts)
            
            res_nm = minimize(
                neg_exact_score,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 500}
            )
            
            if res_nm.success:
                polished = res_nm.x.reshape(-1, 2)
                polished_score = _calculate_score(polished)
                if polished_score > best_score:
                    best_points = polished
                    best_score = polished_score
        except Exception:
            pass

    return best_points, best_score

def construct():
    start_time = time.time()
    # Evaluator kills after 150s. Use 140s deadline to be safe.
    deadline = start_time + 140.0 

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
    
    best_overall_score = 0.0
    best_overall_points = None
    
    # Allocate time to candidates
    # 7-7 is the most symmetric and likely to yield high scores.
    # We'll give it the most time.
    
    # Time allocation strategy:
    # 7-7: 40s
    # 8-6, 6-8: 20s each
    # Others: 5s each
    # Total: 40 + 40 + 5*9 = 125s. Plus overhead. Fits in 140s.
    
    time_allocations = {
        (7, 7): 40.0,
        (8, 6): 20.0,
        (6, 8): 20.0,
        (9, 5): 5.0,
        (5, 9): 5.0,
        (10, 4): 5.0,
        (4, 10): 5.0,
        (11, 3): 5.0,
        (3, 11): 5.0,
        (12, 2): 5.0,
        (2, 12): 5.0,
        (13, 1): 5.0,
        (1, 13): 5.0
    }
    
    for n_out, n_in in candidates:
        if time.time() > deadline - 10.0:
            break
            
        t_limit = time_allocations.get((n_out, n_in), 5.0)
        
        # Ensure we don't exceed the global deadline
        if time.time() + t_limit > deadline - 5.0:
            t_limit = deadline - 5.0 - time.time()
            
        if t_limit < 2.0:
            continue
            
        score, points = _optimize_ring(n_out, n_in, t_limit, deadline)
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_points = points
    
    # Fallback if no valid points found or all failed
    if best_overall_points is None:
        # Return a regular 14-gon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_overall_points = np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_overall_points
