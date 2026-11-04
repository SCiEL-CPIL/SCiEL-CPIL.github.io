import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
import time
import itertools

N = 14

# Precompute triple indices for N=14
TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)
])

# Random state for reproducibility
_RNG = np.random.RandomState(42)

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

def _soft_min_score(points, beta=100.0):
    """
    Smooth surrogate for min triangle area / hull area using log-sum-exp.
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
    
    # Soft min: -1/beta * log(sum(exp(-beta * x_i)))
    # Stabilized: m - 1/beta * log(sum(exp(-beta*(x_i - m)))) where m = max(x)
    m = np.max(areas)
    exp_terms = np.exp(-beta * (areas - m))
    sum_exp = np.sum(exp_terms)
    soft_min_area = m - np.log(sum_exp) / beta
    
    return soft_min_area / hull_area

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

def _soft_score_ring_params(n_out, n_in, params, beta=100.0):
    """
    Soft objective function for gradient-based optimization.
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
    
    return _soft_min_score(points, beta)

def _optimize_ring(n_out, n_in, time_limit):
    """
    Optimize a specific ring configuration.
    """
    start_time = time.time()
    
    # Symmetry of the system
    # The system is invariant under rotation by 2*pi/gcd(n_out, n_in)
    # So we only need to search theta in [0, 2*pi/gcd(n_out, n_in)]
    g = np.gcd(n_out, n_in)
    theta_max = 2 * np.pi / g
    
    bounds = [(0.0, 0.999), (0.0, theta_max)]
    
    best_score = 0.0
    best_params = None
    
    # Phase 1: Global search with Differential Evolution
    try:
        popsize = 12
        maxiter = 40
        
        res = differential_evolution(
            lambda p: -_score_ring_params(n_out, n_in, p),
            bounds,
            seed=42,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-6,
            polish=False
        )
        
        if res.success:
            best_score = -res.fun
            best_params = res.x
    except Exception:
        pass
    
    # Phase 2: Two-stage soft-min L-BFGS-B polish
    # Stage A: low beta for broad exploration
    # Stage B: high beta for sharp refinement
    if time.time() - start_time < time_limit * 0.50:
        starts = []
        
        if best_params is not None:
            starts.append(best_params.copy())
            
        # Diverse heuristic starts
        for r_start in [0.3, 0.5, 0.7, 0.9]:
            for theta_start in [0.0, theta_max/3, 2*theta_max/3]:
                starts.append([r_start, theta_start])
                
        for start in starts:
            if time.time() - start_time > time_limit * 0.70:
                break
            try:
                # Stage A: low beta (broad)
                res_a = minimize(
                    lambda p: -_soft_score_ring_params(n_out, n_in, p, 30.0),
                    start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                exact_score_a = _score_ring_params(n_out, n_in, res_a.x)
                if exact_score_a > best_score:
                    best_score = exact_score_a
                    best_params = res_a.x.copy()
                
                # Stage B: high beta (sharp), starting from Stage A result
                if time.time() - start_time > time_limit * 0.70:
                    break
                res_b = minimize(
                    lambda p: -_soft_score_ring_params(n_out, n_in, p, 250.0),
                    res_a.x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                exact_score_b = _score_ring_params(n_out, n_in, res_b.x)
                if exact_score_b > best_score:
                    best_score = exact_score_b
                    best_params = res_b.x.copy()
                    
            except Exception:
                pass
                
    # Phase 3: High-precision Nelder-Mead on exact objective
    if time.time() - start_time < time_limit * 0.95 and best_params is not None:
        try:
            res_nm = minimize(
                lambda p: -_score_ring_params(n_out, n_in, p),
                best_params,
                method='Nelder-Mead',
                options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 1500, 'adaptive': True}
            )
            if -res_nm.fun > best_score:
                best_score = -res_nm.fun
                best_params = res_nm.x
        except Exception:
            pass

    if best_params is None:
        return 0.0, None
        
    return best_score, best_params

def _polish_points(points, time_limit, start_time):
    """
    Directly polish the point coordinates using Nelder-Mead on the exact objective.
    Multi-start with jitter to escape local optima, then high-precision refinement.
    """
    deadline = start_time + time_limit
    best_score = _calculate_score(points)
    best_points = points.copy()
    
    def neg_score(x):
        pts = x.reshape(-1, 2)
        return -_calculate_score(pts)
    
    # Multi-start Nelder-Mead with jittered initializations
    # Generate several perturbed starting points to break symmetry
    n_starts = 5
    start_spacing = (time_limit * 0.6) / n_starts
    
    for s in range(n_starts):
        if time.time() > deadline - 5:
            break
        try:
            if s == 0:
                x0 = points.flatten().copy()
            else:
                # Jitter: small random perturbation scaled by point spread
                spread = np.std(points, axis=0)
                jitter_scale = 0.02 * (1.0 + 0.5 * s)
                x0 = points.flatten().copy() + _RNG.randn(N, 2) * jitter_scale * spread
                x0 = x0.reshape(-1, 2).flatten()
            
            sub_deadline = min(deadline - 5, start_time + time_limit * (0.15 + 0.45 * (s + 1) / n_starts))
            max_iter = max(500, int((sub_deadline - time.time()) * 50))
            max_iter = min(max_iter, 3000)
            
            res = minimize(
                neg_score,
                x0,
                method='Nelder-Mead',
                options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': max_iter, 'adaptive': True}
            )
            
            if time.time() < deadline:
                candidate_score = -res.fun
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_points = res.x.reshape(-1, 2).copy()
        except Exception:
            pass
    
    # Final high-precision polish on the best point found
    if time.time() < deadline - 3:
        try:
            res2 = minimize(
                neg_score,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'xatol': 1e-14, 'fatol': 1e-14, 'maxiter': 2000, 'adaptive': True}
            )
            
            candidate_score = -res2.fun
            if candidate_score > best_score:
                best_score = candidate_score
                best_points = res2.x.reshape(-1, 2).copy()
        except Exception:
            pass
    
    return best_score, best_points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin (150s limit)
    
    # Candidate configurations (n_out, n_in)
    # We assume outer points are on the hull.
    # n_out + n_in = 14
    
    # Prioritize configurations most likely to yield high scores
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
    
    for n_out, n_in in candidates:
        if time.time() > deadline - 35:  # Reserve 35s for final polish
            break
            
        # Time budget for this candidate
        if (n_out, n_in) == (7, 7):
            t_limit = 55.0
        elif (n_out, n_in) in [(8, 6), (6, 8)]:
            t_limit = 18.0
        elif (n_out, n_in) in [(9, 5), (5, 9)]:
            t_limit = 10.0
        else:
            t_limit = 4.0
            
        score, params = _optimize_ring(n_out, n_in, t_limit)
        
        if score > best_overall_score and params is not None:
            best_overall_score = score
            r, theta = params
            # Generate points
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
        n_out, n_in = 7, 7
        r, theta = 0.5, 0.0
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
    
    # Final direct polish on all 28 coordinates with multi-start
    if time.time() < deadline - 3:
        polish_time = deadline - time.time() - 3
        polished_score, polished_points = _polish_points(
            best_overall_points, polish_time, start_time
        )
        if polished_score > best_overall_score:
            best_overall_score = polished_score
            best_overall_points = polished_points
    
    return best_overall_points
