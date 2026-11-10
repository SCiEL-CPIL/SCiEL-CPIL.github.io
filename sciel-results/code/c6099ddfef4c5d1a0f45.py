# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import itertools
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 590.0  # 10s safety margin

def _all_triangle_areas(points):
    """
    Compute all 286 triangle areas for 13 points. Vectorized.
    Returns array of shape (286,).
    """
    n = len(points)
    indices = np.array(list(itertools.combinations(range(n), 3)))
    p1 = points[indices[:, 0]]
    p2 = points[indices[:, 1]]
    p3 = points[indices[:, 2]]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _min_triangle_area(points):
    """
    Compute the minimum area among all triangles formed by triples of points.
    """
    areas = _all_triangle_areas(points)
    return np.min(areas)

def _hull_area(points):
    """
    Compute the area of the convex hull.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Calculate the metric: A_min / A_hull.
    Returns 0 if invalid.
    """
    if not np.all(np.isfinite(points)):
        return 0.0
    a_min = _min_triangle_area(points)
    a_hull = _hull_area(points)
    if a_hull <= 1e-12 or a_min <= 1e-12:
        return 0.0
    return a_min / a_hull

def _soft_min_objective(points_flat, temperature=0.05):
    """
    Smooth surrogate for -A_min/A_hull using log-sum-exp soft min.
    This makes the objective differentiable for gradient-based optimization.
    """
    n = 13
    points = points_flat.reshape((n, 2))
    
    if not np.all(np.isfinite(points)):
        return 1e10
    
    areas = _all_triangle_areas(points)
    
    # Soft minimum: -T * log(sum(exp(-areas/T)))
    # This approximates min(areas) as T -> 0
    min_val = np.min(areas)
    if min_val < 1e-15:
        return 1e10
    
    # Numerically stable log-sum-exp
    shifted = areas / temperature
    max_shifted = np.max(shifted)
    log_sum_exp = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    soft_min = -temperature * log_sum_exp
    
    if soft_min < 1e-15:
        return 1e10
    
    a_hull = _hull_area(points)
    if a_hull <= 1e-12:
        return 1e10
    
    return -soft_min / a_hull

def _negative_ratio(points_flat):
    """
    Objective: negative of A_min/A_hull (exact, non-smooth).
    Used for final evaluation and coordinate descent.
    """
    n = 13
    points = points_flat.reshape((n, 2))
    s = _score(points)
    if s == 0:
        return 1e10
    return -s

def _coordinate_descent_polish(points, max_iters=200, tol=1e-8):
    """
    Coordinate descent: for each point, optimize its (x,y) position
    to maximize the minimum triangle area, holding others fixed.
    """
    n = len(points)
    points = points.copy()
    
    for iteration in range(max_iters):
        if time.time() > _DEADLINE - 5:
            break
        
        improved = False
        for i in range(n):
            if time.time() > _DEADLINE - 5:
                break
            
            # Get current score
            current_score = _score(points)
            
            # Try small perturbations in x and y
            best_pos = points[i].copy()
            best_local_score = current_score
            
            # Adaptive step size
            step = 0.05
            for _ in range(20):
                for dx, dy in [(step, 0), (-step, 0), (0, step), (0, -step),
                                (step*0.707, step*0.707), (-step*0.707, -step*0.707),
                                (step*0.707, -step*0.707), (-step*0.707, step*0.707)]:
                    new_point = points[i] + np.array([dx, dy])
                    points[i] = new_point
                    s = _score(points)
                    if s > best_local_score + 1e-10:
                        best_local_score = s
                        best_pos = new_point.copy()
                        improved = True
                    points[i] = best_pos  # restore best found so far
                
                if not improved:
                    step *= 0.5
                else:
                    improved = False
        
        if not improved:
            break
    
    return points

def _regular_polygon(n, radius=1.0):
    """Generate regular n-gon centered at origin."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    best_points = None
    best_score = 0.0
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Regular polygon
    reg_poly = _regular_polygon(n, radius=1.0)
    initializations.append(reg_poly.flatten())
    
    # 2. Regular polygon with center point (12 on circle + 1 center)
    rng = np.random.default_rng(seed=42)
    angles_12 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    poly12 = np.column_stack([np.cos(angles_12), np.sin(angles_12)])
    center = np.array([[0.0, 0.0]])
    mixed = np.vstack([poly12, center])
    initializations.append(mixed.flatten())
    
    # 3. Random points in unit circle
    for i in range(8):
        theta = rng.uniform(0, 2 * np.pi, n)
        r = np.sqrt(rng.uniform(0, 1, n))
        rand_pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        initializations.append(rand_pts.flatten())
    
    # 4. Perturbed regular polygon
    for i in range(5):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rng.random(n) * 0.15
        radii = 1.0 + rng.random(n) * 0.15
        circ_pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        initializations.append(circ_pts.flatten())
    
    # 5. Two concentric rings (7 outer, 6 inner)
    for i in range(3):
        outer_angles = np.linspace(0, 2 * np.pi, 7, endpoint=False) + rng.random(7) * 0.2
        inner_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False) + rng.random(6) * 0.2
        outer_r = 1.0 + rng.random(7) * 0.1
        inner_r = 0.5 + rng.random(6) * 0.1
        outer_pts = np.column_stack([outer_r * np.cos(outer_angles), outer_r * np.sin(outer_angles)])
        inner_pts = np.column_stack([inner_r * np.cos(inner_angles), inner_r * np.sin(inner_angles)])
        two_ring = np.vstack([outer_pts, inner_pts])
        initializations.append(two_ring.flatten())
    
    # Evaluate all initializations
    init_scores = []
    for init in initializations:
        pts = init.reshape((n, 2))
        s = _score(pts)
        init_scores.append((s, init))
    
    init_scores.sort(reverse=True)
    
    # Track best from initializations
    for s, init in init_scores:
        if s > best_score:
            best_score = s
            best_points = init.reshape((n, 2)).copy()
    
    # Phase 1: L-BFGS-B optimization with soft-min surrogate from top initializations
    top_k = min(6, len(init_scores))
    for s, init in init_scores[:top_k]:
        if time.time() > _DEADLINE - 30:
            break
        
        try:
            # Start with moderate temperature, anneal
            for temp in [0.1, 0.05, 0.02, 0.01]:
                if time.time() > _DEADLINE - 10:
                    break
                
                res = minimize(
                    _soft_min_objective,
                    init,
                    method='L-BFGS-B',
                    args=(temp,),
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                opt_pts = res.x.reshape((n, 2))
                opt_score = _score(opt_pts)
                
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_pts.copy()
                
                # Use optimized result as next starting point
                init = res.x.copy()
                
        except Exception:
            continue
    
    # Phase 2: Coordinate descent polishing from best point
    if best_points is not None and time.time() < _DEADLINE - 10:
        polished = _coordinate_descent_polish(best_points, max_iters=100)
        polished_score = _score(polished)
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished
    
    # Phase 3: Try L-BFGS-B from the polished result with exact objective
    if best_points is not None and time.time() < _DEADLINE - 5:
        try:
            res = minimize(
                _negative_ratio,
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            opt_pts = res.x.reshape((n, 2))
            opt_score = _score(opt_pts)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_pts.copy()
        except Exception:
            pass
    
    # Final coordinate descent polish
    if best_points is not None and time.time() < _DEADLINE - 2:
        final_polished = _coordinate_descent_polish(best_points, max_iters=50)
        final_score = _score(final_polished)
        if final_score > best_score:
            best_points = final_polished
    
    # Fallback
    if best_points is None:
        best_points = _regular_polygon(n, radius=1.0)
    
    return best_points

# EVOLVE-BLOCK-END
