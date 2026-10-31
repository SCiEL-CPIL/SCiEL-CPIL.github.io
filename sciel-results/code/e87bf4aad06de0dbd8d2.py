import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def _get_triangle_areas(points):
    """
    Vectorized calculation of areas for all triples.
    points: (N, 2) array
    returns: array of shape (num_triangles,)
    """
    n = len(points)
    indices = list(combinations(range(n), 3))
    if not indices:
        return np.array([])
    
    i_idx = np.array([p[0] for p in indices])
    j_idx = np.array([p[1] for p in indices])
    k_idx = np.array([p[2] for p in indices])
    
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return areas

def _hull_area(points):
    """Calculate convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Calculate the raw score: min_triangle_area / hull_area."""
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    areas = _get_triangle_areas(points)
    if len(areas) == 0:
        return 0.0
    min_area = np.min(areas)
    return min_area / hull_area

def _params_to_points(params):
    """
    Convert (radial_perturbations, angular_shifts) to (N, 2) points.
    params: shape (2*N,) where first N are radial multipliers (1 + delta_r)
             and last N are angular shifts (delta_theta)
    Returns: (N, 2) array of points
    """
    delta_r = params[:N]
    delta_theta = params[N:]
    
    base_angles = 2 * np.pi * np.arange(N) / N
    angles = base_angles + delta_theta
    radii = 1.0 + delta_r
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    
    return np.column_stack([x, y])

def _points_to_params(points):
    """
    Convert (N, 2) points to (radial_perturbations, angular_shifts).
    Assumes points are near a unit circle.
    Returns: shape (2*N,) array
    """
    r = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    theta = np.arctan2(points[:, 1], points[:, 0])
    
    base_angles = 2 * np.pi * np.arange(N) / N
    delta_r = r - 1.0
    delta_theta = theta - base_angles
    # Wrap angular shifts to [-pi, pi]
    delta_theta = (delta_theta + np.pi) % (2 * np.pi) - np.pi
    
    return np.concatenate([delta_r, delta_theta])

def _softmin_score_from_params(params, alpha):
    """
    Smooth score computed from parameterized representation.
    """
    points = _params_to_points(params)
    
    # Check convexity: all points should be in convex position
    # For a convex configuration, the convex hull should contain all N points
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        # If not all points are on the hull, it's not a valid convex configuration
        if len(hull.vertices) < N:
            return 0.0
    except Exception:
        return 0.0
    
    if hull_area < 1e-10:
        return 0.0
    
    areas = _get_triangle_areas(points)
    if len(areas) == 0:
        return 0.0
    
    x = -alpha * areas
    x_max = np.max(x)
    softmin_val = -(x_max + np.log(np.sum(np.exp(x - x_max)))) / alpha
    
    return softmin_val / hull_area

def _neg_softmin_from_params(params, alpha):
    """Objective for minimization in parameter space."""
    return -_softmin_score_from_params(params, alpha)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_score = 0.0
    best_points = None
    
    # Strategy: Use parameterized representation (radial + angular perturbations)
    # This reduces the effective search space and leverages near-circular symmetry
    
    # Phase 1: Differential Evolution for global search in parameter space
    # Parameters: N radial perturbations in [-0.5, 0.5], N angular shifts in [-pi/13, pi/13]
    bounds = [(-0.5, 0.5)] * N + [(-np.pi / N, np.pi / N)] * N
    
    if time.time() < deadline - 60:
        try:
            de_result = differential_evolution(
                _neg_softmin_from_params,
                bounds,
                args=(50,),
                maxiter=50,
                popsize=15,
                tol=1e-4,
                seed=42,
                polish=False,
                init='latinhypercube'
            )
            best_params = de_result.x
            cand_pts = _params_to_points(best_params)
            score = _score(cand_pts)
            if score > best_score:
                best_score = score
                best_points = cand_pts
        except Exception:
            pass
    
    # Phase 2: Local refinement with Powell in parameter space
    if time.time() < deadline - 20:
        # Start from the DE result or regular polygon
        if best_points is not None:
            start_params = _points_to_params(best_points)
        else:
            start_params = np.zeros(2 * N)
        
        # Try multiple alphas for refinement
        for alpha in [50, 100, 200, 500]:
            if time.time() > deadline - 5:
                break
            try:
                res = minimize(
                    _neg_softmin_from_params,
                    start_params,
                    args=(alpha,),
                    method='Powell',
                    options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10, 'disp': False}
                )
                cand_pts = _params_to_points(res.x)
                score = _score(cand_pts)
                if score > best_score:
                    best_score = score
                    best_points = cand_pts
                    start_params = res.x
            except Exception:
                continue
    
    # Phase 3: Also try direct coordinate optimization from best found
    if best_points is not None and time.time() < deadline - 10:
        def _neg_score_direct(params):
            points = params.reshape(N, 2)
            return -_score(points)
        
        try:
            res = minimize(
                _neg_score_direct,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            cand_pts = res.x.reshape(N, 2)
            score = _score(cand_pts)
            if score > best_score:
                best_score = score
                best_points = cand_pts
        except Exception:
            pass
    
    # Phase 4: Final refinement with high alpha in parameter space
    if best_points is not None and time.time() < deadline - 3:
        final_params = _points_to_params(best_points)
        try:
            res = minimize(
                _neg_softmin_from_params,
                final_params,
                args=(1000,),
                method='Powell',
                options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12, 'disp': False}
            )
            cand_pts = _params_to_points(res.x)
            score = _score(cand_pts)
            if score > best_score:
                best_score = score
                best_points = cand_pts
        except Exception:
            pass
    
    if best_points is None:
        # Fallback to regular polygon
        angles = 2 * np.pi * np.arange(N) / N
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points
