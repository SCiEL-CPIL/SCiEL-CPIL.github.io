import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 14
# Precompute triangle indices
_INDICES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
_I_IDX = _INDICES[:, 0]
_J_IDX = _INDICES[:, 1]
_K_IDX = _INDICES[:, 2]

def _get_all_triangle_areas(points):
    """
    Calculate areas of all C(N,3) triangles formed by the points.
    Returns a numpy array of areas.
    """
    p_i = points[_I_IDX]
    p_j = points[_J_IDX]
    p_k = points[_K_IDX]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    cross_z = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross_z)
    
    return areas

def _hull_area(points):
    """Calculate convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
        
    areas = _get_all_triangle_areas(points)
    min_area = np.min(areas)
    
    return min_area / hull_area

def _is_convex(points):
    """
    Check if points form a strictly convex polygon in the given order.
    Points are assumed to be ordered by angle around the origin.
    """
    n = len(points)
    if n < 3:
        return False
    
    # Compute cross products of consecutive edge vectors
    # Edge i is P[i] - P[i-1]
    # Cross product of Edge i and Edge i+1
    # If all cross products have the same sign (and non-zero), it's convex.
    
    # We need to be careful with the wrap-around.
    # Let's compute vectors v_i = P_{i+1} - P_i
    v = np.roll(points, -1, axis=0) - points
    
    # Cross product z-component of v_i and v_{i+1}
    # v_i x v_{i+1} = v_i[0]*v_{i+1}[1] - v_i[1]*v_{i+1}[0]
    cross_z = v[:, 0] * np.roll(v, -1, axis=0)[:, 1] - v[:, 1] * np.roll(v, -1, axis=0)[:, 0]
    
    # Check if all are positive or all are negative
    if np.all(cross_z > 1e-9):
        return True
    if np.all(cross_z < -1e-9):
        return True
        
    return False

def _points_from_params(params):
    """
    Convert parameters (14 angles, 14 radii) to Cartesian points.
    Angles are sorted to ensure consistent ordering.
    """
    angles = params[:N]
    radii = params[N:]
    
    # Ensure radii are positive to avoid issues with origin
    radii = np.maximum(radii, 1e-6)
    
    # Sort by angle to ensure convex hull order matches index order
    # This is crucial for the convexity check and for the "polygon" to be simple
    sort_idx = np.argsort(angles)
    angles_sorted = angles[sort_idx]
    radii_sorted = radii[sort_idx]
    
    x = radii_sorted * np.cos(angles_sorted)
    y = radii_sorted * np.sin(angles_sorted)
    
    return np.column_stack([x, y])

def _objective_param(params):
    """
    Objective function for parameter optimization.
    Minimizes -score.
    """
    points = _points_from_params(params)
    
    # Check convexity
    if not _is_convex(points):
        return 1e6 # Penalty for non-convexity
    
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e6
        
    areas = _get_all_triangle_areas(points)
    min_area = np.min(areas)
    
    score = min_area / hull_area
    
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s margin before 150s limit
    
    best_score = -1.0
    best_points = None
    
    # Base configuration: Regular 14-gon
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    base_radii = np.ones(N)
    base_params = np.concatenate([base_angles, base_radii])
    
    # Initial candidates in parameter space
    inits = [base_params]
    
    # Add some random perturbations
    np.random.seed(42)
    for _ in range(20):
        noise_ang = np.random.randn(N) * 0.1
        noise_rad = np.random.randn(N) * 0.1
        params = np.concatenate([base_angles + noise_ang, base_radii + noise_rad])
        inits.append(params)
        
    # Also try some "ring" configurations in param space?
    # E.g., alternate radii
    for scale in [0.5, 0.8, 1.2]:
        radii = np.ones(N)
        radii[::2] = scale
        params = np.concatenate([base_angles, radii])
        inits.append(params)
        
    for x0 in inits:
        if time.time() > deadline:
            break
        
        try:
            # Use Nelder-Mead on the parameter space
            # The parameter space is 28D, but structured.
            res = minimize(_objective_param, x0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            
            if time.time() > deadline:
                break
                
            p_params = res.x
            p_points = _points_from_params(p_params)
            
            # Normalize for consistency (though score is invariant to scaling, 
            # it's good to have a canonical form)
            p_points = p_points - np.mean(p_points, axis=0)
            hull_area = _hull_area(p_points)
            if hull_area > 1e-10:
                p_points = p_points / np.sqrt(hull_area)
            
            s = _score(p_points)
            
            if s > best_score:
                best_score = s
                best_points = p_points
                
        except Exception:
            pass
            
    # If we have time, refine the best point found in Cartesian space too?
    # The param space might be stuck. Let's do a quick Cartesian polish on the best param result.
    if time.time() < deadline - 10.0 and best_points is not None:
        # Convert best param result to Cartesian and polish with Cartesian NM
        # Actually, just do a few restarts in param space around the best param?
        # We didn't store best_params. Let's re-derive or store it.
        # Let's just do a few random restarts in param space if time permits.
        
        np.random.seed(999)
        # We need the best_params to restart around it. 
        # We didn't store it. Let's modify the loop to store best_params.
        # For now, just try a few more random inits if time allows.
        
        remaining = deadline - time.time() - 5.0
        if remaining > 5.0:
            max_restarts = int(remaining / 2.0)
            for i in range(max_restarts):
                if time.time() > deadline - 5.0:
                    break
                noise_ang = np.random.randn(N) * 0.2
                noise_rad = np.random.randn(N) * 0.2
                params = np.concatenate([base_angles + noise_ang, base_radii + noise_rad])
                try:
                    res = minimize(_objective_param, params, method='Nelder-Mead', 
                                   options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
                    p_params = res.x
                    p_points = _points_from_params(p_params)
                    
                    p_points = p_points - np.mean(p_points, axis=0)
                    hull_area = _hull_area(p_points)
                    if hull_area > 1e-10:
                        p_points = p_points / np.sqrt(hull_area)
                    
                    s = _score(p_points)
                    if s > best_score:
                        best_score = s
                        best_points = p_points
                except Exception:
                    pass

    if best_points is None:
        # Fallback to regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        best_points = best_points - np.mean(best_points, axis=0)
        hull_area = _hull_area(best_points)
        if hull_area > 1e-10:
             best_points = best_points / np.sqrt(hull_area)

    return best_points
