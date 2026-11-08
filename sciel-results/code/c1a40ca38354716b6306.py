import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations

N = 14

# Precompute triangle indices for speed
_INDICES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
_I_IDX = _INDICES[:, 0]
_J_IDX = _INDICES[:, 1]
_K_IDX = _INDICES[:, 2]

# Base angles for regular 14-gon
_BASE_ANGLES = np.linspace(0, 2 * np.pi, N, endpoint=False)

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

def _objective_smooth(points_flat, tau=100.0):
    """
    Smooth surrogate for the negative min triangle area.
    We want to maximize the minimum area, so we minimize -min_area.
    Since the hull area must be 1, we normalize inside.
    
    f(x) = min(area_i)
    smooth_f(x) = -1/tau * log(sum(exp(-tau * area_i)))
    
    We minimize -smooth_f(x) = 1/tau * log(sum(exp(-tau * area_i)))
    """
    points = points_flat.reshape(-1, 2)
    
    # 1. Center
    points = points - np.mean(points, axis=0)
    
    # 2. Normalize to unit hull area
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e6  # Penalty for degenerate hull
    
    scale = 1.0 / np.sqrt(hull_area)
    points = points * scale
    
    # 3. Calculate areas
    areas = _get_all_triangle_areas(points)
    
    # 4. Smooth min approximation
    # To avoid overflow/underflow in exp:
    # log(sum(exp(-tau * areas))) = -tau * min_area + log(sum(exp(-tau * (areas - min_area))))
    m = np.min(areas)
    shifted = areas - m
    # shifted >= 0, so exp(-tau * shifted) <= 1, safe to sum
    sum_exp = np.sum(np.exp(-tau * shifted))
    
    # smooth_min = m - (1/tau) * log(sum_exp)
    # We want to minimize -smooth_min = (1/tau) * log(sum_exp) - m
    val = (1.0 / tau) * np.log(sum_exp) - m
    
    return val

def _objective_exact(points_flat):
    """
    Exact objective: -min_triangle_area, assuming points are normalized.
    """
    points = points_flat.reshape(-1, 2)
    
    # 1. Center
    points = points - np.mean(points, axis=0)
    
    # 2. Normalize to unit hull area
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e6
    
    scale = 1.0 / np.sqrt(hull_area)
    points = points * scale
    
    # 3. Calculate min area
    areas = _get_all_triangle_areas(points)
    min_area = np.min(areas)
    
    return -min_area

def _param_to_points(param):
    """
    Convert angular/radial parameters to Cartesian coordinates.
    param: array of shape (2*N,) where first N are radius multipliers, next N are angle offsets
    Returns: array of shape (N, 2)
    """
    radii = 1.0 + param[:N]  # Base radius 1.0, perturbation in param[:N]
    angle_offsets = param[N:]  # Angle perturbations
    
    angles = _BASE_ANGLES + angle_offsets
    points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    # Center
    points = points - np.mean(points, axis=0)
    
    # Normalize to unit hull area
    hull_area = _hull_area(points)
    if hull_area > 1e-10:
        points = points / np.sqrt(hull_area)
    
    return points

def _generate_initial_candidates():
    """
    Generate diverse initial parameter configurations for optimization.
    Returns list of param arrays of shape (2*N,)
    """
    candidates = []
    
    # 1. Regular 14-gon (zero parameters)
    candidates.append(np.zeros(2 * N))
    
    # 2. Random small perturbations
    np.random.seed(42)
    for scale in [0.05, 0.1, 0.2]:
        for _ in range(5):
            param = np.random.randn(2 * N) * scale
            candidates.append(param)
    
    # 3. Structured perturbations - alternate radii
    for r_scale in [0.1, 0.2, 0.3]:
        radii_pert = np.zeros(N)
        radii_pert[::2] = r_scale
        radii_pert[1::2] = -r_scale
        param = np.concatenate([radii_pert, np.zeros(N)])
        candidates.append(param)
    
    # 4. Structured perturbations - angular shifts
    for a_scale in [0.05, 0.1, 0.15]:
        angle_pert = np.zeros(N)
        angle_pert[::2] = a_scale
        angle_pert[1::2] = -a_scale
        param = np.concatenate([np.zeros(N), angle_pert])
        candidates.append(param)
    
    return candidates

def _perturb(x, step=0.1):
    """Perturb a point for basin hopping."""
    return x + np.random.randn(len(x)) * step

def _objective_smooth_param(param, tau=50.0):
    """
    Smooth surrogate objective in parameter space.
    """
    points = _param_to_points(param)
    areas = _get_all_triangle_areas(points)
    
    m = np.min(areas)
    shifted = areas - m
    sum_exp = np.sum(np.exp(-tau * shifted))
    
    val = (1.0 / tau) * np.log(sum_exp) - m
    return val

def _objective_exact_param(param):
    """
    Exact objective in parameter space: -min_triangle_area
    """
    points = _param_to_points(param)
    areas = _get_all_triangle_areas(points)
    return -np.min(areas)

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s margin before 150s limit
    
    best_score = -1.0
    best_points = None
    
    # Get initial candidates in parameter space
    inits = _generate_initial_candidates()
    
    # Phase 1: Local optimization from multiple starting points
    for x0 in inits:
        if time.time() > deadline - 10.0:
            break
        
        try:
            # Smooth local optimization in parameter space
            res1 = minimize(_objective_smooth_param, x0, method='Powell', 
                            options={'maxiter': 200, 'xtol': 1e-8, 'ftol': 1e-8, 'adaptive': True})
            
            if time.time() > deadline - 10.0:
                break
            
            # Polish with exact objective
            res2 = minimize(_objective_exact_param, res1.x, method='Nelder-Mead', 
                            options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            
            if time.time() > deadline - 10.0:
                break
            
            best_param = res2.x
            p2 = _param_to_points(best_param)
            s = _score(p2)
            
            if s > best_score:
                best_score = s
                best_points = p2
                
        except Exception:
            pass
    
    # Phase 2: Global search with differential evolution in parameter space
    if time.time() < deadline - 60.0:
        try:
            # Bounds for parameters: radii in [-0.5, 0.5], angles in [-0.3, 0.3]
            bounds = [(-0.5, 0.5)] * N + [(-0.3, 0.3)] * N
            
            def de_callback(xk, convergence):
                nonlocal best_score, best_points
                p = _param_to_points(xk)
                s = _score(p)
                if s > best_score:
                    best_score = s
                    best_points = p
                return time.time() > deadline - 30.0
            
            res_de = differential_evolution(_objective_exact_param, bounds, 
                                            maxiter=50, tol=1e-8, 
                                            popsize=15, seed=42,
                                            callback=de_callback,
                                            polish=True,
                                            workers=1)
            
            if res_de.x is not None:
                p_de = _param_to_points(res_de.x)
                s_de = _score(p_de)
                if s_de > best_score:
                    best_score = s_de
                    best_points = p_de
                    
        except Exception:
            pass
    
    # Phase 3: Basin hopping around best solution
    if time.time() < deadline - 20.0 and best_points is not None:
        # Convert best points back to parameter space approximately
        # Find closest parameter representation
        best_param = np.zeros(2 * N)
        # Estimate from current best_points
        center = np.mean(best_points, axis=0)
        centered = best_points - center
        hull_area = _hull_area(centered)
        if hull_area > 1e-10:
            normalized = centered / np.sqrt(hull_area)
            # Estimate radii and angles
            radii_est = np.linalg.norm(normalized, axis=1)
            angles_est = np.arctan2(normalized[:, 1], normalized[:, 0])
            # Adjust to match base angles
            angle_diffs = angles_est - _BASE_ANGLES
            # Handle wrap-around
            angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi
            
            best_param = np.concatenate([radii_est - 1.0, angle_diffs])
        
        for _ in range(10):
            if time.time() > deadline - 5.0:
                break
            noise = np.random.randn(2 * N) * 0.1
            p_perturbed = best_param + noise
            try:
                res_refine = minimize(_objective_exact_param, p_perturbed, method='Nelder-Mead', 
                                      options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
                p_ref = _param_to_points(res_refine.x)
                s = _score(p_ref)
                if s > best_score:
                    best_score = s
                    best_points = p_ref
                    best_param = res_refine.x
            except Exception:
                pass

    if best_points is None:
        # Fallback to regular polygon
        best_points = _param_to_points(np.zeros(2 * N))

    return best_points


