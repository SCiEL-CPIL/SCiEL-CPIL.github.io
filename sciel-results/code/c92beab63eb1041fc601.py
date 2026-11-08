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

def _objective_smooth(points_flat, tau=50.0):
    """
    Smooth surrogate for the negative min triangle area.
    We want to maximize the minimum area, so we minimize -min_area.
    f(x) = min(area_i)
    smooth_f(x) = -1/tau * log(sum(exp(-tau * area_i)))
    We minimize -smooth_f(x) = 1/tau * log(sum(exp(-tau * area_i))) - min_area
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
    m = np.min(areas)
    shifted = areas - m
    sum_exp = np.sum(np.exp(-tau * shifted))
    
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

def _generate_initial_candidates():
    """
    Generate diverse initial configurations for optimization.
    """
    candidates = []
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    candidates.append(pts.flatten())
    
    # 2. Multi-ring configurations (Safe integer counts)
    ring_configs = [
        (7, 7, [1.0, 0.6]),
        (7, 7, [1.0, 0.5]),
        (7, 7, [1.0, 0.7]),
        (7, 7, [1.0, 0.4]),
        (5, 5, 4, [1.0, 0.7, 0.4]),
        (4, 5, 5, [1.0, 0.7, 0.4]),
        (3, 5, 6, [1.0, 0.7, 0.4]),
        (2, 6, 6, [1.0, 0.8, 0.5]),
        (1, 6, 7, [1.0, 0.8, 0.5]),
    ]
    
    for config in ring_configs:
        pts_list = []
        
        if len(config) == 3:
            n1, n2, radii = config
            n3 = 0
        else:
            n1, n2, n3, radii = config
            
        # Ensure counts are integers
        n1 = int(n1)
        n2 = int(n2)
        n3 = int(n3)
        
        if n1 + n2 + n3 != N:
            continue
            
        # Ring 1
        if n1 > 0:
            ang1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
            r1 = radii[0]
            pts1 = np.column_stack([r1 * np.cos(ang1), r1 * np.sin(ang1)])
            pts_list.append(pts1)
            
        # Ring 2
        if n2 > 0:
            # Offset to avoid collinearity
            offset = np.pi / n2 if n1 % 2 == 0 else np.pi / max(1, n1)
            ang2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + offset
            r2 = radii[1] if len(radii) > 1 else radii[0]*0.5
            pts2 = np.column_stack([r2 * np.cos(ang2), r2 * np.sin(ang2)])
            pts_list.append(pts2)
            
        # Ring 3
        if n3 > 0:
            offset = np.pi / n3 if n2 % 2 == 0 else np.pi / max(1, n2)
            ang3 = np.linspace(0, 2 * np.pi, n3, endpoint=False) + offset
            r3 = radii[2] if len(radii) > 2 else radii[0]*0.2
            pts3 = np.column_stack([r3 * np.cos(ang3), r3 * np.sin(ang3)])
            pts_list.append(pts3)
            
        if pts_list:
            pts = np.vstack(pts_list)
            if len(pts) == N:
                candidates.append(pts.flatten())
            
    # 3. Random Gaussian
    np.random.seed(42)
    for _ in range(10):
        pts = np.random.randn(N, 2)
        candidates.append(pts.flatten())
        
    # 4. Random points in unit disk
    np.random.seed(123)
    for _ in range(10):
        r = np.sqrt(np.random.rand(N))
        theta = np.random.rand(N) * 2 * np.pi
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        candidates.append(pts.flatten())
        
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s margin before 150s limit
    
    best_score = -1.0
    best_points = None
    
    # Get initial candidates
    inits = _generate_initial_candidates()
    
    # Optimization Strategy:
    # 1. For each candidate, run a local optimization using Powell on the smooth objective.
    # 2. Then polish with Nelder-Mead on the exact objective.
    
    for x0 in inits:
        if time.time() > deadline:
            break
        
        try:
            # Phase 1: Smooth local optimization
            # Powell is good for medium dimensions and doesn't need gradients
            # We use a moderate tau to balance smoothness and fidelity
            res1 = minimize(_objective_smooth, x0, method='Powell', 
                            options={'maxiter': 400, 'xtol': 1e-8, 'ftol': 1e-8, 'adaptive': True})
            
            if time.time() > deadline:
                break
                
            p1 = res1.x.reshape(-1, 2)
            
            # Phase 2: Exact local optimization (Polish)
            # Nelder-Mead is robust for the non-smooth min function
            res2 = minimize(_objective_exact, p1.flatten(), method='Nelder-Mead', 
                            options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            
            if time.time() > deadline:
                break
                
            p2 = res2.x.reshape(-1, 2)
            
            # Normalize final result for consistency
            p2 = p2 - np.mean(p2, axis=0)
            hull_area = _hull_area(p2)
            if hull_area > 1e-10:
                p2 = p2 / np.sqrt(hull_area)
            
            s = _score(p2)
            
            if s > best_score:
                best_score = s
                best_points = p2
                
        except Exception:
            pass
    
    # If we have time, try a broader search using the best point found so far
    # This acts as a form of basin hopping / multi-start around the best solution
    if time.time() < deadline - 10.0 and best_points is not None:
        np.random.seed(999)
        # Number of restarts depends on remaining time
        # Each restart takes ~1-2s for NM polish
        max_restarts = int((deadline - time.time() - 5.0) / 2.0)
        
        for i in range(max_restarts):
            if time.time() > deadline - 5.0:
                break
            
            # Perturbation size can be adaptive, but fixed small step is safe
            noise_scale = 0.1 + 0.1 * np.exp(-i/5.0) # Decay noise
            noise = np.random.randn(N, 2) * noise_scale
            p_perturbed = best_points + noise
            
            try:
                # Use a robust local optimizer
                res_refine = minimize(_objective_exact, p_perturbed.flatten(), method='Nelder-Mead', 
                                      options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
                p_ref = res_refine.x.reshape(-1, 2)
                
                p_ref = p_ref - np.mean(p_ref, axis=0)
                hull_area = _hull_area(p_ref)
                if hull_area > 1e-10:
                    p_ref = p_ref / np.sqrt(hull_area)
                
                s = _score(p_ref)
                if s > best_score:
                    best_score = s
                    best_points = p_ref
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
