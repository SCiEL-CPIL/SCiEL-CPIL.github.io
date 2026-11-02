import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

# Precompute indices for all combinations of 3 points
# C(14, 3) = 364
_indices = np.arange(N)
IDX_I, IDX_J, IDX_K = np.meshgrid(_indices, _indices, _indices, indexing='ij')
mask = (IDX_I < IDX_J) & (IDX_J < IDX_K)
IDX_I = IDX_I[mask].flatten()
IDX_J = IDX_J[mask].flatten()
IDX_K = IDX_K[mask].flatten()

def _compute_min_triangle_area(points):
    pi = points[IDX_I]
    pj = points[IDX_J]
    pk = points[IDX_K]
    
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    
    areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
    return np.min(areas)

def _compute_hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _compute_min_triangle_area(points)
    return min_area / hull_area

def _soft_min_objective(angles, temperature=0.01):
    """
    Soft-min objective using log-sum-exp for smoothness.
    angles: array of shape (N,) with sorted angles in [0, 2*pi)
    """
    # Ensure angles are sorted and in [0, 2*pi)
    angles = np.sort(angles)
    # Normalize to [0, 2*pi)
    angles = angles - angles[0]
    if angles[-1] < 2*np.pi - 1e-10:
        # Add the wrap-around gap
        angles = np.append(angles, angles[-1] + (2*np.pi - angles[-1]))
    
    # Place points on unit circle
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Compute all triangle areas
    pi = points[IDX_I]
    pj = points[IDX_J]
    pk = points[IDX_K]
    
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    
    areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
    
    # Soft min using log-sum-exp
    # -soft_min = -(-log(sum(exp(-areas/T))))/T = log(sum(exp(-areas/T)))/T
    shifted = -areas / temperature
    shifted -= np.max(shifted)  # For numerical stability
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_val = -(log_sum + np.max(shifted)) * temperature
    
    # The hull area for points on unit circle is fixed
    # But we need to normalize by actual hull area
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    
    return -soft_min_val / hull_area

def _angle_to_points(angles):
    """Convert sorted angles to points on unit circle."""
    angles = np.sort(angles)
    return np.column_stack((np.cos(angles), np.sin(angles)))

def _points_to_angles(points):
    """Convert points to their angles (for convex position)."""
    # Only valid if points are in convex position
    cx = np.mean(points[:, 0])
    cy = np.mean(points[:, 1])
    angles = np.arctan2(points[:, 1] - cy, points[:, 0] - cx)
    return np.sort(angles)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_points = None
    best_score = 0.0
    
    # Strategy 1: Optimize in angle space (convex position parameterization)
    # Start with regular 14-gon angles
    regular_angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    
    # Objective for angle optimization
    def angle_objective(angles_flat):
        angles = angles_flat.copy()
        # Ensure angles are in [0, 2*pi)
        angles = np.mod(angles, 2*np.pi)
        angles = np.sort(angles)
        return _soft_min_objective(angles, temperature=0.01)
    
    # Try multiple initial angle configurations
    angle_inits = [
        regular_angles.copy(),
        regular_angles + 0.1 * np.sin(np.arange(N) * 0.5),  # Perturbed regular
        regular_angles + 0.05 * np.cos(np.arange(N) * 1.0),  # Another perturbation
    ]
    
    for i, ang_init in enumerate(angle_inits):
        if time.time() > deadline - 15.0:
            break
        
        # Nelder-Mead on angles
        try:
            res = minimize(angle_objective, ang_init, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            if res.fun < angle_objective(ang_init):
                cand_angles = res.x
                cand_angles = np.mod(cand_angles, 2*np.pi)
                cand_angles = np.sort(cand_angles)
                cand_points = _angle_to_points(cand_angles)
                cand_score = _score(cand_points)
                if cand_score > best_score:
                    best_points = cand_points
                    best_score = cand_score
        except Exception:
            pass
        
        # Also try Powell on angles
        if time.time() > deadline - 15.0:
            break
        try:
            res2 = minimize(angle_objective, ang_init, method='Powell',
                           options={'maxiter': 300, 'xtol': 1e-6, 'ftol': 1e-6})
            if res2.fun < angle_objective(ang_init):
                cand_angles = res2.x
                cand_angles = np.mod(cand_angles, 2*np.pi)
                cand_angles = np.sort(cand_angles)
                cand_points = _angle_to_points(cand_angles)
                cand_score = _score(cand_points)
                if cand_score > best_score:
                    best_points = cand_points
                    best_score = cand_score
        except Exception:
            pass
    
    # Strategy 2: If we have a good convex solution, refine in Cartesian space
    if best_points is not None:
        if time.time() > deadline - 10.0:
            return best_points
        
        def cart_objective(flat_points):
            pts = flat_points.reshape(N, 2)
            return -_score(pts)
        
        try:
            res3 = minimize(cart_objective, best_points.flatten(), method='Nelder-Mead',
                           options={'maxiter': 1500, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True})
            if res3.fun < cart_objective(best_points.flatten()):
                cand_pts = res3.x.reshape(N, 2)
                cand_score = -res3.fun
                if cand_score > best_score:
                    best_points = cand_pts
                    best_score = cand_score
        except Exception:
            pass
        
        # Final polish with tighter tolerance
        if time.time() > deadline - 5.0:
            return best_points
        try:
            res4 = minimize(cart_objective, best_points.flatten(), method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
            if res4.fun < cart_objective(best_points.flatten()):
                cand_pts = res4.x.reshape(N, 2)
                cand_score = -res4.fun
                if cand_score > best_score:
                    best_points = cand_pts
                    best_score = cand_score
        except Exception:
            pass
    
    # Fallback: if angle-based optimization didn't work well, try original Cartesian approach
    if best_points is None or best_score < 0.001:
        # Regular 14-gon as fallback
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        best_score = _score(best_points)
        
        if time.time() > deadline - 10.0:
            return best_points
        
        def cart_objective(flat_points):
            pts = flat_points.reshape(N, 2)
            return -_score(pts)
        
        try:
            res5 = minimize(cart_objective, best_points.flatten(), method='Powell',
                           options={'maxiter': 500, 'xtol': 1e-5, 'ftol': 1e-5})
            if res5.fun < cart_objective(best_points.flatten()):
                cand_pts = res5.x.reshape(N, 2)
                cand_score = -res5.fun
                if cand_score > best_score:
                    best_points = cand_pts
                    best_score = cand_score
        except Exception:
            pass
    
    return best_points
