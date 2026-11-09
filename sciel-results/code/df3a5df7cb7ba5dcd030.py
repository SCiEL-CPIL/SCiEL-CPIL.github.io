"""
Heilbronn convex regions: 13 points maximizing min triangle area / hull area.
Strategy: Polar coordinate parameterization with Nelder-Mead simplex optimization.
We fix the angular order (sorted by angle) and optimize radii and slight angular jitters.
A penalty is added if the polygon formed by points in angular order is not strictly convex
(all cross products must have the same sign and be non-zero).
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13
DEADLINE = 115.0  # safety margin below 120s limit

def _min_triangle_area(points):
    """Compute the minimum triangle area among all triples of points."""
    n = len(points)
    # For N=13, there are 286 triples. Vectorized calculation is fast.
    idx = np.array(list(combinations(range(n), 3)))
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _hull_area(points):
    """Compute the convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-9

def _score(points):
    """Score = min triangle area / convex hull area."""
    min_ta = _min_triangle_area(points)
    hull_a = _hull_area(points)
    if hull_a <= 1e-9:
        return 0.0
    return min_ta / hull_a

def _is_strictly_convex(points):
    """
    Check if points (assumed sorted by angle around centroid) form a strictly convex polygon.
    We check that all cross products of consecutive edges have the same sign and are sufficiently large.
    """
    n = len(points)
    # Compute centroid
    centroid = np.mean(points, axis=0)
    # Compute angles relative to centroid
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    # Sort indices by angle
    sorted_idx = np.argsort(angles)
    sorted_pts = points[sorted_idx]
    
    # Check cross products of consecutive edges
    # Edge i: sorted_pts[i+1] - sorted_pts[i]
    # Edge i+1: sorted_pts[i+2] - sorted_pts[i+1]
    # Cross product z-component: (x2-x1)*(y3-y2) - (y2-y1)*(x3-x2)
    # Actually simpler: cross of (p_{i+1}-p_i) and (p_{i+2}-p_{i+1})
    # Or just cross of (p_{i+1}-p_i) and (p_{i+2}-p_i) ? No, standard convexity check is cross of consecutive edges.
    
    cross_vals = []
    for i in range(n):
        p0 = sorted_pts[i]
        p1 = sorted_pts[(i+1)%n]
        p2 = sorted_pts[(i+2)%n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        cross_vals.append(cross)
        
    cross_vals = np.array(cross_vals)
    
    # All must be positive (or all negative) and not close to zero
    if np.all(cross_vals > 1e-4):
        return True
    if np.all(cross_vals < -1e-4):
        return True
    return False

def _penalty(points):
    """
    Penalty function: 0 if strictly convex, else large value.
    We use a soft penalty based on how "non-convex" it is.
    """
    if _is_strictly_convex(points):
        return 0.0
    
    # If not convex, compute the magnitude of the "bad" cross products
    centroid = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    sorted_idx = np.argsort(angles)
    sorted_pts = points[sorted_idx]
    
    min_cross = np.inf
    for i in range(len(sorted_pts)):
        p0 = sorted_pts[i]
        p1 = sorted_pts[(i+1)%len(sorted_pts)]
        p2 = sorted_pts[(i+2)%len(sorted_pts)]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        if abs(cross) < min_cross:
            min_cross = abs(cross)
            
    # If min_cross is small, it's nearly degenerate. 
    # We want to push it away from 0. 
    # If signs are mixed, it's non-convex.
    # Simple penalty: 1 / (min_cross + eps) if non-convex? 
    # Let's just return a large constant if not strictly convex to act as a hard barrier for NM.
    return 1000.0

def _objective(params):
    """
    Objective for Nelder-Mead.
    params: 2*N array [r1, theta1, r2, theta2, ...]
    """
    pts = np.zeros((N, 2))
    for i in range(N):
        r = params[2*i]
        theta = params[2*i+1]
        pts[i, 0] = r * np.cos(theta)
        pts[i, 1] = r * np.sin(theta)
        
    # Normalize to unit area? 
    # The score is invariant to scaling. 
    # However, NM works better if the scale is reasonable.
    # Let's normalize the hull area to 1.0 before scoring?
    # No, the score is min_area / hull_area. This is scale invariant.
    # But if points collapse to 0, hull_area is 0.
    # Let's ensure points are reasonably spread out.
    
    if not _is_strictly_convex(pts):
        return 1000.0 - _score(pts) # Penalize non-convexity heavily
    
    return -_score(pts)

def construct():
    start_time = time.time()
    
    # Initial setup
    # We want to start near a regular polygon but allow flexibility.
    # Regular 13-gon angles: 2*pi*i/13
    base_angles = 2 * np.pi * np.arange(N) / N
    base_radii = np.ones(N)
    
    # Flatten to params: [r0, a0, r1, a1, ...]
    x0 = np.zeros(2*N)
    for i in range(N):
        x0[2*i] = base_radii[i]
        x0[2*i+1] = base_angles[i]
        
    # We will run multiple restarts with different initial perturbations
    # to avoid local optima.
    
    best_score = -1.0
    best_points = None
    best_params = x0.copy()
    
    rng = np.random.RandomState(42)
    
    # Number of restarts
    n_restarts = 10
    
    for k in range(n_restarts):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        # Generate perturbed initial parameters
        # Perturb radii and angles slightly
        r_init = base_radii + rng.normal(0, 0.2, N)
        a_init = base_angles + rng.normal(0, 0.1, N)
        
        # Ensure angles are sorted? 
        # The objective function sorts by angle internally for convexity check,
        # but the parameterization is not sorted.
        # This is a bit messy. 
        # Alternative: Fix angles to be sorted, only optimize radii?
        # If we fix angles, we lose the ability to "move" points angularly, 
        # which might be necessary to break symmetry.
        # However, if we optimize both, the mapping is not one-to-one if angles cross.
        # Let's stick to optimizing radii only for a fixed sorted angular grid?
        # No, the inspiration says "radial jitter parameters for each vertex".
        # Let's try optimizing radii only, with angles fixed at regular polygon positions.
        # This is a simpler manifold.
        
        # Let's redefine the objective to take only radii.
        # params: [r0, ..., r12]
        
        # Re-initialize for radii-only optimization
        r_init = np.ones(N) + rng.normal(0, 0.1, N)
        
        def _obj_radii(r):
            if time.time() - start_time > DEADLINE:
                return 1000.0
            pts = np.zeros((N, 2))
            for i in range(N):
                pts[i, 0] = r[i] * np.cos(base_angles[i])
                pts[i, 1] = r[i] * np.sin(base_angles[i])
                
            # Check convexity
            if not _is_strictly_convex(pts):
                return 1000.0 - _score(pts)
                
            return -_score(pts)
        
        try:
            res = minimize(
                _obj_radii,
                r_init,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            r_opt = res.x
            pts_opt = np.zeros((N, 2))
            for i in range(N):
                pts_opt[i, 0] = r_opt[i] * np.cos(base_angles[i])
                pts_opt[i, 1] = r_opt[i] * np.sin(base_angles[i])
                
            # Verify convexity
            if _is_strictly_convex(pts_opt):
                s = _score(pts_opt)
                if s > best_score:
                    best_score = s
                    best_points = pts_opt.copy()
                    best_params = r_opt.copy()
                    
        except Exception:
            pass

    # If radii-only didn't work well, try full polar optimization with sorting fix?
    # Let's try a second phase with full (r, theta) but enforcing sorted theta in the objective.
    # This is complex. Let's try one more strategy: 
    # Start from the best radii solution and do a full coordinate Nelder-Mead.
    
    if best_points is not None and time.time() - start_time < DEADLINE - 5:
        try:
            # Full coordinate optimization
            def _obj_full(pts_flat):
                if time.time() - start_time > DEADLINE:
                    return 1000.0
                pts = pts_flat.reshape(N, 2)
                if not _is_strictly_convex(pts):
                    return 1000.0 - _score(pts)
                return -_score(pts)
                
            res_full = minimize(
                _obj_full,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            pts_full = res_full.x.reshape(N, 2)
            if _is_strictly_convex(pts_full):
                s_full = _score(pts_full)
                if s_full > best_score:
                    best_score = s_full
                    best_points = pts_full.copy()
                    
        except Exception:
            pass

    # Fallback
    if best_points is None:
        # Regular polygon
        best_points = np.zeros((N, 2))
        for i in range(N):
            best_points[i, 0] = np.cos(base_angles[i])
            best_points[i, 1] = np.sin(base_angles[i])
            
    return best_points
