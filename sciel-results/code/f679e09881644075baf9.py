from scipy.optimize import minimize
import math
import time
import numpy as np
from scipy.optimize import minimize_scalar

N = 11
SQRT3 = math.sqrt(3)

def _hex_vertices(cx, cy, angle_deg, s=1.0):
    """Vertices of a regular hexagon with side s, centered at (cx,cy), rotated angle_deg."""
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([s * np.cos(angles), s * np.sin(angles)])
    return vs + np.array([cx, cy])

def _outer_hex_normals(angle_deg):
    """
    Return the 6 outward unit normals of a regular hexagon with the given rotation.
    A regular hexagon with side s has apothem a = s*sqrt(3)/2.
    The vertices are at angles angle_deg + k*60.
    The edge normals (perpendicular to edges, pointing outward) are at angles angle_deg + 30 + k*60.
    """
    base = np.deg2rad(angle_deg + 30)
    normals = []
    for k in range(6):
        theta = base + k * np.pi / 3
        normals.append(np.array([np.cos(theta), np.sin(theta)]))
    return np.array(normals)

def _min_enclosing_side(centers, inner_angles, outer_angle_deg):
    """
    Compute the minimum side length of the outer hexagon (centered at origin) 
    such that all inner hexagons are contained.
    Uses the property that for a regular hexagon, containment is equivalent to
    each vertex's projection onto each of the 6 edge normals being <= apothem.
    Apothem = side * sqrt(3) / 2.
    So side = 2 * max_projection / sqrt(3).
    """
    normals = _outer_hex_normals(outer_angle_deg)
    max_proj = 0.0
    for i in range(len(centers)):
        cx, cy = centers[i]
        a = inner_angles[i]
        verts = _hex_vertices(cx, cy, a)
        # Project all vertices onto all normals
        dots = verts @ normals.T
        # For containment in a centered hexagon, we need |projection| <= apothem
        # Actually, the hexagon is defined by -apothem <= n_i . v <= apothem for all vertices v
        # So we need max over all vertices and normals of |n_i . v|
        abs_dots = np.abs(dots)
        m = np.max(abs_dots)
        if m > max_proj:
            max_proj = m
    
    return 2.0 * max_proj / np.sqrt(3)

def _check_no_overlap(centers, min_dist=SQRT3 - 1e-9):
    """Check that no two inner hexagons overlap (center distance >= sqrt(3))."""
    n = len(centers)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_dist:
                return False
    return True

def _seed_configuration():
    """
    Return the standard concentric ring configuration:
    - 1 hexagon at center
    - 6 hexagons in first ring at distance sqrt(3)
    - 4 hexagons in second ring at distance 3 (filling gaps)
    All inner hexagons have the same orientation (30 degrees).
    """
    eps = 1e-10
    centers = []
    angles = []
    
    # Center
    centers.append([0.0, 0.0])
    angles.append(30.0)
    
    # First ring: 6 hexagons at distance sqrt(3)
    for k in range(6):
        angle = np.deg2rad(k * 60)
        r = SQRT3 + eps
        centers.append([r * np.cos(angle), r * np.sin(angle)])
        angles.append(30.0)
    
    # Second ring: 4 hexagons at distance 3, at angles 30, 90, 150, 210 degrees
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        r = 3.0 + eps
        centers.append([r * np.cos(angle), r * np.sin(angle)])
        angles.append(30.0)
    
    return np.array(centers), np.array(angles)

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    safety_margin = 5.0
    
    # Start with the seed configuration
    centers, inner_angles = _seed_configuration()
    outer_angle = 30.0
    
    # Compute initial side length
    best_side = _min_enclosing_side(centers, inner_angles, outer_angle)
    best_centers = centers.copy()
    best_outer_angle = outer_angle
    
    # Verify no overlap
    if not _check_no_overlap(best_centers):
        # Fallback: spread out slightly
        best_centers = best_centers * 1.001
        best_side = _min_enclosing_side(best_centers, inner_angles, outer_angle)
    
    # Phase 1: Optimize the outer hexagon rotation angle using the fast projection method
    # The inner hexagons are fixed at the seed positions; we only vary the outer rotation
    # to find the tightest enclosing hexagon.
    
    def _side_for_rotation(theta_deg):
        return _min_enclosing_side(best_centers, inner_angles, theta_deg)
    
    # Coarse grid search over [0, 60) degrees (periodic with period 60)
    best_theta = 30.0
    best_theta_side = best_side
    
    if time.time() < DEADLINE - safety_margin - 10:
        # Coarse grid
        thetas = np.linspace(0, 60, 601)
        for t in thetas:
            if time.time() > DEADLINE - safety_margin - 10:
                break
            s = _side_for_rotation(float(t))
            if s < best_theta_side:
                best_theta_side = s
                best_theta = float(t)
        
        # Refine with minimize_scalar around the best point
        step = 60.0 / 600
        lo = max(0.0, best_theta - 10 * step)
        hi = min(60.0, best_theta + 10 * step)
        
        try:
            res = minimize_scalar(_side_for_rotation, bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-14, 'maxiter': 2000})
            if res.fun < best_theta_side:
                best_theta_side = res.fun
                best_theta = res.x
        except Exception:
            pass
        
        # Ultra-tight polish
        if time.time() < DEADLINE - safety_margin - 5:
            lo2 = max(0.0, best_theta - 1e-10)
            hi2 = min(60.0, best_theta + 1e-10)
            try:
                res = minimize_scalar(_side_for_rotation, bounds=(lo2, hi2), method='bounded',
                                      options={'xatol': 1e-16, 'maxiter': 500})
                if res.fun < best_theta_side:
                    best_theta_side = res.fun
                    best_theta = res.x
            except Exception:
                pass
    
    if best_theta_side < best_side:
        best_side = best_theta_side
        best_outer_angle = best_theta
    
    # Phase 2: Try small perturbations of the inner hexagon positions to see if we can
    # reduce the side length further. Use a fast objective based on projection.
    
    def _fast_objective(params):
        """
        Fast objective: outer side length + small penalty for overlaps.
        params: flat array of length 3*n: [x1, y1, a1, x2, y2, a2, ...]
        """
        n = N
        centers_arr = params[:2 * n].reshape(n, 2)
        angles_arr = params[2 * n:]
        
        # Check for overlaps (center distance)
        min_dist = SQRT3 - 1e-9
        overlap_penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(centers_arr[i] - centers_arr[j])
                if d < min_dist:
                    overlap_penalty += 100.0 * (min_dist - d) ** 2
        
        side = _min_enclosing_side(centers_arr, angles_arr, best_outer_angle)
        return side + overlap_penalty
    
    # Multi-start local optimization
    remaining = DEADLINE - safety_margin - (time.time() - start_time)
    if remaining > 10:
        # Use Powell's method with multiple starts
        init_params = np.concatenate([best_centers.flatten(), inner_angles.flatten()])
        
        best_params = init_params.copy()
        best_val = _fast_objective(best_params)
        
        n_starts = min(8, int(remaining / 5))
        for i in range(n_starts):
            if time.time() > DEADLINE - safety_margin - 5:
                break
            
            if i == 0:
                x_start = init_params.copy()
            else:
                x_start = init_params + np.random.randn(len(init_params)) * (0.05 * (i + 1))
                x_start[2::3] = np.mod(x_start[2::3], 360)
            
            try:
                res = minimize(_fast_objective, x_start, method='Powell',
                              options={'maxiter': 200, 'disp': False, 'ftol': 1e-12, 'xtol': 1e-12})
                if res.fun < best_val:
                    best_val = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass
        
        # Check if the optimized configuration is valid
        if best_val < best_side:
            candidate_centers = best_params[:2 * N].reshape(N, 2)
            candidate_angles = best_params[2 * N:]
            
            if _check_no_overlap(candidate_centers):
                candidate_side = _min_enclosing_side(candidate_centers, candidate_angles, best_outer_angle)
                if candidate_side < best_side:
                    best_side = candidate_side
                    best_centers = candidate_centers
                    inner_angles = candidate_angles
    
    # Phase 3: Final polish with very tight Nelder-Mead on the outer angle
    remaining = DEADLINE - safety_margin - (time.time() - start_time)
    if remaining > 3:
        def _final_objective(theta):
            return _min_enclosing_side(best_centers, inner_angles, float(theta))
        
        try:
            res = minimize_scalar(_final_objective, bounds=(best_outer_angle - 0.01, best_outer_angle + 0.01),
                                  method='bounded', options={'xatol': 1e-15, 'maxiter': 1000})
            if res.fun < best_side:
                best_side = res.fun
                best_outer_angle = res.x
        except Exception:
            pass
    
    # Build output
    inner_rows = []
    for i in range(N):
        inner_rows.append([float(best_centers[i, 0]), float(best_centers[i, 1]), float(inner_angles[i])])
    
    center = (0.0, 0.0)
    side = float(best_side)
    angle_deg = float(best_outer_angle)
    
    return inner_rows, center, side, angle_deg
