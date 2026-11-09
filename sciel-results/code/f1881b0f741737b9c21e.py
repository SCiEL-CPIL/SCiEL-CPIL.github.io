import numpy as np
import time
from scipy.optimize import minimize
import sys

N = 12
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin for 150s limit

def get_hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon."""
    angle_rad = np.radians(angle_deg)
    verts = np.zeros((6, 2))
    for k in range(6):
        a = angle_rad + k * np.pi / 3
        verts[k, 0] = cx + side * np.cos(a)
        verts[k, 1] = cy + side * np.sin(a)
    return verts

def get_hex_edge_normals(verts):
    """Return outward normals for a hexagon."""
    normals = np.zeros((6, 2))
    for i in range(6):
        a = verts[i]
        b = verts[(i + 1) % 6]
        edge_vec = b - a
        normal = np.array([-edge_vec[1], edge_vec[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-10:
            normal /= norm
        normals[i] = normal
    return normals

def compute_gaps(verts1, verts2, normals1, normals2):
    """
    Compute the gap in each of the 12 normal directions.
    """
    gaps = []
    
    # Project verts2 onto normals1
    for i in range(6):
        n = normals1[i]
        proj2 = np.dot(verts2, n)
        min2, max2 = proj2.min(), proj2.max()
        
        proj1 = np.dot(verts1, n)
        min1, max1 = proj1.min(), proj1.max()
        
        if max1 < min2:
            gap = min2 - max1
        elif max2 < min1:
            gap = min1 - max2
        else:
            gap = 0.0
        gaps.append(gap)
        
    # Project verts1 onto normals2
    for i in range(6):
        n = normals2[i]
        proj1 = np.dot(verts1, n)
        min1, max1 = proj1.min(), proj1.max()
        
        proj2 = np.dot(verts2, n)
        min2, max2 = proj2.min(), proj2.max()
        
        if max1 < min2:
            gap = min2 - max1
        elif max2 < min1:
            gap = min1 - max2
        else:
            gap = 0.0
        gaps.append(gap)
    
    return np.array(gaps)

def smooth_penetration_depth(gaps, beta=10.0):
    """
    Compute a smooth approximation of the penetration depth.
    """
    g_max = np.max(gaps)
    if g_max > 1e-6:
        return 0.0
    
    shifted = gaps - g_max
    sum_exp = np.sum(np.exp(-beta * shifted))
    if sum_exp < 1e-300:
        return 0.0
    soft_min = g_max - (1.0 / beta) * np.log(sum_exp)
    
    depth = -soft_min
    if depth < 0:
        depth = 0.0
    return depth

def penalty_overlap(h1, h2, beta=10.0):
    """
    Compute smooth overlap penalty between two hexagons.
    """
    v1 = get_hex_vertices(h1[0], h1[1], 1.0, h1[2])
    v2 = get_hex_vertices(h2[0], h2[1], 1.0, h2[2])
    
    n1 = get_hex_edge_normals(v1)
    n2 = get_hex_edge_normals(v2)
    
    gaps = compute_gaps(v1, v2, n1, n2)
    depth = smooth_penetration_depth(gaps, beta)
    return depth * depth

def penalty_boundary(h, outer_center, outer_side, outer_angle, beta=10.0):
    """
    Compute penalty if hexagon h is outside the outer hexagon.
    """
    v_inner = get_hex_vertices(h[0], h[1], 1.0, h[2])
    v_outer = get_hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    n_outer = get_hex_edge_normals(v_outer)
    
    max_violation = 0.0
    for i in range(6):
        n = n_outer[i]
        
        proj_inner = np.dot(v_inner, n)
        min_in, max_in = proj_inner.min(), proj_inner.max()
        
        proj_outer = np.dot(v_outer, n)
        min_out, max_out = proj_outer.min(), proj_outer.max()
        
        viol1 = max_in - max_out
        viol2 = min_out - min_in
        
        max_violation = max(max_violation, viol1, viol2)
    
    if max_violation > 0:
        return max_violation * max_violation
    else:
        return 0.0

def objective(x, weight_bound=1000.0, weight_overlap=1000.0):
    """
    Objective function for optimization.
    """
    inner = []
    for i in range(N):
        cx = x[i*3]
        cy = x[i*3+1]
        th = x[i*3+2]
        inner.append((cx, cy, th))
    
    X = x[36]
    Y = x[37]
    Phi = x[38]
    L = x[39]
    
    if L < 3.0:
        return 1000.0 + (3.0 - L)**2
    
    outer_center = (X, Y)
    outer_angle = Phi
    
    bound_pen = 0.0
    for i in range(N):
        bound_pen += penalty_boundary(inner[i], outer_center, L, outer_angle)
    
    overlap_pen = 0.0
    for i in range(N):
        for j in range(i+1, N):
            overlap_pen += penalty_overlap(inner[i], inner[j])
    
    return L + weight_bound * bound_pen + weight_overlap * overlap_pen

def is_valid(inner, center, side, angle, tol=1e-5):
    """Check if the configuration is valid (no overlaps, inside boundary)."""
    # Check overlaps
    for i in range(N):
        for j in range(i+1, N):
            h1 = (inner[i][0], inner[i][1], inner[i][2])
            h2 = (inner[j][0], inner[j][1], inner[j][2])
            v1 = get_hex_vertices(h1[0], h1[1], 1.0, h1[2])
            v2 = get_hex_vertices(h2[0], h2[1], 1.0, h2[2])
            n1 = get_hex_edge_normals(v1)
            n2 = get_hex_edge_normals(v2)
            gaps = compute_gaps(v1, v2, n1, n2)
            # If min gap < -tol, they overlap
            if np.min(gaps) < -tol:
                return False
    
    # Check boundary
    v_outer = get_hex_vertices(center[0], center[1], side, angle)
    n_outer = get_hex_edge_normals(v_outer)
    for i in range(N):
        h = (inner[i][0], inner[i][1], inner[i][2])
        v_inner = get_hex_vertices(h[0], h[1], 1.0, h[2])
        for k in range(6):
            n = n_outer[k]
            proj_in = np.dot(v_inner, n)
            proj_out = np.dot(v_outer, n)
            if proj_in.min() < proj_out.min() - tol or proj_in.max() > proj_out.max() + tol:
                return False
    return True

def construct():
    global START_TIME
    START_TIME = time.time()
    
    # Initial configuration: Hexagonal lattice (3 rows, 4 cols)
    # This is a valid non-overlapping configuration.
    dx = 3.0
    dy = np.sqrt(3)
    
    x0 = np.zeros(40)
    idx = 0
    # Center the lattice around (0,0) for easier optimization
    # Row 0: y=0, x = -4.5, -1.5, 1.5, 4.5 (offset -4.5)
    # Row 1: y=sqrt(3), x = -3.0, 0.0, 3.0, 6.0 (offset -3.0) -> wait, honeycomb offset is 1.5
    # Let's generate relative to center (0,0)
    # Row 0 (y=0): x = -4.5, -1.5, 1.5, 4.5
    # Row 1 (y=sqrt(3)): x = -3.0, 0.0, 3.0, 6.0 ? No.
    # Standard honeycomb:
    # Row i, Col j
    # x = j * 3 + (i % 2) * 1.5
    # y = i * sqrt(3)
    # For 3 rows (0,1,2) and 4 cols (0,1,2,3):
    # R0: 0, 3, 6, 9 -> y=0
    # R1: 1.5, 4.5, 7.5, 10.5 -> y=sqrt(3)
    # R2: 0, 3, 6, 9 -> y=2*sqrt(3)
    
    # Center of this block:
    # X_center = (0 + 10.5) / 2 = 5.25
    # Y_center = (0 + 2*sqrt(3)) / 2 = sqrt(3)
    
    cx_center = 5.25
    cy_center = np.sqrt(3)
    
    for row in range(3):
        for col in range(4):
            cx = col * 3.0 + (row % 2) * 1.5
            cy = row * np.sqrt(3)
            # Shift to center
            x0[idx] = cx - cx_center
            x0[idx+1] = cy - cy_center
            x0[idx+2] = 0.0
            idx += 3
    
    # Outer hexagon initial guess
    x0[36] = 0.0 # Center X
    x0[37] = 0.0 # Center Y
    x0[38] = 0.0 # Rotation
    # Estimate L:
    # Max distance from center to any vertex of inner hexes.
    # Furthest inner center is at approx (5.25-5.25, 2*sqrt(3)-sqrt(3)) = (0, sqrt(3))?
    # No, R2 col 3 is (9, 2*sqrt(3)). Shifted: (9-5.25, 2*sqrt(3)-sqrt(3)) = (3.75, sqrt(3)).
    # Distance to center: sqrt(3.75^2 + 3) = sqrt(14 + 3) = sqrt(17) approx 4.1.
    # Add radius of inner hex (1). So L > 5.1.
    # Let's start with L = 6.0 to be safe.
    x0[39] = 6.0
    
    bounds = []
    for i in range(N):
        bounds.append((-10, 10)) # x
        bounds.append((-10, 10)) # y
        bounds.append((-180, 180)) # angle
    bounds.append((-10, 10)) # X
    bounds.append((-10, 10)) # Y
    bounds.append((-180, 180)) # Phi
    bounds.append((3.0, 10.0)) # L
    
    best_x = x0.copy()
    best_val = objective(best_x)
    
    # Optimization
    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 5000})
        
        if res.fun < best_val:
            best_x = res.x
            best_val = res.fun
            
        # Refine if time permits
        if time.time() - START_TIME < TIME_LIMIT - 20:
            # Try to tighten L by increasing penalty weights
            res2 = minimize(objective, best_x, method='L-BFGS-B', bounds=bounds, 
                            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10},
                            args=(5000.0, 5000.0))
            if res2.fun < best_val:
                best_x = res2.x
                best_val = res2.fun
                
    except Exception as e:
        # If optimization fails, stick with initial guess
        pass
    
    # Extract result
    inner = []
    for i in range(N):
        inner.append([best_x[i*3], best_x[i*3+1], best_x[i*3+2]])
    
    X = best_x[36]
    Y = best_x[37]
    Phi = best_x[38]
    L = best_x[39]
    
    # Validation and Fallback
    if not is_valid(inner, (X, Y), L, Phi):
        # If the optimized result is invalid, return the initial valid lattice with a safe L
        # We need to calculate a safe L for the initial lattice
        # The initial lattice is centered at (0,0) in x0, but we shifted it.
        # Let's just use the x0 configuration with a generous L.
        # Reconstruct inner from x0
        inner_fallback = []
        for i in range(N):
            inner_fallback.append([x0[i*3], x0[i*3+1], x0[i*3+2]])
        
        # Calculate required L for fallback
        # Find max distance from origin (0,0) to any vertex of any inner hex
        max_dist = 0
        for h in inner_fallback:
            v = get_hex_vertices(h[0], h[1], 1.0, h[2])
            dists = np.linalg.norm(v, axis=1)
            max_dist = max(max_dist, np.max(dists))
        
        # The circumradius of the outer hex is L.
        # We need L >= max_dist.
        L_fallback = max_dist + 0.1 # small margin
        X_fallback = 0.0
        Y_fallback = 0.0
        Phi_fallback = 0.0
        
        return inner_fallback, (X_fallback, Y_fallback), L_fallback, Phi_fallback
    
    return inner, (X, Y), L, Phi
