# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _hex_vertices(cx, cy, s, a_deg):
    """Return 6 vertices of a hexagon with center (cx,cy), side s, angle a_deg."""
    a = np.radians(a_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _edge_normals(vertices):
    """Return 6 outward edge normals (unit vectors) for a convex hexagon."""
    n = len(vertices)
    normals = np.zeros((n, 2))
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        edge = p2 - p1
        # Outward normal: rotate edge by -90 degrees and normalize
        norm = np.array([edge[1], -edge[0]])
        norm /= np.linalg.norm(norm)
        normals[i] = norm
    return normals

def _check_separation(v1, v2, eps=1e-6):
    """Check if two hexagons are separated (no overlap) using edge normals.
    Returns True if separated, False if overlapping."""
    n1 = len(v1)
    n2 = len(v2)
    normals1 = _edge_normals(v1)
    normals2 = _edge_normals(v2)
    
    for i in range(n1):
        proj1 = v1 @ normals1[i]
        proj2 = v2 @ normals1[i]
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if max1 + eps < min2 or max2 + eps < min1:
            return True  # Separated along this normal
    
    for i in range(n2):
        proj1 = v1 @ normals2[i]
        proj2 = v2 @ normals2[i]
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if max1 + eps < min2 or max2 + eps < min1:
            return True  # Separated along this normal
    
    return False  # Not separated = overlapping

def _check_inside_outer(vertices_inner, vertices_outer):
    """Check if all inner vertices are inside or on the outer hexagon."""
    n_outer = len(vertices_outer)
    for p in vertices_inner:
        for i in range(n_outer):
            p1 = vertices_outer[i]
            p2 = vertices_outer[(i + 1) % n_outer]
            cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
            if cross < -1e-6:
                return False
    return True

def _compute_outer_side(inner_hex_data, outer_center, outer_angle):
    """Compute minimum outer hexagon side length that contains all inner hexagons."""
    # For each inner hexagon, compute the max distance from outer center in each of 6 directions
    # The outer hexagon with side s has inradius = s * sqrt(3) / 2
    # A point (x,y) is inside if for each of 6 edge normals, dot(point - center, normal) <= inradius
    
    outer_angle_rad = np.radians(outer_angle)
    normals = np.zeros((6, 2))
    for i in range(6):
        a = outer_angle_rad + i * np.pi / 3 + np.pi / 6  # edge normal direction
        normals[i] = np.array([np.cos(a), np.sin(a)])
    
    max_proj = 0.0
    for (cx, cy, a_deg) in inner_hex_data:
        verts = _hex_vertices(cx, cy, 1.0, a_deg)
        for p in verts:
            shifted = p - outer_center
            for n in normals:
                proj = np.dot(shifted, n)
                if proj > max_proj:
                    max_proj = proj
    
    # inradius = s * sqrt(3) / 2, so s = max_proj / (sqrt(3)/2) = 2 * max_proj / sqrt(3)
    return 2.0 * max_proj / np.sqrt(3)

def _objective(params):
    """Objective function for optimization.
    params: [x1,y1,a1, x2,y2,a2, ..., x11,y11,a11, outer_x, outer_y, outer_angle]
    First 33 params: 11 hexagons (x, y, angle)
    Last 3 params: outer center (x, y) and angle
    """
    n_hex = 11
    inner = params[:n_hex * 3].reshape(n_hex, 3)
    outer_center = np.array([params[33], params[34]])
    outer_angle = params[35]
    
    # Compute outer side
    outer_side = _compute_outer_side(inner, outer_center, outer_angle)
    
    # Check separation between all pairs
    for i in range(n_hex):
        for j in range(i + 1, n_hex):
            v1 = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
            v2 = _hex_vertices(inner[j, 0], inner[j, 1], 1.0, inner[j, 2])
            if not _check_separation(v1, v2):
                return outer_side + 100.0  # Penalty for overlap
    
    # Check all inner vertices inside outer
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    for i in range(n_hex):
        v_inner = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
        if not _check_inside_outer(v_inner, outer_verts):
            return outer_side + 100.0  # Penalty for being outside
    
    return outer_side

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    # Initial symmetric configuration:
    # Center hexagon at origin
    # 6 hexagons in a ring at distance ~2.5 (touching)
    # 4 more hexagons
    
    # For unit hexagons, the distance between centers of touching hexagons is 2 * inradius = sqrt(3)
    # inradius = sqrt(3)/2, so distance = sqrt(3) * 2 / 2 = ... let me recalculate
    # Two unit hexagons touch when center distance = 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3)
    
    # Actually for regular hexagons with side 1, the distance between centers when edges touch
    # is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3) ≈ 1.732
    
    # Let me set up a better initial configuration
    # Center at (0,0)
    # 6 around at distance sqrt(3) in 6 directions
    # 4 more further out
    
    d = np.sqrt(3)  # distance for touching
    
    init = [0, 0, 0]  # center
    for i in range(6):
        angle = i * 60
        init.extend([d * np.cos(np.radians(angle)), d * np.sin(np.radians(angle)), 0])
    # 4 more: at distance 2*d in 4 diagonal directions
    for i in range(4):
        angle = 30 + i * 90
        init.extend([2 * d * np.cos(np.radians(angle)), 2 * d * np.sin(np.radians(angle)), 0])
    
    # Outer center and angle
    init.extend([0, 0, 0])
    
    x0 = np.array(init)
    
    best_result = None
    best_side = np.inf
    
    # Try Nelder-Mead optimization
    try:
        result = minimize(_objective, x0, method='Nelder-Mead', 
                         options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
        elapsed = time.time() - start_time
        if elapsed < deadline and result.fun < best_side:
            best_side = result.fun
            best_result = result.x
    except Exception as e:
        pass
    
    # If we have time, try more optimizations with different initial configs
    if time.time() - start_time < deadline - 60:
        # Try with slightly perturbed initial conditions
        for trial in range(3):
            if time.time() - start_time > deadline - 30:
                break
            x0_trial = x0 + 0.1 * np.random.randn(len(x0))
            # Keep angles reasonable
            for i in range(11):
                x0_trial[i*3+2] = np.random.uniform(-30, 30)
            
            try:
                result = minimize(_objective, x0_trial, method='Nelder-Mead',
                                 options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-6})
                if result.fun < best_side:
                    best_side = result.fun
                    best_result = result.x
            except Exception:
                pass
    
    if best_result is None:
        # Fallback to initial configuration
        best_result = x0
        best_side = _objective(x0)
    
    inner_hex_data = best_result[:33].reshape(11, 3)
    outer_hex_data = np.array([best_result[33], best_result[34], best_result[35]])
    outer_hex_side_length = float(best_side)
    
    # Verify and adjust if needed
    # Ensure side >= sqrt(11)
    if outer_hex_side_length < np.sqrt(11):
        outer_hex_side_length = np.sqrt(11)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
