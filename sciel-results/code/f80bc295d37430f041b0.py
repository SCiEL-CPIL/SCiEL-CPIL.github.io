import numpy as np
import time
from scipy.optimize import minimize


def _hex_vertices(cx, cy, d, a_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side d, angle a_deg."""
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack((cx + d * np.cos(angles), cy + d * np.sin(angles)))


def _min_containing_hex_side(inner):
    """
    Compute the minimum side length s of a regular hexagon (centered at origin, angle 0)
    that contains all vertices of all inner hexagons.
    Uses the support function: for a regular hexagon with side s centered at origin,
    the distance from center to each edge is s * sqrt(3)/2.
    A point (x,y) is inside if for all 6 edge normals n_k, dot(n_k, p) <= s * sqrt(3)/2.
    So s >= max_k dot(n_k, p) / (sqrt(3)/2) for all points p.
    """
    # Edge normals for a regular hexagon with angle 0 (vertices at angles 0,60,120,180,240,300)
    # Edges are between consecutive vertices. The inward normal to edge k points toward center.
    # For a hexagon with vertices at angles 0,60,...,300, edge from v_k to v_{k+1} has 
    # outward normal at angle (60k + 30) degrees.
    # Inward normal = -outward normal, so inward normal angle = 60k + 210 degrees.
    # Actually, let's think differently.
    
    # A regular hexagon with side s, center at origin, "flat-top" orientation (angle 0 means 
    # vertex at angle 0) has vertices at:
    # (s, 0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s, 0), (-s/2, -s*sqrt(3)/2), (s/2, -s*sqrt(3)/2)
    
    # The 6 edges have outward normals at angles 30, 90, 150, 210, 270, 330 degrees.
    # For a point to be inside, dot(outward_normal, point) <= s * sqrt(3)/2 for all 6 normals.
    
    # Outward normal angles: 30 + 60*i for i=0..5
    normal_angles = np.deg2rad(30.0 + 60.0 * np.arange(6))
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    max_s = 0.0
    for row in inner:
        cx, cy, a = row
        hv = _hex_vertices(cx, cy, 1.0, a)
        for v in hv:
            for n in normals:
                val = np.dot(n, v)
                required_s = val / (np.sqrt(3.0) / 2.0)
                if required_s > max_s:
                    max_s = required_s
    return max_s


def _check_validity(inner, outer, s):
    """Validate packing; return (is_valid, error_msg)."""
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)):
        return False, "non-finite values"
    if s < np.sqrt(12.0) - 1e-12:
        return False, "s < sqrt(12)"
    
    # Outer hexagon vertices must be distinct
    ov = _hex_vertices(outer[0], outer[1], s, outer[2])
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(ov[i] - ov[j]) < 1e-10:
                return False, "outer vertices not distinct"
    
    # Check unit edge length for inner hexagons
    for row in inner:
        cx, cy, a = row
        hv = _hex_vertices(cx, cy, 1.0, a)
        for i in range(6):
            edge_len = np.linalg.norm(hv[(i + 1) % 6] - hv[i])
            if abs(edge_len - 1.0) > 1e-8:
                return False, f"unit edge length differs: {edge_len}"
    
    # Check inner-inner separation using SAT
    def _separated(v1, v2):
        """Check if two hexagons are separated (non-overlapping) using SAT."""
        edges = []
        for i in range(6):
            edges.append(v1[(i + 1) % 6] - v1[i])
            edges.append(v2[(i + 1) % 6] - v2[i])
        
        for edge in edges:
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal /= norm
            
            proj1 = v1 @ normal
            proj2 = v2 @ normal
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            gap1 = min2 - max1
            gap2 = min1 - max2
            if gap1 > 1e-6 or gap2 > 1e-6:
                return True
        
        return False
    
    inner_verts = [_hex_vertices(row[0], row[1], 1.0, row[2]) for row in inner]
    for i in range(12):
        for j in range(i + 1, 12):
            if not _separated(inner_verts[i], inner_verts[j]):
                return False, f"inner hexagons {i} and {j} overlap"
    
    # Check inner vertices inside outer hexagon
    for row in inner:
        cx, cy, a = row
        hv = _hex_vertices(cx, cy, 1.0, a)
        for v in hv:
            for k in range(6):
                p1 = ov[k]
                p2 = ov[(k + 1) % 6]
                edge = p2 - p1
                to_v = v - p1
                cross = edge[0] * to_v[1] - edge[1] * to_v[0]
                if cross < -1e-6:
                    return False, f"inner vertex outside outer hexagon"
    
    return True, ""


def _compute_min_s(inner, outer_center=np.array([0.0, 0.0]), outer_angle=0.0):
    """
    Compute the minimum side length s such that all inner hexagon vertices 
    are inside the outer hexagon with given center and angle.
    """
    # Shift inner hexagons so outer center is at origin
    inner_shifted = inner.copy()
    inner_shifted[:, 0] -= outer_center[0]
    inner_shifted[:, 1] -= outer_center[1]
    
    # For an outer hexagon with center at origin, side s, angle a_outer:
    # Vertices at angles a_outer + 60*i, i=0..5
    # Edges connect consecutive vertices. The outward normal to edge k is at angle 
    # a_outer + 60*k + 30 degrees.
    # A point p is inside if dot(outward_normal_k, p) <= s * sqrt(3)/2 for all k.
    # So s >= max_k dot(outward_normal_k, p) / (sqrt(3)/2) for all points p.
    
    a_outer = np.deg2rad(outer_angle)
    normal_angles = a_outer + np.deg2rad(30.0 + 60.0 * np.arange(6))
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    max_s = 0.0
    for row in inner_shifted:
        cx, cy, a = row
        hv = _hex_vertices(cx, cy, 1.0, a)
        for v in hv:
            for n in normals:
                val = np.dot(n, v)
                required_s = val / (np.sqrt(3.0) / 2.0)
                if required_s > max_s:
                    max_s = required_s
    return max_s


def _rotate_point(x, y, angle_deg):
    """Rotate point (x, y) by angle_deg degrees around origin."""
    a = np.deg2rad(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    return ca * x - sa * y, sa * x + ca * y


def _apply_transform(local_positions, rot_deg, cx, cy):
    """
    Apply global rotation and translation to local positions.
    Returns array of (x, y, angle_deg) for each hexagon.
    The local positions are relative to the outer hexagon center.
    """
    result = []
    for px, py, pa in local_positions:
        rx, ry = _rotate_point(px, py, rot_deg)
        final_x = rx + cx
        final_y = ry + cy
        final_angle = (pa + rot_deg) % 360.0
        result.append([final_x, final_y, final_angle])
    return np.array(result)


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    
    Strategy: Use a fixed rigid-body arrangement of 12 hexagons (hexagonal close-packed),
    then optimize only the global rotation angle and outer hexagon center offset.
    This reduces the search space from 36 to 3 variables (rotation, cx, cy) and
    eliminates pairwise separation checks since the arrangement is fixed.
    """
    start_time = time.time()
    deadline = start_time + 580.0
    
    sqrt3 = np.sqrt(3.0)
    
    # Fixed rigid-body arrangement: 12 hexagons in a hexagonal close-packed pattern
    # Center + 6 first shell + 5 second shell (one position left empty for asymmetry)
    # All hexagons have angle 0 (flat-top) in the local frame
    
    # First shell: 6 hexagons at distance 3 from center
    # Directions: 0, 60, 120, 180, 240, 300 degrees
    first_shell = []
    for i in range(6):
        a = np.deg2rad(60.0 * i)
        first_shell.append([3.0 * np.cos(a), 3.0 * np.sin(a), 0.0])
    
    # Second shell: 6 positions at distance sqrt(3)*3 from center
    # But we only use 5 of them (leave one empty for better packing)
    second_shell_full = []
    for i in range(6):
        a = np.deg2rad(60.0 * i + 30.0)  # Offset by 30 degrees
        r = 3.0 * sqrt3  # Distance for second shell
        second_shell_full.append([r * np.cos(a), r * np.sin(a), 0.0])
    
    # Use all 6 second shell positions for now (12 total: 1 + 6 + 5)
    # Actually, let's use 1 center + 6 first shell + 5 second shell = 12
    # Leave out one second shell position to allow optimization
    
    local_positions = np.array([
        [0.0, 0.0, 0.0],  # Center
    ] + first_shell + second_shell_full[:5])  # 1 + 6 + 5 = 12
    
    # Optimize: global rotation angle, outer center offset (cx, cy)
    # Parameters: [rotation_deg, outer_cx, outer_cy]
    
    def objective(params):
        """
        params: [rotation_deg, outer_cx, outer_cy]
        Returns: computed s
        """
        rot_deg = params[0]
        outer_cx = params[1]
        outer_cy = params[2]
        
        # Rotate all local positions by rot_deg
        rotated = []
        for px, py, pa in local_positions:
            rx, ry = _rotate_point(px, py, rot_deg)
            # Inner hexagon angle also rotates
            ra = (pa + rot_deg) % 360.0
            rotated.append([rx, ry, ra])
        
        inner = np.array(rotated)
        
        # Compute min s with outer hexagon centered at (outer_cx, outer_cy), angle 0
        s_val = _compute_min_s(inner, np.array([outer_cx, outer_cy]), 0.0)
        
        return s_val
    
    # Initial guess: rotation = 0, outer center at centroid of inner hexagons
    centroid_x = np.mean(local_positions[:, 0])
    centroid_y = np.mean(local_positions[:, 1])
    x0 = np.array([0.0, centroid_x, centroid_y])
    
    # Compute initial s
    s_init = objective(x0)
    inner_init = _apply_transform(local_positions, x0[0], x0[1], x0[2])
    outer_init = np.array([x0[1], x0[2], 0.0])
    s_init += 0.001
    
    # Validate initial
    valid, msg = _check_validity(inner_init, outer_init, s_init)
    if not valid:
        print(f"Initial guess invalid: {msg}")
        s_init += 1.0
        valid, msg = _check_validity(inner_init, outer_init, s_init)
        if not valid:
            # Fallback
            s_fallback = 20.0
            return inner_init, outer_init, s_fallback
    
    best_s = s_init
    best_inner = inner_init.copy()
    best_outer = outer_init.copy()
    print(f"Initial: s = {best_s:.6f}, 1/s = {1/best_s:.6f}")
    
    # Optimize rotation and outer center
    try:
        result = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8,
                                   'adaptive': True})
        
        if result.fun < best_s - 1e-10:
            rot_opt, cx_opt, cy_opt = result.x
            inner_opt = _apply_transform(local_positions, rot_opt, cx_opt, cy_opt)
            s_opt = _compute_min_s(inner_opt, np.array([cx_opt, cy_opt]), 0.0) + 0.0001
            outer_opt = np.array([cx_opt, cy_opt, 0.0])
            
            valid, msg = _check_validity(inner_opt, outer_opt, s_opt)
            if valid:
                best_inner = inner_opt
                best_outer = outer_opt
                best_s = s_opt
                print(f"Optimized: s = {best_s:.6f}, 1/s = {1/best_s:.6f}")
            else:
                print(f"Optimized result invalid: {msg}")
        
        # Second round with tighter tolerance
        if time.time() < deadline - 60:
            result2 = minimize(objective, result.x, method='Nelder-Mead',
                              options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10,
                                       'adaptive': True})
            
            if result2.fun < best_s - 1e-10:
                rot_opt, cx_opt, cy_opt = result2.x
                inner_opt = _apply_transform(local_positions, rot_opt, cx_opt, cy_opt)
                s_opt = _compute_min_s(inner_opt, np.array([cx_opt, cy_opt]), 0.0) + 0.0001
                outer_opt = np.array([cx_opt, cy_opt, 0.0])
                
                valid, msg = _check_validity(inner_opt, outer_opt, s_opt)
                if valid:
                    best_inner = inner_opt
                    best_outer = outer_opt
                    best_s = s_opt
                    print(f"Further optimized: s = {best_s:.6f}, 1/s = {1/best_s:.6f}")
    
    except Exception as e:
        print(f"Optimization error: {e}")
    
    # Final validation
    valid, msg = _check_validity(best_inner, best_outer, best_s)
    if not valid:
        print(f"Final result invalid: {msg}, using fallback")
        s = 20.0
        return best_inner, best_outer, s
    
    return best_inner, best_outer, best_s


def _apply_transform(local_positions, rot_deg, cx, cy):
    """
    Apply global rotation and translation to local positions.
    Returns array of (x, y, angle_deg) for each hexagon.
    """
    result = []
    for px, py, pa in local_positions:
        rx, ry = _rotate_point(px, py, rot_deg)
        # Translate so that the centroid is at (cx, cy)
        # Actually, we want the outer hexagon center at (cx, cy)
        # The inner hexagons are positioned relative to the outer center
        # So we add (cx, cy) to the rotated positions
        final_x = rx + cx
        final_y = ry + cy
        final_angle = (pa + rot_deg) % 360.0
        result.append([final_x, final_y, final_angle])
    return np.array(result)


# EVOLVE-BLOCK-END
