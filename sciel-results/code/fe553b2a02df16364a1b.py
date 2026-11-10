import numpy as np
import time
from scipy.optimize import minimize


def _hex_vertices(cx, cy, d, a_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side d, angle a_deg."""
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack((cx + d * np.cos(angles), cy + d * np.sin(angles)))


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
            if np.linalg.norm(normal) < 1e-12:
                continue
            normal /= np.linalg.norm(normal)
            
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


def _compute_min_s(inner, outer_angle_deg=0.0):
    """
    Compute the minimal side length s of a regular hexagon centered at origin
    with given angle that contains all inner hexagons.
    """
    a_outer = np.deg2rad(outer_angle_deg)
    # The 6 edge normals of the outer hexagon (pointing inward)
    # Edge i goes from vertex i to vertex i+1
    # Vertex i is at angle (a_outer + 60*i)
    # Edge direction is from vertex i to vertex i+1
    # The inward normal for edge i
    
    # For a regular hexagon with center at origin, the constraint for a point p
    # to be inside is: for each edge, the dot product of (p - midpoint) with 
    # the inward normal must be >= 0
    
    # Actually, let's use the cross product test directly.
    # For outer hexagon with center (0,0), side s, angle a_outer:
    # Vertex i: (s*cos(a_outer + 60*i), s*sin(a_outer + 60*i))
    # Edge i: from vertex i to vertex i+1
    # For point v to be inside: cross(v_{i+1} - v_i, v - v_i) >= 0 for all i
    
    # cross(v_{i+1} - v_i, v - v_i) = cross(v_{i+1} - v_i, v) - cross(v_{i+1} - v_i, v_i)
    
    # Let's compute this more carefully.
    # v_i = s * (cos(a_i), sin(a_i)) where a_i = a_outer + 60*i
    # v_{i+1} = s * (cos(a_{i+1}), sin(a_{i+1}))
    
    # cross(v_{i+1} - v_i, v - v_i) = cross(v_{i+1} - v_i, v) - cross(v_{i+1} - v_i, v_i)
    
    # cross(u, w) = u_x * w_y - u_y * w_x
    
    # Let's parameterize: for a given s, check if all inner vertices are inside.
    # We want the minimum s such that all inner vertices are inside.
    
    # For each inner vertex v, the constraint is:
    # cross(v_{i+1}(s) - v_i(s), v - v_i(s)) >= 0 for all i=0..5
    
    # v_i(s) = s * (cos(a_i), sin(a_i))
    # v_{i+1}(s) = s * (cos(a_{i+1}), sin(a_{i+1}))
    
    # v_{i+1}(s) - v_i(s) = s * (cos(a_{i+1}) - cos(a_i), sin(a_{i+1}) - sin(a_i))
    # v - v_i(s) = v - s * (cos(a_i), sin(a_i))
    
    # cross = s*(cos(a_{i+1})-cos(a_i)) * (v_y - s*sin(a_i)) - s*(sin(a_{i+1})-sin(a_i)) * (v_x - s*cos(a_i))
    # = s*(cos(a_{i+1})-cos(a_i))*v_y - s^2*(cos(a_{i+1})-cos(a_i))*sin(a_i)
    #   - s*(sin(a_{i+1})-sin(a_i))*v_x + s^2*(sin(a_{i+1})-sin(a_i))*cos(a_i)
    # = s*[(cos(a_{i+1})-cos(a_i))*v_y - (sin(a_{i+1})-sin(a_i))*v_x]
    #   + s^2*[-(cos(a_{i+1})-cos(a_i))*sin(a_i) + (sin(a_{i+1})-sin(a_i))*cos(a_i)]
    
    # Let A_i = (cos(a_{i+1})-cos(a_i))*v_y - (sin(a_{i+1})-sin(a_i))*v_x
    # Let B_i = -(cos(a_{i+1})-cos(a_i))*sin(a_i) + (sin(a_{i+1})-sin(a_i))*cos(a_i)
    # B_i = -cos(a_{i+1})*sin(a_i) + cos(a_i)*sin(a_i) + sin(a_{i+1})*cos(a_i) - sin(a_i)*cos(a_i)
    # = sin(a_{i+1})*cos(a_i) - cos(a_{i+1})*sin(a_i)
    # = sin(a_{i+1} - a_i)
    # = sin(60 degrees) = sqrt(3)/2
    
    # So cross = s*A_i + s^2*sqrt(3)/2 >= 0
    # s*A_i + s^2*sqrt(3)/2 >= 0
    # s*(A_i + s*sqrt(3)/2) >= 0
    # Since s > 0: A_i + s*sqrt(3)/2 >= 0
    # s >= -A_i / (sqrt(3)/2) = -2*A_i/sqrt(3)
    
    # So for each inner vertex v and each edge i, we need:
    # s >= max(0, -2*A_i/sqrt(3))
    
    # Where A_i = (cos(a_{i+1})-cos(a_i))*v_y - (sin(a_{i+1})-sin(a_i))*v_x
    
    min_s = np.sqrt(12.0)  # Lower bound
    
    a_i = a_outer + np.deg2rad(60.0 * np.arange(6))
    a_ip1 = a_outer + np.deg2rad(60.0 * np.arange(1, 7))
    
    cos_ai = np.cos(a_i)
    sin_ai = np.sin(a_i)
    cos_aip1 = np.cos(a_ip1)
    sin_aip1 = np.sin(a_ip1)
    
    # For each inner hexagon, compute its 6 vertices
    for row in inner:
        cx, cy, a_inner = row
        hv = _hex_vertices(cx, cy, 1.0, a_inner)
        for v in hv:
            vx, vy = v
            # A_i for each edge
            A = (cos_aip1 - cos_ai) * vy - (sin_aip1 - sin_ai) * vx
            # s >= -2*A_i/sqrt(3) for each i where A_i < 0
            required_s = np.maximum(0, -2.0 * A / np.sqrt(3.0))
            min_s = max(min_s, np.max(required_s))
    
    return min_s


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    """
    start_time = time.time()
    deadline = start_time + 580.0
    
    # Initial guess: symmetric arrangement with all angles at 0
    # 12 hexagons in a hexagonal-like arrangement
    # Center + 6 around center + 6 more
    h3 = np.sqrt(3.0) / 2.0  # height of unit hexagon = sqrt(3)/2
    # For unit hexagons with flat top (angle 0), the horizontal spacing is 2, vertical is 3*sqrt(3)/2
    # Actually, let me think about this more carefully.
    
    # A unit regular hexagon with angle 0 has vertices at:
    # (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5,-sqrt(3)/2), (0.5,-sqrt(3)/2)
    # Width = 2, Height = sqrt(3)
    
    # For packing, adjacent hexagons can be placed at distance 2 in the x-direction
    # or at distance sqrt(3) in the y-direction with x-offset of 1
    
    # Let's use a standard hexagonal grid arrangement
    # Row 0: 4 hexagons at x = -1.5, -0.5, 0.5, 1.5, y = 0
    # Row 1: 4 hexagons at x = -1.0, 0.0, 1.0, 2.0, y = sqrt(3)
    # Row 2: 4 hexagons at x = -1.5, -0.5, 0.5, 1.5, y = 2*sqrt(3)
    
    # Wait, that's 12 hexagons in a 4x3 grid.
    # But we need to check if they fit without overlapping.
    
    # For hexagons with angle 0, the minimum distance between centers for non-overlapping
    # is 2 in the horizontal direction.
    
    # Let me try a different arrangement.
    # Place 12 hexagons in a more compact arrangement.
    
    # Actually, let's start with a simple grid and optimize.
    inner = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [-2.0, 0.0, 0.0],
        [1.0, h3*2.0, 0.0],
        [-1.0, h3*2.0, 0.0],
        [1.0, -h3*2.0, 0.0],
        [-1.0, -h3*2.0, 0.0],
        [3.0, h3*2.0, 0.0],
        [-3.0, h3*2.0, 0.0],
        [3.0, -h3*2.0, 0.0],
        [-3.0, -h3*2.0, 0.0],
        [0.0, h3*4.0, 0.0],
    ])
    
    outer = np.array([0.0, 0.0, 0.0])
    s = _compute_min_s(inner)
    
    # Validate initial guess
    valid, msg = _check_validity(inner, outer, s)
    if not valid:
        print(f"Initial guess invalid: {msg}")
        # Try a larger s
        s = 10.0
        valid, msg = _check_validity(inner, outer, s)
        if not valid:
            print(f"Still invalid: {msg}")
            s = 20.0
            return inner, outer, s
    
    best_s = s
    best_inner = inner.copy()
    best_outer = outer.copy()
    
    # Optimize positions (2D only, angles fixed at 0)
    try:
        def objective(params):
            """
            params: 24 values: 12*(x, y)
            Returns: s + penalty for overlaps
            """
            if time.time() > deadline:
                return 1e10
            
            inner_p = params.reshape(12, 2)
            # Add angle column
            inner_full = np.column_stack([inner_p, np.zeros(12)])
            
            s_p = _compute_min_s(inner_full)
            
            # Penalty for overlaps
            inner_verts = [_hex_vertices(row[0], row[1], 1.0, row[2]) for row in inner_full]
            penalty = 0.0
            
            for i in range(12):
                for j in range(i + 1, 12):
                    # Use SAT to check for overlap
                    v1 = inner_verts[i]
                    v2 = inner_verts[j]
                    
                    # Quick distance check
                    dist = np.linalg.norm(v1[0] - v2[0])
                    if dist > 4.0:
                        continue
                    
                    # SAT check
                    separated = False
                    edges = []
                    for k in range(6):
                        edges.append(v1[(k + 1) % 6] - v1[k])
                        edges.append(v2[(k + 1) % 6] - v2[k])
                    
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
                        if gap1 > 0 or gap2 > 0:
                            separated = True
                            break
                    
                    if not separated:
                        # Compute penetration depth as penalty
                        # Use the minimum gap among all axes
                        min_gap = np.inf
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
                            min_gap = min(min_gap, max(0, -gap1), max(0, -gap2))
                        
                        penalty += min_gap ** 2
            
            return s_p + penalty
        
        x0 = best_inner[:, :2].flatten()
        
        result = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-5, 'maxfev': 10000})
        
        if result.fun < best_s:
            inner_opt = result.x.reshape(12, 2)
            inner_full = np.column_stack([inner_opt, np.zeros(12)])
            s_opt = _compute_min_s(inner_full)
            outer_opt = np.array([0.0, 0.0, 0.0])
            
            valid, msg = _check_validity(inner_full, outer_opt, s_opt)
            if valid:
                best_inner = inner_full.copy()
                best_outer = outer_opt.copy()
                best_s = s_opt
                print(f"Optimized: s = {best_s:.6f}, 1/s = {1/best_s:.6f}")
            else:
                print(f"Optimized result invalid: {msg}")
        
        if time.time() > deadline:
            return best_inner, best_outer, best_s
        
        # Try more iterations if time permits
        if time.time() < deadline - 60:
            result2 = minimize(objective, result.x, method='Nelder-Mead',
                              options={'maxiter': 5000, 'xatol': 1e-7, 'fatol': 1e-7, 'maxfev': 20000})
            
            if result2.fun < best_s:
                inner_opt = result2.x.reshape(12, 2)
                inner_full = np.column_stack([inner_opt, np.zeros(12)])
                s_opt = _compute_min_s(inner_full)
                outer_opt = np.array([0.0, 0.0, 0.0])
                
                valid, msg = _check_validity(inner_full, outer_opt, s_opt)
                if valid:
                    best_inner = inner_full.copy()
                    best_outer = outer_opt.copy()
                    best_s = s_opt
                    print(f"Further optimized: s = {best_s:.6f}, 1/s = {1/best_s:.6f}")
    
    except Exception as e:
        print(f"Optimization error: {e}")
    
    # Final validation
    valid, msg = _check_validity(best_inner, best_outer, best_s)
    if not valid:
        print(f"Final result invalid: {msg}, falling back to safe configuration")
        s = 20.0
        return best_inner, best_outer, s
    
    return best_inner, best_outer, best_s
