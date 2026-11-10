import numpy as np
import time
from scipy.optimize import minimize


def _hex_vertices(cx, cy, d, a_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side d, angle a_deg."""
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack((cx + d * np.cos(angles), cy + d * np.sin(angles)))


def _compute_min_s(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum side length s such that all inner hexagon vertices 
    are inside the outer hexagon with given center and angle.
    """
    outer_center = np.asarray(outer_center, dtype=float)
    inner_shifted = inner.copy()
    inner_shifted[:, 0] -= outer_center[0]
    inner_shifted[:, 1] -= outer_center[1]
    
    a_outer = np.deg2rad(outer_angle_deg)
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


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    
    Strategy: Use a hexagonal lattice representation where 12 unit hexagons are placed
    at fixed relative positions (ensuring non-overlap by construction via lattice spacing),
    then optimize the affine transformation (rotation + uniform scale) that maps this
    fixed lattice configuration into the minimal containing regular hexagon.
    """
    start_time = time.time()
    deadline = start_time + 580.0
    
    sqrt3 = np.sqrt(3.0)
    
    # Define 12 hexagons on a hexagonal lattice with guaranteed non-overlap.
    # For unit hexagons (side 1), the minimum center-to-center distance for non-overlap
    # is 3 (for flat-top orientation, same row) or sqrt(3)*1.5 ≈ 2.598 (adjacent rows).
    # We use a compact arrangement: center + 6 neighbors + 5 more.
    
    # Lattice basis for hexagonal packing of unit hexagons (flat-top):
    # e1 = (3, 0), e2 = (1.5, 2.598)
    # Any position = a*e1 + b*e2 for integers a, b gives non-overlapping hexagons.
    
    # Choose 12 lattice points that form a compact cluster:
    # (0,0), (1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), 
    # (-1,1), (-1,-1), (2,0), (2,1), (2,-1)
    # Let's verify these don't overlap:
    # Distance between (a1,b1) and (a2,b2) in lattice coords:
    # dx = 3*(a1-a2) + 1.5*(b1-b2), dy = 2.598*(b1-b2)
    # dist^2 = (3*da + 1.5*db)^2 + (2.598*db)^2
    # For da=1, db=0: dist = 3 ✓
    # For da=0, db=1: dist = sqrt(1.5^2 + 2.598^2) = sqrt(2.25+6.75) = 3 ✓
    # For da=1, db=1: dist = sqrt(4.5^2 + 2.598^2) = sqrt(20.25+6.75) = sqrt(27) ≈ 5.196 ✓
    # For da=1, db=-1: dist = sqrt(1.5^2 + 2.598^2) = 3 ✓
    # All pairs are at distance >= 3, so no overlap.
    
    lattice_points = np.array([
        [0, 0],
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1],
        [1, 1],
        [1, -1],
        [-1, 1],
        [-1, -1],
        [2, 0],
        [2, 1],
        [2, -1],
    ], dtype=float)
    
    # Convert lattice coords to Cartesian positions
    e1 = np.array([3.0, 0.0])
    e2 = np.array([1.5, 2.598076211353316])  # 1.5*sqrt(3)
    
    base_positions = lattice_points @ np.array([e1, e2]).T
    
    # Center the configuration at origin
    centroid = np.mean(base_positions, axis=0)
    base_positions -= centroid
    
    # Now optimize: rotation angle theta and scale factor t
    # The transformed position is: R(theta) * t * base_position
    # We want to minimize the containing hexagon side length s.
    
    def objective(params):
        """
        params: [theta_rad, log_t]
        theta: rotation angle in radians
        log_t: log of scale factor (to keep t positive)
        """
        theta = params[0]
        t = np.exp(params[1])
        
        # Rotation matrix
        c, s_rot = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s_rot], [s_rot, c]])
        
        # Transform positions
        positions = (base_positions * t) @ R.T
        
        # Build inner array
        inner = np.zeros((12, 3))
        inner[:, 0] = positions[:, 0]
        inner[:, 1] = positions[:, 1]
        inner[:, 2] = theta * 180.0 / np.pi  # Angle in degrees
        
        # Compute minimum containing hexagon side
        s_val = _compute_min_s(inner, np.array([0.0, 0.0]), 0.0)
        
        return s_val
    
    # Initial guess: no rotation, scale = 1
    x0 = np.array([0.0, 0.0])  # theta=0, t=1
    
    # First, check the initial configuration
    inner_init = np.zeros((12, 3))
    inner_init[:, 0] = base_positions[:, 0]
    inner_init[:, 1] = base_positions[:, 1]
    inner_init[:, 2] = 0.0
    
    outer_init = np.array([0.0, 0.0, 0.0])
    s_init = _compute_min_s(inner_init, np.array([0.0, 0.0]), 0.0)
    
    valid, msg = _check_validity(inner_init, outer_init, s_init + 1e-4)
    if not valid:
        # Shouldn't happen with lattice spacing, but fallback
        print(f"Initial lattice config invalid: {msg}")
        # Use a larger margin
        s_init += 1.0
        valid, msg = _check_validity(inner_init, outer_init, s_init)
        if not valid:
            print(f"Still invalid: {msg}, using large s")
            return inner_init, outer_init, 20.0
    
    best_s = s_init
    best_inner = inner_init.copy()
    best_outer = outer_init.copy()
    print(f"Initial lattice: s = {best_s:.6f}, 1/s = {1/best_s:.6f}")
    
    # Optimize rotation and scale
    try:
        result = minimize(objective, x0, method='Nelder-Mead',
                         options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10,
                                  'adaptive': True})
        
        if result.fun < best_s:
            theta_opt = result.x[0]
            t_opt = np.exp(result.x[1])
            
            c, s_rot = np.cos(theta_opt), np.sin(theta_opt)
            R = np.array([[c, -s_rot], [s_rot, c]])
            positions = (base_positions * t_opt) @ R.T
            
            inner_opt = np.zeros((12, 3))
            inner_opt[:, 0] = positions[:, 0]
            inner_opt[:, 1] = positions[:, 1]
            inner_opt[:, 2] = theta_opt * 180.0 / np.pi
            
            s_opt = _compute_min_s(inner_opt, np.array([0.0, 0.0]), 0.0)
            outer_opt = np.array([0.0, 0.0, 0.0])
            
            valid, msg = _check_validity(inner_opt, outer_opt, s_opt + 1e-4)
            if valid:
                best_inner = inner_opt
                best_outer = outer_opt
                best_s = s_opt + 1e-4
                print(f"Optimized (theta={theta_opt:.4f}, t={t_opt:.4f}): s = {best_s:.6f}, 1/s = {1/best_s:.6f}")
            else:
                print(f"Optimized result invalid: {msg}")
        
        if time.time() > deadline:
            return best_inner, best_outer, best_s
        
        # Second round: tighter optimization
        if time.time() < deadline - 120:
            result2 = minimize(objective, result.x, method='Nelder-Mead',
                             options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12,
                                      'adaptive': True})
            
            if result2.fun < best_s:
                theta_opt = result2.x[0]
                t_opt = np.exp(result2.x[1])
                
                c, s_rot = np.cos(theta_opt), np.sin(theta_opt)
                R = np.array([[c, -s_rot], [s_rot, c]])
                positions = (base_positions * t_opt) @ R.T
                
                inner_opt = np.zeros((12, 3))
                inner_opt[:, 0] = positions[:, 0]
                inner_opt[:, 1] = positions[:, 1]
                inner_opt[:, 2] = theta_opt * 180.0 / np.pi
                
                s_opt = _compute_min_s(inner_opt, np.array([0.0, 0.0]), 0.0)
                outer_opt = np.array([0.0, 0.0, 0.0])
                
                valid, msg = _check_validity(inner_opt, outer_opt, s_opt + 1e-4)
                if valid:
                    best_inner = inner_opt
                    best_outer = outer_opt
                    best_s = s_opt + 1e-4
                    print(f"Further optimized: s = {best_s:.6f}, 1/s = {1/best_s:.6f}")
    
    except Exception as e:
        print(f"Optimization error: {e}")
    
    # Also try: optimize the outer hexagon angle as well
    # For a given inner configuration, the optimal outer hexagon angle may differ from 0
    try:
        if time.time() < deadline - 60:
            def objective_with_outer_angle(params):
                """
                params: [theta_inner_rad, log_t, outer_angle_rad]
                """
                theta = params[0]
                t = np.exp(params[1])
                outer_angle = params[2]
                
                c, s_rot = np.cos(theta), np.sin(theta)
                R = np.array([[c, -s_rot], [s_rot, c]])
                positions = (base_positions * t) @ R.T
                
                inner = np.zeros((12, 3))
                inner[:, 0] = positions[:, 0]
                inner[:, 1] = positions[:, 1]
                inner[:, 2] = theta * 180.0 / np.pi
                
                s_val = _compute_min_s(inner, np.array([0.0, 0.0]), outer_angle * 180.0 / np.pi)
                return s_val
            
            x0_3 = np.array([0.0, 0.0, 0.0])
            result3 = minimize(objective_with_outer_angle, x0_3, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10,
                                      'adaptive': True})
            
            if result3.fun < best_s:
                theta_opt = result3.x[0]
                t_opt = np.exp(result3.x[1])
                outer_angle_opt = result3.x[2]
                
                c, s_rot = np.cos(theta_opt), np.sin(theta_opt)
                R = np.array([[c, -s_rot], [s_rot, c]])
                positions = (base_positions * t_opt) @ R.T
                
                inner_opt = np.zeros((12, 3))
                inner_opt[:, 0] = positions[:, 0]
                inner_opt[:, 1] = positions[:, 1]
                inner_opt[:, 2] = theta_opt * 180.0 / np.pi
                
                s_opt = _compute_min_s(inner_opt, np.array([0.0, 0.0]), outer_angle_opt * 180.0 / np.pi)
                outer_opt = np.array([0.0, 0.0, outer_angle_opt * 180.0 / np.pi])
                
                valid, msg = _check_validity(inner_opt, outer_opt, s_opt + 1e-4)
                if valid:
                    best_inner = inner_opt
                    best_outer = outer_opt
                    best_s = s_opt + 1e-4
                    print(f"With outer angle: s = {best_s:.6f}, 1/s = {1/best_s:.6f}")
    except Exception as e:
        print(f"Outer angle optimization error: {e}")
    
    # Final validation
    valid, msg = _check_validity(best_inner, best_outer, best_s)
    if not valid:
        print(f"Final result invalid: {msg}, using fallback")
        return best_inner, best_outer, 20.0
    
    return best_inner, best_outer, best_s
