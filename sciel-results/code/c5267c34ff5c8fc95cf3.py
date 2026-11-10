# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, s, a_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy), side s, angle a_deg."""
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _separation_distance(v1, v2):
    """
    Compute the minimum distance between two convex polygons (hexagons).
    Uses separating axis theorem: for each edge normal of both polygons,
    project both polygons and find the gap.
    Returns the minimum gap (positive if separated, 0 if touching/overlapping).
    """
    min_gap = np.inf
    for v in [v1, v2]:
        for k in range(6):
            e = v[(k + 1) % 6] - v[k]
            n = np.array([-e[1], e[0]])
            norm = np.linalg.norm(n)
            if norm < 1e-12:
                continue
            n = n / norm
            proj1 = v1 @ n
            proj2 = v2 @ n
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            # Gap is max(0, min2 - max1, min1 - max2)
            gap = max(0.0, min2 - max1, min1 - max2)
            min_gap = min(min_gap, gap)
    return min_gap

def _min_outer_side_for_inner(inner_data, outer_center, outer_angle_deg):
    """
    Compute the minimum outer hex side length such that all inner hex vertices
    are inside the outer hexagon centered at outer_center with given angle.
    """
    ox, oy, oa = outer_center
    oa_rad = np.deg2rad(oa)
    
    # Compute all inner vertices
    all_inner_verts = []
    for row in inner_data:
        x, y, ang = row
        verts = _hex_vertices(x, y, 1.0, ang)
        all_inner_verts.append(verts)
    all_inner_verts = np.vstack(all_inner_verts)
    
    # Shift by outer center
    shifted = all_inner_verts - np.array([ox, oy])
    
    # Inward normal directions for outer hex with angle a
    normal_dirs = []
    for i in range(6):
        theta_i = oa_rad + np.deg2rad(60.0 * i)
        theta_next = oa_rad + np.deg2rad(60.0 * (i + 1))
        v_i = np.array([np.cos(theta_i), np.sin(theta_i)])
        v_next = np.array([np.cos(theta_next), np.sin(theta_next)])
        edge = v_next - v_i
        # Inward normal: for CCW polygon, interior is to the left of edge direction
        # Left of (dx,dy) is (-dy, dx)
        n = np.array([-edge[1], edge[0]])
        n = n / np.linalg.norm(n)
        normal_dirs.append(n)
    normal_dirs = np.array(normal_dirs)  # shape (6, 2)
    
    # For each shifted inner vertex, compute max dot product with each normal
    # We need: for each edge i, max over p of dot(n_i, p) <= apothem = s * sqrt(3)/2
    # So s >= (2/sqrt(3)) * max_i (max_p dot(n_i, p))
    
    # Compute max dot for each normal direction
    max_dots = np.max(shifted @ normal_dirs.T, axis=0)  # shape (6,)
    max_dot = np.max(max_dots)
    
    if max_dot <= 0:
        return 0.0
    
    s_min = (2.0 / np.sqrt(3.0)) * max_dot
    return s_min

def _check_validity(inner_data, outer_data, side):
    """Check if the packing is valid."""
    if not np.all(np.isfinite(inner_data)) or not np.all(np.isfinite(outer_data)):
        return False
    if side < np.sqrt(11.0) - 1e-9:
        return False
    
    ox, oy, oa = outer_data
    
    # Check all inner vertices are inside outer hex
    all_inner_verts = []
    for row in inner_data:
        x, y, ang = row
        verts = _hex_vertices(x, y, 1.0, ang)
        all_inner_verts.append(verts)
    all_inner_verts = np.vstack(all_inner_verts)
    
    # Outer hex vertices
    outer_verts = _hex_vertices(ox, oy, side, oa)
    
    # For each outer edge, check cross product constraint
    for i in range(6):
        v1 = outer_verts[i]
        v2 = outer_verts[(i + 1) % 6]
        edge = v2 - v1
        # cross(edge, p - v1) >= -1e-6 for all p
        diffs = all_inner_verts - v1
        crosses = diffs[:, 0] * edge[1] - diffs[:, 1] * edge[0]
        if np.min(crosses) < -1e-6:
            return False
    
    # Check pairwise separation of inner hexagons
    for i in range(11):
        for j in range(i + 1, 11):
            vi = _hex_vertices(*inner_data[i])
            vj = _hex_vertices(*inner_data[j])
            
            # Get all 12 normal directions
            normals = []
            for v in [vi, vj]:
                for k in range(6):
                    e = v[(k + 1) % 6] - v[k]
                    n = np.array([-e[1], e[0]])
                    norm = np.linalg.norm(n)
                    if norm > 1e-12:
                        normals.append(n / norm)
            
            separated = False
            for n in normals:
                proj_i = vi @ n
                proj_j = vj @ n
                min_i, max_i = proj_i.min(), proj_i.max()
                min_j, max_j = proj_j.min(), proj_j.max()
                # Intervals are apart if max_i < min_j or max_j < min_i
                if max_i < min_j - 1e-6 or max_j < min_i - 1e-6:
                    separated = True
                    break
            
            if not separated:
                return False
    
    # Check edge lengths of shifted inner hexagons
    for row in inner_data:
        x, y, ang = row
        verts = _hex_vertices(x, y, 1.0, ang)
        shifted = verts - np.array([ox, oy])
        for k in range(6):
            e = shifted[(k + 1) % 6] - shifted[k]
            length = np.linalg.norm(e)
            if abs(length - 1.0) > 1e-8:
                return False
    
    return True

def _objective(params, inner_data_template, outer_angle_deg):
    """
    Objective function: outer side length + penalty for violations.
    params: flattened array of [x1,y1,ang1, x2,y2,ang2, ..., x11,y11,ang11, ox, oy, s]
    """
    n = 11
    # Unpack
    inner = params[:n*3].reshape(n, 3)
    outer = params[n*3:n*3+3]
    
    # Compute outer side
    s = _min_outer_side_for_inner(inner, outer[:2], outer_angle_deg)
    
    # Penalty for inner hexagons not being separated
    penalty = 0.0
    for i in range(n):
        for j in range(i+1, n):
            vi = _hex_vertices(*inner[i])
            vj = _hex_vertices(*inner[j])
            gap = _separation_distance(vi, vj)
            if gap < 1e-6:
                penalty += (1e-6 - gap) ** 2 * 1000
    
    # Penalty if inner hexagons are not inside outer (should be handled by s computation,
    # but add a small penalty if the computed s is too small)
    if s < np.sqrt(11.0):
        penalty += (np.sqrt(11.0) - s) ** 2 * 100
    
    return s + penalty

def _initial_guess_lattice():
    """
    Generate initial guess based on hexagonal lattice.
    7 hexagons in a flower + 4 more.
    """
    sq3 = np.sqrt(3.0)
    
    # Hexagonal lattice positions for unit hexagons (center-to-center distance = 2*sqrt(3)/2 = sqrt(3) for touching)
    # For unit side hexagons, the distance between centers of touching hexagons is 2 * apothem = sqrt(3)
    
    positions = [
        [0.0, 0.0],
        [0.0, sq3],
        [1.5, sq3/2],
        [1.5, -sq3/2],
        [0.0, -sq3],
        [-1.5, -sq3/2],
        [-1.5, sq3/2],
        # Add 4 more
        [0.0, 2.0 * sq3],
        [0.0, -2.0 * sq3],
        [3.0, sq3],
        [3.0, -sq3],
    ]
    
    inner_data = np.array([[x, y, 0.0] for x, y in positions])
    outer_data = np.array([0.0, 0.0, 0.0])
    
    return inner_data, outer_data

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin (630s limit, keep 50s buffer)
    
    best_result = None
    best_score = np.inf
    
    # Try multiple outer angles
    outer_angles = [0.0, 30.0, 60.0, 15.0, 45.0, 75.0, 105.0, 135.0]
    
    for outer_angle in outer_angles:
        if time.time() > deadline:
            break
        
        # Initial guess
        inner_data, outer_data = _initial_guess_lattice()
        outer_data[2] = outer_angle
        
        # Compute initial side
        s_init = _min_outer_side_for_inner(inner_data, outer_data[:2], outer_angle)
        
        # Use local optimization to adjust inner positions
        # We'll optimize the inner hexagon positions to minimize the outer side
        # while maintaining separation
        
        def objective_local(params):
            """
            params: 22 values (11 x, 11 y) - we fix angles to 0 for simplicity
            Returns: outer side length + penalty for violations
            """
            inner_x = params[:11]
            inner_y = params[11:22]
            
            inner = np.column_stack([inner_x, inner_y, np.zeros(11)])
            
            # Compute outer side
            s = _min_outer_side_for_inner(inner, np.array([0.0, 0.0]), outer_angle)
            
            # Penalty for inner hexagons not being separated
            penalty = 0.0
            for i in range(11):
                for j in range(i+1, 11):
                    vi = _hex_vertices(*inner[i])
                    vj = _hex_vertices(*inner[j])
                    gap = _separation_distance(vi, vj)
                    if gap < 1e-6:
                        penalty += (1e-6 - gap) ** 2 * 1000
            
            # Also check that inner hexagons are not too far apart (to keep them compact)
            # This is a soft constraint to prevent degenerate solutions
            center_x = np.mean(inner_x)
            center_y = np.mean(inner_y)
            spread = np.mean(np.sqrt((inner_x - center_x)**2 + (inner_y - center_y)**2))
            if spread > 5.0:
                penalty += (spread - 5.0) ** 2 * 100
            
            return s + penalty
        
        # Initial parameters
        x0 = np.concatenate([inner_data[:, 0], inner_data[:, 1]])
        
        # Run optimization
        try:
            result = minimize(objective_local, x0, method='Nelder-Mead',
                            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            
            if result.success or result.fun < s_init:
                best_x = result.x
                inner_x = best_x[:11]
                inner_y = best_x[11:22]
                inner_opt = np.column_stack([inner_x, inner_y, np.zeros(11)])
                
                s_opt = _min_outer_side_for_inner(inner_opt, np.array([0.0, 0.0]), outer_angle)
                
                # Add small margin for numerical safety
                s_opt += 1e-6
                
                outer_opt = np.array([0.0, 0.0, outer_angle])
                
                # Verify validity
                if _check_validity(inner_opt, outer_opt, s_opt):
                    if s_opt < best_score:
                        best_score = s_opt
                        best_result = (inner_opt, outer_opt, s_opt)
        except Exception as e:
            pass
    
    # If no valid result from optimization, use the initial lattice with computed side
    if best_result is None:
        inner_data, outer_data = _initial_guess_lattice()
        for outer_angle in [0.0, 30.0, 60.0]:
            outer_data[2] = outer_angle
            s = _min_outer_side_for_inner(inner_data, outer_data[:2], outer_angle)
            s += 1e-6
            if _check_validity(inner_data, outer_data, s):
                return inner_data, outer_data, s
    
    return best_result

# EVOLVE-BLOCK-END
