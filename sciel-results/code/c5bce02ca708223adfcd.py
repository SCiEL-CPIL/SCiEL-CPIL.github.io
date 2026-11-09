# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = start_time + 590  # safety margin before 600s kill

    # Precompute unit hexagon vertices (angle 0)
    unit_angles = np.deg2rad(60 * np.arange(6))
    unit_verts = np.column_stack([np.cos(unit_angles), np.sin(unit_angles)])

    # Precompute edge normals for unit hexagon at angle 0
    # Edge i is from vertex i to vertex i+1. Normal is perpendicular to edge, pointing outward.
    # For a regular hexagon centered at origin, the outward normals are at angles 30, 90, 150, 210, 270, 330.
    normal_angles_0 = np.deg2rad(30 + 60 * np.arange(6))
    unit_normals_0 = np.column_stack([np.cos(normal_angles_0), np.sin(normal_angles_0)])

    def hex_vertices(cx, cy, s, a_deg):
        a = np.deg2rad(a_deg)
        angles = a + unit_angles
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

    def get_rotated_normals(a_deg):
        a = np.deg2rad(a_deg)
        ca, sa = np.cos(a), np.sin(a)
        R = np.array([[ca, -sa], [sa, ca]])
        return unit_normals_0 @ R.T

    def hexagons_separated(v1, n1, v2, n2):
        """
        Check if two hexagons are separated.
        v1, v2: (6,2) vertex arrays.
        n1, n2: (6,2) outward normal arrays (unit vectors).
        Returns True if separated by > 1e-6 on at least one of the 12 normals.
        """
        # Check 6 normals of hex 1
        for i in range(6):
            d = n1[i]
            proj1 = v1 @ d
            proj2 = v2 @ d
            if proj1.max() < proj2.min() - 1e-6 or proj2.max() < proj1.min() - 1e-6:
                return True
        # Check 6 normals of hex 2
        for i in range(6):
            d = n2[i]
            proj1 = v1 @ d
            proj2 = v2 @ d
            if proj1.max() < proj2.min() - 1e-6 or proj2.max() < proj1.min() - 1e-6:
                return True
        return False

    def enclosing_hex_side(inner_data, outer_center, outer_angle):
        """
        Compute the minimum side length of a regular hexagon (center, angle)
        that encloses all vertices of the inner hexagons.
        inner_data: (11, 3) (x, y, angle_deg)
        """
        ocx, ocy = outer_center
        oa = np.deg2rad(outer_angle)
        
        all_verts = []
        for i in range(11):
            cx, cy, a_deg = inner_data[i]
            v = hex_vertices(cx, cy, 1.0, a_deg)
            all_verts.append(v)
        all_verts = np.vstack(all_verts)
        
        # Shift by outer center
        all_verts_shifted = all_verts - np.array([ocx, ocy])
        
        min_s = 0.0
        for i in range(6):
            a_i = oa + np.deg2rad(60 * i)
            a_ip1 = oa + np.deg2rad(60 * (i + 1))
            e_unit = np.array([np.cos(a_ip1) - np.cos(a_i), np.sin(a_ip1) - np.sin(a_i)])
            # cross(e_unit, p_j) for all j
            crosses = all_verts_shifted[:, 0] * e_unit[1] - all_verts_shifted[:, 1] * e_unit[0]
            min_cross = np.min(crosses)
            required_s = -2 * min_cross / np.sqrt(3)
            if required_s > min_s:
                min_s = required_s
        
        return max(min_s, 0.0)

    def objective(params):
        if time.time() > deadline:
            return 1e10
        
        # Parse parameters: 11*(x,y,angle) + outer(x,y,angle) = 36 params
        inner_params = params[:33].reshape(11, 3)
        outer_params = params[33:36]
        
        inner_positions = inner_params[:, :2]
        inner_angles = inner_params[:, 2]
        outer_center = outer_params[:2]
        outer_angle = outer_params[2]
        
        # Compute vertices and normals of all inner hexagons
        inner_verts = []
        inner_norms = []
        for i in range(11):
            cx, cy = inner_positions[i]
            a_deg = inner_angles[i]
            v = hex_vertices(cx, cy, 1.0, a_deg)
            n = get_rotated_normals(a_deg)
            inner_verts.append(v)
            inner_norms.append(n)
        
        # Check separation between all pairs
        penalty = 0.0
        for i in range(11):
            for j in range(i+1, 11):
                if not hexagons_separated(inner_verts[i], inner_norms[i], inner_verts[j], inner_norms[j]):
                    # Compute minimum violation (overlap)
                    min_gap = np.inf
                    # Check all 12 normals
                    for k in range(6):
                        d = inner_norms[i][k]
                        proj_i = inner_verts[i] @ d
                        proj_j = inner_verts[j] @ d
                        gap = max(proj_i.min() - proj_j.max(), proj_j.min() - proj_i.max())
                        if gap < min_gap:
                            min_gap = gap
                    for k in range(6):
                        d = inner_norms[j][k]
                        proj_i = inner_verts[i] @ d
                        proj_j = inner_verts[j] @ d
                        gap = max(proj_i.min() - proj_j.max(), proj_j.min() - proj_i.max())
                        if gap < min_gap:
                            min_gap = gap
                    # Penalty for overlap
                    if min_gap < 1e-6:
                        penalty += (1e-6 - min_gap) ** 2
        
        # Compute enclosing hexagon side
        s = enclosing_hex_side(inner_params, outer_center, outer_angle)
        
        # Also ensure side >= sqrt(11)
        if s < np.sqrt(11):
            s = np.sqrt(11)
        
        return s + 1000 * penalty

    # Initial guess: "Flower + 4" configuration
    # This is a valid packing of 11 unit hexagons.
    sqrt3 = np.sqrt(3)
    centers = [
        [0, 0],
        [2, 0],
        [-2, 0],
        [1, sqrt3],
        [-1, sqrt3],
        [1, -sqrt3],
        [-1, -sqrt3],
        [4, 0],
        [-4, 0],
        [3, sqrt3],
        [3, -sqrt3]
    ]
    
    # Flatten to 1D array for minimize
    x0 = np.zeros(36)
    for i, c in enumerate(centers):
        x0[3*i] = c[0]
        x0[3*i+1] = c[1]
        x0[3*i+2] = 0.0
    x0[33] = 0.0 # outer center x
    x0[34] = 0.0 # outer center y
    x0[35] = 0.0 # outer angle
    
    best_params = x0.copy()
    best_val = objective(x0)
    
    # Optimize using Powell method which is robust for this type of problem
    try:
        result = minimize(objective, x0, method='Powell', 
                         options={'maxiter': 500, 'ftol': 1e-10, 'xtol': 1e-10})
        
        if result.fun < best_val:
            best_val = result.fun
            best_params = result.x
    except Exception:
        pass

    # If time remains, try a second run with perturbation
    if time.time() < deadline - 100:
        x0_pert = x0 + np.random.normal(0, 0.05, x0.shape)
        try:
            result2 = minimize(objective, x0_pert, method='Powell', 
                             options={'maxiter': 300, 'ftol': 1e-10, 'xtol': 1e-10})
            if result2.fun < best_val:
                best_val = result2.fun
                best_params = result2.x
        except Exception:
            pass

    # Extract best solution
    inner_params = best_params[:33].reshape(11, 3)
    outer_params = best_params[33:36]
    
    inner_positions = inner_params[:, :2]
    inner_angles = inner_params[:, 2]
    outer_center = outer_params[:2]
    outer_angle = outer_params[2]
    
    # Compute actual enclosing side
    s = enclosing_hex_side(inner_params, outer_center, outer_angle)
    s = max(s, np.sqrt(11))
    
    inner_hex_data = np.column_stack([inner_positions, inner_angles])
    outer_hex_data = np.array([outer_center[0], outer_center[1], outer_angle])
    
    return inner_hex_data, outer_hex_data, float(s)

# EVOLVE-BLOCK-END
