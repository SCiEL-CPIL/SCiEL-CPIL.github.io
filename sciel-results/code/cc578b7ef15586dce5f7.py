# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = start_time + 590  # safety margin
    
    SQRT3 = np.sqrt(3.0)
    
    # Precompute angle offsets for hex vertices
    hex_angle_offsets = np.radians(60 * np.arange(6))
    
    # Hexagon vertices function (vectorized)
    def hex_vertices(cx, cy, s, a_rad):
        angles = a_rad + hex_angle_offsets
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])
    
    # Compute all 11 hex vertices at once
    def all_hex_vertices(inner_positions, inner_angles):
        n = len(inner_positions)
        verts = np.zeros((n, 6, 2))
        for i in range(n):
            verts[i] = hex_vertices(inner_positions[i, 0], inner_positions[i, 1], 1.0, np.radians(inner_angles[i]))
        return verts
    
    # Check if two hexagons are separated (no intersection)
    # Returns True if separated by > 1e-6 on at least one edge normal
    def hexagons_separated(v1, v2):
        for i in range(6):
            e1 = v1[(i+1) % 6] - v1[i]
            n1 = np.array([-e1[1], e1[0]])
            norm = np.linalg.norm(n1)
            if norm > 1e-15:
                n1 = n1 / norm
            proj1 = v1 @ n1
            proj2 = v2 @ n1
            if proj1.max() < proj2.min() - 1e-6 or proj2.max() < proj1.min() - 1e-6:
                return True
        for i in range(6):
            e2 = v2[(i+1) % 6] - v2[i]
            n2 = np.array([-e2[1], e2[0]])
            norm = np.linalg.norm(n2)
            if norm > 1e-15:
                n2 = n2 / norm
            proj1 = v1 @ n2
            proj2 = v2 @ n2
            if proj1.max() < proj2.min() - 1e-6 or proj2.max() < proj1.min() - 1e-6:
                return True
        return False
    
    # Compute minimum side length of enclosing hexagon
    def enclosing_hex_side(inner_positions, inner_angles, outer_center, outer_angle_deg):
        ocx, ocy = outer_center
        oa = np.radians(outer_angle_deg)
        
        all_verts = []
        for i in range(len(inner_positions)):
            cx, cy = inner_positions[i]
            a_rad = np.radians(inner_angles[i])
            v = hex_vertices(cx, cy, 1.0, a_rad)
            all_verts.append(v)
        all_verts = np.vstack(all_verts)
        
        all_verts_shifted = all_verts - np.array([ocx, ocy])
        
        min_s = 0.0
        for i in range(6):
            a_i = oa + np.radians(60 * i)
            a_ip1 = oa + np.radians(60 * (i + 1))
            e_unit = np.array([np.cos(a_ip1) - np.cos(a_i), np.sin(a_ip1) - np.sin(a_i)])
            crosses = all_verts_shifted[:, 0] * e_unit[1] - all_verts_shifted[:, 1] * e_unit[0]
            min_cross = np.min(crosses)
            required_s = -2 * min_cross / SQRT3
            if required_s > min_s:
                min_s = required_s
        
        return max(min_s, 0.0)
    
    # Objective function: minimize outer hexagon side length with penalty for overlaps
    def objective(params):
        if time.time() > deadline:
            return 1e10
        
        if not np.all(np.isfinite(params)):
            return 1e10
        
        # Parse parameters: 11*(x,y,angle) + outer(x,y,angle) = 36 params
        inner_params = params[:33].reshape(11, 3)
        outer_params = params[33:36]
        
        inner_positions = inner_params[:, :2]
        inner_angles = inner_params[:, 2]
        outer_center = outer_params[:2]
        outer_angle = outer_params[2]
        
        # Compute vertices of all inner hexagons
        inner_verts = all_hex_vertices(inner_positions, inner_angles)
        
        # Check separation between all pairs
        penalty = 0.0
        for i in range(11):
            for j in range(i+1, 11):
                if not hexagons_separated(inner_verts[i], inner_verts[j]):
                    # Compute minimum overlap
                    min_gap = np.inf
                    for vi in [inner_verts[i], inner_verts[j]]:
                        for k in range(6):
                            e = vi[(k+1)%6] - vi[k]
                            n = np.array([-e[1], e[0]])
                            norm = np.linalg.norm(n)
                            if norm > 1e-15:
                                n = n / norm
                            proj_i = inner_verts[i] @ n
                            proj_j = inner_verts[j] @ n
                            gap = max(proj_i.min() - proj_j.max(), proj_j.min() - proj_i.max())
                            if gap < min_gap:
                                min_gap = gap
                    # Penalty for overlap
                    if min_gap < 1e-6:
                        penalty += (1e-6 - min_gap) ** 2
        
        # Compute enclosing hexagon side
        s = enclosing_hex_side(inner_positions, inner_angles, outer_center, outer_angle)
        
        # Also ensure side >= sqrt(11)
        if s < np.sqrt(11):
            s = np.sqrt(11)
        
        return s + 1000 * penalty
    
    # Initial guess: "Flower + 4" configuration
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
    
    # Strategy 1: Multi-start Powell optimization with different initial conditions
    np.random.seed(42)
    num_starts = 10
    for start_idx in range(num_starts):
        if time.time() > deadline - 70:
            break
        
        if start_idx == 0:
            x_start = x0.copy()
        else:
            # Perturb the initial guess
            x_start = x0.copy()
            # Add random noise to positions
            noise_scale = 0.4 * (start_idx + 1)
            for i in range(11):
                x_start[3*i] += np.random.randn() * noise_scale
                x_start[3*i+1] += np.random.randn() * noise_scale
                x_start[3*i+2] += np.random.randn() * 25  # angle in degrees
            # Keep outer center near origin
            x_start[33] = np.random.randn() * 0.4
            x_start[34] = np.random.randn() * 0.4
            x_start[35] = np.random.randn() * 25
        
        try:
            result = minimize(objective, x_start, method='Powell', 
                             options={'maxiter': 250, 'ftol': 1e-12, 'xtol': 1e-12})
            
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Strategy 2: Use Nelder-Mead for local refinement from best found
    if time.time() < deadline - 40:
        try:
            result2 = minimize(objective, best_params, method='Nelder-Mead',
                              options={'maxiter': 800, 'xatol': 1e-13, 'fatol': 1e-13})
            
            if result2.fun < best_val:
                best_val = result2.fun
                best_params = result2.x.copy()
        except Exception:
            pass
    
    # Strategy 3: Final high-precision polish with COBYLA
    if time.time() < deadline - 15:
        try:
            result3 = minimize(objective, best_params, method='COBYLA',
                              options={'maxiter': 3000, 'rhobeg': 1e-9, 'tol': 1e-15})
            
            if result3.fun < best_val:
                best_val = result3.fun
                best_params = result3.x.copy()
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
    s = enclosing_hex_side(inner_positions, inner_angles, outer_center, outer_angle)
    s = max(s, np.sqrt(11))
    
    inner_hex_data = np.column_stack([inner_positions, inner_angles])
    outer_hex_data = np.array([outer_center[0], outer_center[1], outer_angle])
    
    return inner_hex_data, outer_hex_data, float(s)

# EVOLVE-BLOCK-END
