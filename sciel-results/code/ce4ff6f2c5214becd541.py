"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (evaluator limit 150s, keep 10s buffer)
    
    SQRT3 = math.sqrt(3)
    
    # Helper: Get vertices of a unit hexagon centered at (cx, cy) with rotation angle
    # Returns a (6, 2) NumPy array
    def get_hex_vertices(cx, cy, angle_deg):
        angles = np.radians(angle_deg) + np.arange(6) * (np.pi / 3.0)
        verts = np.column_stack((np.cos(angles), np.sin(angles)))
        return verts + np.array([cx, cy])

    # Helper: Get axes for SAT (normals to edges)
    # Returns a (6, 2) NumPy array
    def get_hex_axes(angle_deg):
        # Normal angle is offset by 30 degrees (pi/6) from vertex angle
        angles = np.radians(angle_deg) + np.pi/6 + np.arange(6) * (np.pi / 3.0)
        axes = np.column_stack((np.cos(angles), np.sin(angles)))
        return axes

    def project_poly(vertices, axes):
        """
        Project vertices (6,2) onto axes (6,2).
        Returns min_proj (6,) and max_proj (6,)
        """
        # (6,2) @ (2,6) -> (6,6)
        projs = vertices @ axes.T
        return projs.min(axis=0), projs.max(axis=0)

    def is_disjoint(hex1_verts, hex2_verts, axes1, axes2):
        """
        Check if two hexagons are disjoint using SAT.
        hex1_verts, hex2_verts: (6,2) arrays
        axes1, axes2: (6,2) arrays
        Returns True if disjoint.
        """
        # Check axes from hex1
        min1, max1 = project_poly(hex1_verts, axes1)
        min2, max2 = project_poly(hex2_verts, axes1)
        # If any axis separates them, they are disjoint
        if np.any(max1 < min2) or np.any(max2 < min1):
            return True
            
        # Check axes from hex2
        min1, max1 = project_poly(hex1_verts, axes2)
        min2, max2 = project_poly(hex2_verts, axes2)
        if np.any(max1 < min2) or np.any(max2 < min1):
            return True
            
        return False

    def calc_outer_side(hexagons, outer_angle_deg):
        """
        Calculate the minimum side length S of a regular hexagon centered at (0,0)
        with rotation outer_angle_deg that contains all given hexagons.
        hexagons: list of (6,2) arrays
        """
        base_angles = np.radians([30, 90, 150, 210, 270, 330])
        axis_angles = base_angles + np.radians(outer_angle_deg)
        axis_dirs = np.column_stack((np.cos(axis_angles), np.sin(axis_angles))) # (6,2)
        
        max_proj = 0.0
        for hex_verts in hexagons:
            # (6,2) @ (2,6) -> (6,6)
            projs = hex_verts @ axis_dirs.T
            # We need max absolute projection for each axis
            # Actually, for containment in a centered hexagon, we need max(|proj|)
            # Wait, the hexagon is centered at 0. The constraints are:
            # For each normal axis n_i: min_proj >= -S*sqrt(3)/2 and max_proj <= S*sqrt(3)/2
            # So S*sqrt(3)/2 >= max( |min_proj|, |max_proj| ) for all i
            # Let M = max( |min_proj|, |max_proj| ) over all vertices and axes
            # Then S >= 2*M / sqrt(3)
            
            abs_projs = np.abs(projs)
            max_abs = np.max(abs_projs)
            if max_abs > max_proj:
                max_proj = max_abs
                
        return 2.0 * max_proj / SQRT3

    # Helper to evaluate a configuration
    # p: array of shape (3N,)
    def evaluate(p, outer_angle_deg=0.0):
        centers = p[:2*N].reshape(N, 2)
        angles = p[2*N:]
        
        # Center the configuration
        cx_avg = np.mean(centers[:, 0])
        cy_avg = np.mean(centers[:, 1])
        shifted_centers = centers - np.array([cx_avg, cy_avg])
        
        hexagons = []
        hex_axes = []
        for i in range(N):
            cx, cy = shifted_centers[i]
            a = angles[i]
            hex_verts = get_hex_vertices(cx, cy, a)
            hexagons.append(hex_verts)
            hex_axes.append(get_hex_axes(a))
        
        # Check for overlaps
        overlap_count = 0
        for i in range(N):
            for j in range(i+1, N):
                if not is_disjoint(hexagons[i], hexagons[j], hex_axes[i], hex_axes[j]):
                    overlap_count += 1
                    # Early exit if too many overlaps? No, need total count for smooth penalty?
                    # Hard penalty is better for SA validity.
                    
        if overlap_count > 0:
            # Hard penalty
            return 1000.0 + overlap_count + calc_outer_side(hexagons, outer_angle_deg)
            
        return calc_outer_side(hexagons, outer_angle_deg)

    def axial_to_cartesian(q, r, spacing=1.0):
        x = SQRT3 * spacing * (q + r / 2.0)
        y = 1.5 * spacing * r
        return x, y

    # Define initial configurations
    coords1 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (2,-1), (2,1), (1,1)]
    coords2 = [(-1,-1),(0,-1),(1,-1),(-2,0),(-1,0),(0,0),(1,0),(2,0),(-1,1),(0,1),(1,1)]
    coords3 = [(0,0), (1,0), (2,0), (3,0), (4,0), (-1,0), (-2,0), (0,1), (1,1), (2,1), (3,1)]
    
    configs = [coords1, coords2, coords3]
    
    best_params = None
    best_score = float('inf')
    
    for config in configs:
        if time.time() > deadline - 30.0:
            break
            
        params = []
        for q, r in config:
            x, y = axial_to_cartesian(q, r, spacing=1.0)
            params.extend([x, y, 0.0])
        params = np.array(params)
        
        current_score = evaluate(params)
        local_best_params = params.copy()
        local_best_score = current_score
        
        T = 15.0
        T_min = 0.01
        cooling_rate = 0.995
        steps_per_temp = 50
        
        step = 0
        while T > T_min and time.time() < deadline - 15.0:
            new_params = params.copy()
            idx = random.randint(0, N-1)
            
            step_scale = 0.2 + 0.2 * (T / 15.0)
            dx = random.gauss(0, step_scale)
            dy = random.gauss(0, step_scale)
            da = random.gauss(0, 10.0)
            
            new_params[idx*3] += dx
            new_params[idx*3+1] += dy
            new_params[idx*3+2] += da
            
            new_score = evaluate(new_params)
            
            delta_e = new_score - current_score
            if delta_e < 0 or random.random() < math.exp(-delta_e / T):
                params = new_params
                current_score = new_score
                if current_score < local_best_score:
                    local_best_score = current_score
                    local_best_params = params.copy()
            
            step += 1
            if step % steps_per_temp == 0:
                T *= cooling_rate
        
        if local_best_score < best_score:
            best_score = local_best_score
            best_params = local_best_params.copy()
            
    if best_params is None:
        params = []
        for q, r in coords1:
            x, y = axial_to_cartesian(q, r, spacing=1.0)
            params.extend([x, y, 0.0])
        best_params = np.array(params)

    # Final Polish:
    # 1. Optimize outer angle for the current best positions
    # 2. Use Nelder-Mead to refine positions, with objective that includes optimal outer angle
    
    def get_best_outer_angle(params):
        centers = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        
        cx_avg = np.mean(centers[:, 0])
        cy_avg = np.mean(centers[:, 1])
        shifted_centers = centers - np.array([cx_avg, cy_avg])
        
        hexagons = []
        for i in range(N):
            cx, cy = shifted_centers[i]
            a = angles[i]
            hex_verts = get_hex_vertices(cx, cy, a)
            hexagons.append(hex_verts)
            
        def side_len(oa):
            return calc_outer_side(hexagons, oa)
            
        res = minimize_scalar(side_len, bounds=(0, 60), method='bounded')
        return res.fun, res.x

    def objective_nm(p):
        # Check validity first
        centers = p[:2*N].reshape(N, 2)
        angles = p[2*N:]
        cx_avg = np.mean(centers[:, 0])
        cy_avg = np.mean(centers[:, 1])
        shifted_centers = centers - np.array([cx_avg, cy_avg])
        
        hexagons = []
        hex_axes = []
        for i in range(N):
            cx, cy = shifted_centers[i]
            a = angles[i]
            hex_verts = get_hex_vertices(cx, cy, a)
            hexagons.append(hex_verts)
            hex_axes.append(get_hex_axes(a))
            
        for i in range(N):
            for j in range(i+1, N):
                if not is_disjoint(hexagons[i], hexagons[j], hex_axes[i], hex_axes[j]):
                    return 1000.0 + calc_outer_side(hexagons, 0.0)
                    
        # If valid, find best outer angle
        best_side, _ = get_best_outer_angle(p)
        return best_side

    try:
        # Initial guess for NM is the SA result
        # We might want to re-center the SA result to 0,0 explicitly before NM
        # But evaluate does it internally.
        
        res = minimize(objective_nm, best_params, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
        
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
    except Exception:
        pass

    # Final extraction
    final_params = best_params
    centers = final_params[:2*N].reshape(N, 2)
    angles = final_params[2*N:]
    
    cx_avg = np.mean(centers[:, 0])
    cy_avg = np.mean(centers[:, 1])
    shifted_centers = centers - np.array([cx_avg, cy_avg])
    
    # Get final best outer angle
    hexagons = []
    for i in range(N):
        cx, cy = shifted_centers[i]
        a = angles[i]
        hex_verts = get_hex_vertices(cx, cy, a)
        hexagons.append(hex_verts)
        
    def side_len(oa):
        return calc_outer_side(hexagons, oa)
        
    res_ao = minimize_scalar(side_len, bounds=(0, 60), method='bounded')
    best_side = res_ao.fun
    best_oa = res_ao.x

    inner_list = []
    for i in range(N):
        inner_list.append([shifted_centers[i][0], shifted_centers[i][1], angles[i]])
        
    return (inner_list, (0.0, 0.0), best_side, best_oa)

# EVOLVE-BLOCK-END
