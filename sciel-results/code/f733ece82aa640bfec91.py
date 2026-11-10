import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 590

    sqrt3 = np.sqrt(3)

    # Define the 11 hexagon centers in a lattice configuration
    # Using axial coordinates (q, r) for a flat-top hexagon grid
    # Center (0,0)
    # Ring 1: 6 hexagons
    # Ring 2: 4 hexagons chosen to minimize bounding box
    # A good compact shape is the "hexagon of side 2" minus 8 corners?
    # Let's use the set:
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # (2,0), (0,2), (-2,0), (0,-2) -> This is the "plus" shape.
    # Let's try another set:
    # (2,-1), (1,1), (-1,2), (-2,1) -> This is a "triangle" shape?
    
    # Let's stick to the "plus" shape first as it is symmetric.
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (0, 2), (-2, 0), (0, -2)
    ]

    def axial_to_cartesian(q, r):
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        return x, y

    initial_centers = np.array([axial_to_cartesian(q, r) for q, r in axial_coords])
    # initial_centers shape: (11, 2)

    def get_hex_vertices(cx, cy, angle_deg):
        a = np.deg2rad(angle_deg)
        angles = a + np.deg2rad(np.arange(0, 360, 60))
        return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

    def calc_S(centers, outer_angle_deg):
        """
        Calculate minimum side length S of outer hexagon (center 0,0) to contain all hex vertices.
        centers: (11, 2)
        outer_angle_deg: angle of outer hexagon
        """
        # Vertices of all inner hexagons (assuming inner angle 0)
        all_verts = np.zeros((11, 6, 2))
        for i in range(11):
            all_verts[i] = get_hex_vertices(centers[i, 0], centers[i, 1], 0)
        
        flat_verts = all_verts.reshape(-1, 2)
        
        # Normals of outer hexagon
        a = np.deg2rad(outer_angle_deg)
        normal_angles = a + np.deg2rad(np.arange(30, 360, 60))
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        # Project vertices onto normals
        projections = flat_verts @ normals.T
        max_proj = np.max(projections, axis=1)
        global_max = np.max(max_proj)
        
        # Side length S from max projection
        # Distance from center to side is S * sqrt(3) / 2
        # So S = 2 * max_proj / sqrt(3)
        return 2.0 * global_max / sqrt3

    def overlap_penalty(centers):
        """
        Penalty for overlapping inner hexagons.
        Min distance between centers for non-overlap is sqrt(3).
        """
        penalty = 0.0
        for i in range(11):
            for j in range(i + 1, 11):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                d = np.sqrt(dx * dx + dy * dy)
                if d < sqrt3:
                    penalty += (sqrt3 - d) ** 2
        return penalty

    def objective(params):
        if time.time() > deadline:
            return 1e10
        
        # params: [dx1, dy1, ..., dx11, dy11, theta]
        # 22 shifts + 1 angle = 23 params
        shifts = params[:22].reshape(11, 2)
        theta = params[22]
        
        centers = initial_centers + shifts
        
        S = calc_S(centers, theta)
        pen = overlap_penalty(centers)
        
        return S + 100.0 * pen

    def exact_check(centers, theta):
        """
        Perform exact validation.
        Returns (is_valid, S, min_gap)
        """
        S = calc_S(centers, theta)
        
        # Check separation with SAT
        min_gap = np.inf
        for i in range(11):
            v1 = get_hex_vertices(centers[i, 0], centers[i, 1], 0)
            for j in range(i + 1, 11):
                v2 = get_hex_vertices(centers[j, 0], centers[j, 1], 0)
                
                # SAT check
                edges1 = np.roll(v1, -1, axis=0) - v1
                normals1 = np.column_stack([edges1[:, 1], -edges1[:, 0]])
                norms1 = np.linalg.norm(normals1, axis=1, keepdims=True)
                norms1[norms1 < 1e-12] = 1.0
                normals1 /= norms1
                
                edges2 = np.roll(v2, -1, axis=0) - v2
                normals2 = np.column_stack([edges2[:, 1], -edges2[:, 0]])
                norms2 = np.linalg.norm(normals2, axis=1, keepdims=True)
                norms2[norms2 < 1e-12] = 1.0
                normals2 /= norms2
                
                all_normals = np.vstack([normals1, normals2])
                
                proj1 = v1 @ all_normals.T
                proj2 = v2 @ all_normals.T
                
                min1 = np.min(proj1, axis=1)
                max1 = np.max(proj1, axis=1)
                min2 = np.min(proj2, axis=1)
                max2 = np.max(proj2, axis=1)
                
                gaps = np.maximum(min2 - max1, min1 - max2)
                gap = np.min(gaps)
                if gap < min_gap:
                    min_gap = gap
        
        if min_gap < 1e-6:
            return False, S, min_gap
            
        # Check edge lengths (should be 1)
        for i in range(11):
            for j in range(6):
                p1 = get_hex_vertices(centers[i, 0], centers[i, 1], 0)[j]
                p2 = get_hex_vertices(centers[i, 0], centers[i, 1], 0)[(j + 1) % 6]
                if abs(np.linalg.norm(p2 - p1) - 1.0) > 1e-8:
                    return False, S, min_gap
                    
        # Check containment
        outer_verts = get_hex_vertices(0, 0, theta) * S
        for i in range(11):
            verts_i = get_hex_vertices(centers[i, 0], centers[i, 1], 0)
            for p in verts_i:
                for k in range(6):
                    p1 = outer_verts[k]
                    p2 = outer_verts[(k + 1) % 6]
                    cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                    if cross < -1e-6:
                        return False, S, min_gap

        return True, S, min_gap

    # Initial guess for parameters
    # Small random shifts and angle 0
    np.random.seed(42)
    initial_params = np.zeros(23)
    initial_params[:22] = np.random.uniform(-0.1, 0.1, 22)
    initial_params[22] = 0.0

    best_S = 1e10
    best_params = None
    best_valid = False

    # Run optimization
    try:
        res = minimize(objective, initial_params, method='Nelder-Mead', 
                       options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
        
        if res.fun < best_S:
            best_S = res.fun
            best_params = res.x.copy()
            
        # Check validity
        shifts = best_params[:22].reshape(11, 2)
        theta = best_params[22]
        centers = initial_centers + shifts
        valid, S_val, gap_val = exact_check(centers, theta)
        
        if valid:
            best_S = S_val
            best_valid = True
            
    except Exception as e:
        pass

    # If not valid, try a second run with different initial guess
    if not best_valid:
        np.random.seed(123)
        initial_params2 = np.zeros(23)
        initial_params2[:22] = np.random.uniform(-0.2, 0.2, 22)
        initial_params2[22] = np.deg2rad(30) # Try rotating outer hex
        
        try:
            res2 = minimize(objective, initial_params2, method='Nelder-Mead', 
                            options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
            
            if res2.fun < best_S:
                best_S = res2.fun
                best_params = res2.x.copy()
                
            shifts = best_params[:22].reshape(11, 2)
            theta = best_params[22]
            centers = initial_centers + shifts
            valid, S_val, gap_val = exact_check(centers, theta)
            
            if valid:
                best_S = S_val
                best_valid = True
                
        except Exception as e:
            pass

    # Fallback to initial lattice if optimization failed
    if not best_valid:
        centers = initial_centers
        theta = 0.0
        valid, S_val, gap_val = exact_check(centers, theta)
        if valid:
            best_S = S_val
            best_valid = True
        else:
            # Return a very safe large hexagon
            inner_hex_data = np.zeros((11, 3))
            for i in range(11):
                inner_hex_data[i] = [initial_centers[i, 0], initial_centers[i, 1], 0]
            outer_hex_data = np.array([0.0, 0.0, 0.0])
            outer_hex_side_length = 10.0
            return inner_hex_data, outer_hex_data, outer_hex_side_length

    # Format output
    shifts = best_params[:22].reshape(11, 2)
    theta = best_params[22]
    centers = initial_centers + shifts
    
    inner_hex_data = np.zeros((11, 3))
    for i in range(11):
        inner_hex_data[i] = [centers[i, 0], centers[i, 1], 0.0]
    
    outer_hex_data = np.array([0.0, 0.0, theta])
    outer_hex_side_length = float(best_S)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length
