import numpy as np
import time
from scipy.optimize import minimize_scalar

def hexagon_packing_12():
    start_time = time.time()
    deadline = 620.0  # Safety margin before 630s kill

    # Helper to convert axial to cartesian
    def axial_to_cartesian(q, r):
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y

    # Helper to get vertices of a unit hexagon at (cx, cy) with given angle
    def get_hex_vertices(cx, cy, angle_deg=0):
        angles = np.deg2rad(angle_deg + np.arange(6) * 60)
        x = cx + np.cos(angles)
        y = cy + np.sin(angles)
        return np.column_stack((x, y))

    # Define candidate configurations (axial coordinates)
    # These are different packings of 12 hexagons on a hexagonal lattice
    config_1 = [
        (0,0), 
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (0,2), (-2,2), (-2,0), (0,-2)
    ]
    
    config_2 = [
        (0,0), 
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (0,2), (-2,2), (-2,0), (0,-2), (2,-2)
    ]
    
    config_3 = [
        (0,0), 
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (1,1), (0,2), (-1,2), (-2,2)
    ]
    
    config_4 = [
        (0,0), 
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (1,1), (0,2), (-1,2), (0,-2)
    ]

    config_5 = [
        (0,0), 
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (2,-1), (1,-2), (0,-2), (-1,-2)
    ]

    config_6 = [
        (0,0), 
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (-2,2), (-2,1), (-1,0), (0,-1), (1,-2)
    ]

    candidates = [config_1, config_2, config_3, config_4, config_5, config_6]
    
    best_s = float('inf')
    best_config = None
    best_theta = 0
    best_centers = None
    
    for config in candidates:
        if time.time() > deadline - 10:
            break
            
        # Calculate centers
        centers = np.array([axial_to_cartesian(q, r) for q, r in config])
        
        # Calculate all vertices
        all_vertices = []
        for cx, cy in centers:
            verts = get_hex_vertices(cx, cy)
            all_vertices.append(verts)
        all_vertices = np.vstack(all_vertices)
        
        # Objective function: minimize the side length of the containing hexagon
        def objective(theta_deg):
            theta = np.deg2rad(theta_deg)
            # The outer hexagon has side length s, so its apothem is s * sqrt(3)/2
            # For a hexagon with side s, the distance from center to each side is s * sqrt(3)/2
            # We need to find the minimum s such that all vertices fit inside
            
            # The constraint is that for each of the 6 directions (normals to sides),
            # the maximum projection of any vertex onto that normal must be <= apothem
            # where apothem = s * sqrt(3)/2
            
            max_apothem_needed = 0
            for k in range(6):
                n_angle = theta + k * np.pi / 3
                n = np.array([np.cos(n_angle), np.sin(n_angle)])
                projections = all_vertices @ n
                max_proj = np.max(projections)
                min_proj = np.min(projections)
                # The width in this direction is max_proj - min_proj
                # The apothem needed is at least (max_proj - min_proj) / 2
                # But we also need to account for the center offset
                # Actually, the hexagon is centered at origin, so we need:
                # max_proj <= apothem AND min_proj >= -apothem
                # So apothem >= max(abs(max_proj), abs(min_proj))
                required_apothem = max(abs(max_proj), abs(min_proj))
                if required_apothem > max_apothem_needed:
                    max_apothem_needed = required_apothem
            
            # s = apothem * 2 / sqrt(3)
            s = max_apothem_needed * 2 / np.sqrt(3)
            return s
            
        # Optimize theta in [0, 60]
        res = minimize_scalar(objective, bounds=(0, 60), method='bounded', options={'xatol': 1e-8})
        
        if res.success:
            s = res.fun
            if s < best_s:
                best_s = s
                best_config = config
                best_theta = res.x
                best_centers = centers
                
    # If no config found, fallback
    if best_config is None:
        inner_hex_data = np.zeros((12, 3))
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 10.0
        return inner_hex_data, outer_hex_data, outer_hex_side_length

    # Construct result
    # Center the configuration at origin
    centroid = np.mean(best_centers, axis=0)
    shifted_centers = best_centers - centroid
    
    inner_hex_data = np.zeros((12, 3))
    for i, (cx, cy) in enumerate(shifted_centers):
        inner_hex_data[i] = [cx, cy, 0]
        
    outer_hex_data = np.array([0, 0, best_theta])
    outer_hex_side_length = best_s

    return inner_hex_data, outer_hex_data, outer_hex_side_length
