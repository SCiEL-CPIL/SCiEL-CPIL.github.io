import numpy as np
import time
from scipy.optimize import minimize_scalar

def hexagon_packing_12():
    start_time = time.time()
    deadline = 600.0 # Safety margin
    
    # Helper to convert axial to cartesian
    def axial_to_cartesian(q, r):
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y

    # Helper to get vertices of a unit hexagon at (cx, cy)
    def get_hex_vertices(cx, cy, angle_deg=0):
        angles = np.deg2rad(angle_deg + np.arange(6) * 60)
        x = cx + np.cos(angles)
        y = cy + np.sin(angles)
        return np.column_stack((x, y))

    # Define candidate configurations (axial coordinates)
    # Config 1: Star missing (2, -2)
    config_1 = [
        (0,0), 
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (0,2), (-2,2), (-2,0), (0,-2)
    ]
    
    # Config 2: Star missing (2, 0)
    config_2 = [
        (0,0), 
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (0,2), (-2,2), (-2,0), (0,-2), (2,-2)
    ]
    
    # Config 3: Cap (5 consecutive in ring 2)
    config_3 = [
        (0,0), 
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (1,1), (0,2), (-1,2), (-2,2)
    ]
        
    # Config 4: Another variant
    config_4 = [
        (0,0), 
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (1,1), (0,2), (-1,2), (0,-2)
    ]

    candidates = [config_1, config_2, config_3, config_4]
    
    best_s = float('inf')
    best_config = None
    best_theta = 0
    best_centers = None
    
    for config in candidates:
        if time.time() > deadline:
            break
            
        # Calculate centers
        centers = np.array([axial_to_cartesian(q, r) for q, r in config])
        
        # Calculate all vertices
        all_vertices = []
        for cx, cy in centers:
            verts = get_hex_vertices(cx, cy)
            all_vertices.append(verts)
        all_vertices = np.vstack(all_vertices)
        
        # Objective function
        def objective(theta_deg):
            theta = np.deg2rad(theta_deg)
            # Normals to the outer hexagon sides
            # Outer hexagon side normals are at theta, theta+60, ...
            # The width of the inner cluster in direction n is max(dot) - min(dot)
            # s = max_width / sqrt(3)
            max_width = 0
            for k in range(6):
                n_angle = theta + k * np.pi / 3
                n = np.array([np.cos(n_angle), np.sin(n_angle)])
                projections = all_vertices @ n
                width = np.max(projections) - np.min(projections)
                if width > max_width:
                    max_width = width
            return max_width / np.sqrt(3)
            
        # Optimize theta in [0, 60]
        res = minimize_scalar(objective, bounds=(0, 60), method='bounded', options={'xatol': 1e-6})
        
        if res.success:
            s = res.fun
            if s < best_s:
                best_s = s
                best_config = config
                best_theta = res.x
                best_centers = centers
                
    # If no config found, fallback to simple grid (should not happen)
    if best_config is None:
        # Fallback to seed-like solution
        inner_hex_data = np.array([
            [0, 0, 0], [-2.5, 0, 0], [2.5, 0, 0],
            [-1.25, 2.17, 0], [1.25, 2.17, 0], [-1.25, -2.17, 0],
            [1.25, -2.17, 0], [-3.75, 2.17, 0], [3.75, 2.17, 0],
            [-3.75, -2.17, 0], [3.75, -2.17, 0], [0, -4, 0]
        ])
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8
        return inner_hex_data, outer_hex_data, outer_hex_side_length

    # Construct result
    # Inner hexagons: centers are best_centers, angle is 0 (lattice aligned)
    inner_hex_data = np.zeros((12, 3))
    for i, (cx, cy) in enumerate(best_centers):
        inner_hex_data[i] = [cx, cy, 0]
        
    # Outer hexagon: center at origin (we can shift everything to center the cluster)
    # To center the cluster, we should calculate the centroid of the vertices or centers?
    # The problem says "after shifting all centers by minus the outer center".
    # It's best to set the outer center to the centroid of the inner centers to minimize translation effects?
    # Actually, the width calculation is translation invariant.
    # So we can set outer center to (0,0) and keep inner centers as calculated.
    # However, the "inner vertex v outside outer hexagon" check uses absolute coordinates.
    # If we set outer center to (0,0), the outer hexagon is centered at (0,0).
    # The inner hexagons are at best_centers.
    # We should ensure the inner cluster is centered at (0,0) to be safe?
    # The width calculation doesn't depend on translation.
    # But the final coordinates returned must be consistent.
    # Let's shift the inner centers so their centroid is at (0,0).
    centroid = np.mean(best_centers, axis=0)
    shifted_centers = best_centers - centroid
    
    inner_hex_data = np.zeros((12, 3))
    for i, (cx, cy) in enumerate(shifted_centers):
        inner_hex_data[i] = [cx, cy, 0]
        
    outer_hex_data = np.array([0, 0, best_theta])
    outer_hex_side_length = best_s

    return inner_hex_data, outer_hex_data, outer_hex_side_length
