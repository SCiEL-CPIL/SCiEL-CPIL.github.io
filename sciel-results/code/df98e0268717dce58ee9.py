import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_11():
    start_time = time.time()
    timeout = 600 # seconds
    
    # 11 unit hexagons
    # Initial guess: a hexagonal grid pattern
    # Center (0,0)
    # 6 neighbors at distance sqrt(3)
    # 4 more... maybe further out?
    
    # Let's try a specific configuration that is known to be compact.
    # A 3x3 grid of hexagons has 9. We need 2 more.
    # Or the 7-hex flower + 4.
    
    # Let's define the vertices of a unit hexagon at origin, angle 0
    # Vertices: (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    
    hex_verts = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    # Normals of the outer hexagon (angle 0)
    # Edges are perpendicular to normals at 0, 60, ...
    # Normal vectors: (1,0), (0.5, sqrt(3)/2), ...
    normals = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    def calculate_min_side(positions):
        # positions: (11, 2) array of centers
        # Calculate all vertices
        # v_i = p_i + hex_verts
        # We need to find the minimal s such that all vertices are inside the hexagon of side s centered at C.
        # To minimize s, we should also optimize C.
        # For a fixed set of vertices, the optimal C is the centroid? Not necessarily.
        # The minimal enclosing hexagon of a set of points.
        # This is a convex optimization problem.
        # s = max_k (max_v ( (v - C) dot n_k )) * 2/sqrt(3)
        # Let's fix C to the centroid of the centers for now, or optimize it.
        
        # Simple approach: 
        # 1. Compute centroid of centers.
        # 2. Compute max projection in each of the 6 directions from the centroid.
        # 3. s = max_proj * 2/sqrt(3).
        
        centroid = np.mean(positions, axis=0)
        
        max_proj = 0.0
        for k in range(6):
            n = normals[k]
            # Project all vertices onto n
            # vertices = positions[:, np.newaxis, :] + hex_verts[np.newaxis, :, :]
            # proj = (vertices - centroid) dot n
            # max_proj_k = max(proj)
            
            # Efficiently:
            # proj = (positions - centroid) dot n + hex_verts dot n
            base_proj = (positions - centroid) @ n
            hex_proj = hex_verts @ n
            # For each hexagon i, the max projection is base_proj[i] + max(hex_proj)
            # max over i of (base_proj[i] + max(hex_proj))
            # = max(base_proj) + max(hex_proj)
            
            max_base = np.max(base_proj)
            max_hex = np.max(hex_proj)
            max_proj_k = max_base + max_hex
            if max_proj_k > max_proj:
                max_proj = max_proj_k
                
        s = max_proj * 2.0 / np.sqrt(3)
        return s

    # Initial positions
    # Let's use the 7-hex flower and add 4 more at distance 2*sqrt(3) or so?
    # Or just a 3x3 grid + 2.
    
    # 3x3 grid of hexagons (offset)
    # Row 0: 0, 1, 2
    # Row 1: 0.5, 1.5, 2.5 (shifted)
    # Row 2: 0, 1, 2
    # This is 9 hexagons.
    # Add 2 more?
    
    # Let's try to optimize a random set of 11 points.
    # We need to ensure they don't overlap.
    # Overlap constraint: dist(p_i, p_j) >= sqrt(3)
    
    # We will use a penalty function.
    
    def objective_and_penalty(x):
        # x: (22,) flattened positions
        positions = x.reshape(11, 2)
        
        # Penalty for overlap
        penalty = 0.0
        for i in range(11):
            for j in range(i+1, 11):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < np.sqrt(3):
                    penalty += (np.sqrt(3) - d) ** 2
                    
        # Calculate side
        s = calculate_min_side(positions)
        
        # We want to minimize s, but avoid overlaps.
        # If penalty > 0, we are invalid.
        # Return s + large_penalty * penalty
        return s + 1000 * penalty

    # Initial guess
    np.random.seed(42)
    # Start with a compact cluster
    # Center at 0
    # 6 around
    # 4 further
    
    # Let's just start with a grid and let it move
    x0 = np.zeros(22)
    # Place them in a rough 3x4 grid
    for i in range(11):
        row = i // 4
        col = i % 4
        x0[2*i] = col * 1.8 - 1.5
        x0[2*i+1] = row * 1.6 - 1.0
        
    # Optimize
    result = None
    best_s = float('inf')
    
    # Run optimization
    # Use Nelder-Mead as it doesn't require gradients and is robust for small dims
    options = {'maxiter': 10000, 'xatol': 1e-4, 'fatol': 1e-4}
    
    try:
        res = minimize(objective_and_penalty, x0, method='Nelder-Mead', options=options)
        if res.success or res.fun < best_s:
            best_s = res.fun
            # Extract s from res.fun? No, res.fun includes penalty.
            # We need to recalculate s for the best positions
            positions = res.x.reshape(11, 2)
            # Check if valid
            valid = True
            for i in range(11):
                for j in range(i+1, 11):
                    if np.linalg.norm(positions[i] - positions[j]) < np.sqrt(3) - 1e-6:
                        valid = False
            if valid:
                s = calculate_min_side(positions)
                if s < best_s: # best_s was penalized, so s might be smaller
                    best_s = s
                    result = positions
    except Exception as e:
        pass
        
    if result is None:
        # Fallback to a known valid configuration
        # 7-hex flower + 4 more
        # This might have a larger s, but it's valid.
        positions = np.array([
            [0, 0],
            [np.sqrt(3), 0],
            [np.sqrt(3)/2, 1.5],
            [-np.sqrt(3)/2, 1.5],
            [-np.sqrt(3), 0],
            [-np.sqrt(3)/2, -1.5],
            [np.sqrt(3)/2, -1.5],
            # Add 4 more further out
            [2*np.sqrt(3), 0],
            [np.sqrt(3), 3],
            [-np.sqrt(3), 3],
            [-2*np.sqrt(3), 0]
        ])
        s = calculate_min_side(positions)
        result = positions
        best_s = s
    else:
        s = best_s
        
    # Construct return values
    inner_hex_data = np.zeros((11, 3))
    inner_hex_data[:, 0] = result[:, 0]
    inner_hex_data[:, 1] = result[:, 1]
    inner_hex_data[:, 2] = 0 # Angle 0
    
    outer_hex_data = np.array([0, 0, 0])
    
    return inner_hex_data, outer_hex_data, s

