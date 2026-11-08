import numpy as np
from scipy.optimize import minimize
import time

N = 12

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 100  # 100s limit to be safe
    
    # Initial configuration: 3 rows of 4 hexagons
    # Pointy-top hexagons, side length 1
    # Horizontal spacing: 1.5
    # Vertical spacing: sqrt(3)
    
    positions = []
    angles = [0.0] * N
    
    # Generate 3x4 grid
    for r in range(3):
        for c in range(4):
            x = 1.5 * (c - 1.5) # Center the 4 columns: -1.5, -0.5, 0.5, 1.5
            y = np.sqrt(3) * (r - 1.0) # Center the 3 rows: -sqrt(3), 0, sqrt(3)
            # Offset odd rows (if 0-indexed, row 1 is offset)
            if r % 2 != 0:
                x += 0.75
            positions.append([x, y])
            
    positions = np.array(positions)
    
    # Objective function: Minimize S
    # S = (2/sqrt(3)) * max( dot(p, n_j) ) over all points p and normals n_j
    # Plus penalty for overlaps
    
    def objective(params):
        if time.time() > deadline:
            return 1e10
            
        x_coords = params[:N*2:2]
        y_coords = params[1:N*2:2]
        container_angle = params[-1]
        
        # Check overlaps
        # Distance between centers must be >= sqrt(3)
        # For aligned hexes, min distance is sqrt(3)
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = x_coords[i] - x_coords[j]
                dy = y_coords[i] - y_coords[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = np.sqrt(3) # Approx, actually depends on relative rotation, but we fix rotation to 0
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2 * 1000 # Heavy penalty
        
        if penalty > 1e-6:
            return 1e10 + penalty
            
        # Calculate required S
        # Container normals
        # Standard hexagon normals at 0, 60, 120, 180, 240, 300
        base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
        normals = np.column_stack((np.cos(base_angles + container_angle), np.sin(base_angles + container_angle)))
        
        # Vertices of each unit hexagon (pointy top)
        # Vertices at angles 30, 90, 150, 210, 270, 330?
        # Pointy top: vertices at 90, 30, -30, -90, -150, 150?
        # Let's use standard: 
        # v0: (0, 1)
        # v1: (sqrt(3)/2, 0.5)
        # v2: (sqrt(3)/2, -0.5)
        # v3: (0, -1)
        # v4: (-sqrt(3)/2, -0.5)
        # v5: (-sqrt(3)/2, 0.5)
        hex_vertices = np.array([
            [0, 1],
            [np.sqrt(3)/2, 0.5],
            [np.sqrt(3)/2, -0.5],
            [0, -1],
            [-np.sqrt(3)/2, -0.5],
            [-np.sqrt(3)/2, 0.5]
        ])
        
        max_proj = 0.0
        for i in range(N):
            center = np.array([x_coords[i], y_coords[i]])
            # All vertices are center + hex_vertices (since angle is 0)
            verts = center + hex_vertices
            # Project all vertices onto all normals
            # dots = verts @ normals.T
            # max_proj = max(max_proj, np.max(dots))
            # Vectorized:
            dots = verts @ normals.T
            local_max = np.max(dots)
            if local_max > max_proj:
                max_proj = local_max
                
        S = (2.0 / np.sqrt(3)) * max_proj
        return S

    # Initial parameters
    x0 = []
    for p in positions:
        x0.append(p[0])
        x0.append(p[1])
    x0.append(0.0) # container angle
    
    # Run Nelder-Mead
    best_result = None
    best_S = 1e10
    
    try:
        # Run for a limited time
        # Nelder-Mead options
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if res.success or res.fun < best_S:
            best_S = res.fun
            best_result = res.x
            
        # If time remains, try another start or more iterations?
        # For now, one run should be enough to beat S=40.
        
    except Exception as e:
        pass
        
    if best_result is None:
        # Fallback to initial guess
        best_result = np.array(x0)
        best_S = objective(best_result)
        
    # Extract final positions
    final_inner = []
    for i in range(N):
        x = best_result[i*2]
        y = best_result[i*2+1]
        angle = 0.0
        final_inner.append([x, y, angle])
        
    final_center = (0.0, 0.0)
    final_side = best_S
    final_angle = best_result[-1] * 180.0 / np.pi
    
    return final_inner, final_center, final_side, final_angle

