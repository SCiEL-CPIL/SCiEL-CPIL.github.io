import numpy as np
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    d = math.sqrt(3)
    
    # 11 hexagons in a compact grid
    axial_coords = [
        (0, 0),
        (1, 0), (1, -1), (0, -1),
        (-1, 0), (-1, 1), (0, 1),
        (2, 0), (2, -1), (1, -2), (0, -2)
    ]
    
    centers = []
    for q, r in axial_coords:
        x = d * (q + r / 2.0)
        y = d * (math.sqrt(3) / 2.0) * r
        centers.append((x, y))
        
    vertices = []
    for cx, cy in centers:
        for k in range(6):
            angle = math.radians(60 * k)
            vx = cx + 1.0 * math.cos(angle)
            vy = cy + 1.0 * math.sin(angle)
            vertices.append((vx, vy))
            
    vertices = np.array(vertices)
    centroid = np.mean(vertices, axis=0)
    
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Scan angles 0-60 deg
    for theta_deg in range(0, 61, 1):
        theta = math.radians(theta_deg)
        normals = []
        for k in range(6):
            a = theta + math.radians(60 * k)
            normals.append((math.cos(a), math.sin(a)))
        
        # Initial center: centroid
        c = centroid
        
        # Local search for center
        # We want to minimize max_k max_p |(p-c).n_k|
        # Gradient descent or simple grid search around centroid
        best_local_side = float('inf')
        best_local_c = c
        
        # Simple grid search for center in a small box around centroid
        # Step size 0.1
        for dx in np.linspace(-0.5, 0.5, 11):
            for dy in np.linspace(-0.5, 0.5, 11):
                c_test = (c[0] + dx, c[1] + dy)
                max_proj = 0.0
                for nx, ny in normals:
                    projs = (vertices[:, 0] - c_test[0]) * nx + (vertices[:, 1] - c_test[1]) * ny
                    local_max = np.max(np.abs(projs))
                    if local_max > max_proj:
                        max_proj = local_max
                side = 2.0 * max_proj / math.sqrt(3)
                if side < best_local_side:
                    best_local_side = side
                    best_local_c = c_test
                    
        if best_local_side < best_side:
            best_side = best_local_side
            best_center = best_local_c
            best_angle = theta_deg
            
    inner = [[cx, cy, 0.0] for cx, cy in centers]
    return inner, best_center, best_side, best_angle

# EVOLVE-BLOCK-END
