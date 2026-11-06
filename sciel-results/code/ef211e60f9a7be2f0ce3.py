"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    # Use a hexagonal packing configuration
    # 13 columns, 16 rows gives 208 points, we will select 200
    C = 13
    R = 16
    
    # Calculate optimal spacing s
    # Width constraint: (C + 0.5) * s <= 1 (assuming last row is shifted)
    # Height constraint: ((R - 1) * sqrt(3)/2 + 1) * s <= 1
    sqrt3_2 = np.sqrt(3) / 2
    s_width = 1.0 / (C + 0.5)
    s_height = 1.0 / ((R - 1) * sqrt3_2 + 1.0)
    s = min(s_width, s_height)
    r = s / 2.0
    
    # Generate points
    xs = -0.5 + r + np.arange(C) * s
    ys = -0.5 + r + np.arange(R) * (s * sqrt3_2)
    
    points = []
    for j in range(R):
        for i in range(C):
            x = xs[i]
            y = ys[j]
            if j % 2 == 1:
                x += s / 2.0
            points.append((x, y))
            
    points = np.array(points)
    
    # We have 208 points, need 200
    # Select the 200 points closest to the center of the point cloud
    center = np.mean(points, axis=0)
    dists = np.linalg.norm(points - center, axis=1)
    indices = np.argsort(dists)
    selected_indices = indices[:N]
    
    result = points[selected_indices]
    
    # Ensure shape is (N, 2)
    return result
# EVOLVE-BLOCK-END
