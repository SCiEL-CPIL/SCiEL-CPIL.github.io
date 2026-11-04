import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    sqrt3 = np.sqrt(3)
    
    # 2-3-4-3 hexagonal packing
    inner = []
    # Bottom row: 2
    for x in [-1, 1]:
        inner.append([x, -1.5*sqrt3, 0.0])
    # Second row: 3
    for x in [-2, 0, 2]:
        inner.append([x, -0.5*sqrt3, 0.0])
    # Third row: 4
    for x in [-3, -1, 1, 3]:
        inner.append([x, 0.5*sqrt3, 0.0])
    # Top row: 3
    for x in [-2, 0, 2]:
        inner.append([x, 1.5*sqrt3, 0.0])
    
    # Compute minimum side length needed
    # Container edge normals at angles 30, 90, 150, 210, 270, 330 degrees
    max_proj = 0.0
    for (cx, cy, angle) in inner:
        for k in range(6):
            theta = k * np.pi / 3.0
            vx = cx + np.cos(theta)
            vy = cy + np.sin(theta)
            for m in range(6):
                phi = (m + 0.5) * np.pi / 3.0  # 30, 90, 150, ...
                proj = vx * np.cos(phi) + vy * np.sin(phi)
                max_proj = max(max_proj, proj)
    
    s = max_proj / (sqrt3 / 2.0)
    
    return inner, (0.0, 0.0), s, 0.0
# EVOLVE-BLOCK-END
