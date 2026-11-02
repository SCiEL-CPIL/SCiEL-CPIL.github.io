import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Generate centers for a 3x4 rhombus of hexagons
    # This shape is compact and regular.
    points = []
    for i in range(4):
        for j in range(3):
            x = np.sqrt(3) * (i + 0.5 * j)
            y = 1.5 * j
            points.append([x, y])
    
    points = np.array(points)
    
    # Center the points at origin
    center_offset = np.mean(points, axis=0)
    points = points - center_offset
    
    # Optimize rotation of the cluster relative to the outer hexagon
    # We scan the angle theta of the outer hexagon's orientation.
    # The inner hexagons are aligned with the outer hexagon.
    
    def calc_side(theta_deg):
        theta = np.radians(theta_deg)
        # Normals of the outer hexagon sides
        n0 = np.array([np.cos(theta), np.sin(theta)])
        n1 = np.array([np.cos(theta + np.pi/3), np.sin(theta + np.pi/3)])
        n2 = np.array([np.cos(theta + 2*np.pi/3), np.sin(theta + 2*np.pi/3)])
        
        max_proj = 0
        for n in [n0, n1, n2]:
            # Project all centers onto the normal
            projs = np.abs(points @ n)
            max_proj = max(max_proj, np.max(projs))
        
        # The outer hexagon inradius R must be >= max_proj + inradius_unit
        # R = S * sqrt(3) / 2
        # inradius_unit = sqrt(3) / 2
        # S * sqrt(3) / 2 >= max_proj + sqrt(3) / 2
        # S >= (max_proj + sqrt(3) / 2) * 2 / sqrt(3)
        S = (max_proj + np.sqrt(3)/2) * 2 / np.sqrt(3)
        return S

    # Scan for optimal theta in [0, 60)
    thetas = np.linspace(0, 60, 600)
    sides = np.array([calc_side(t) for t in thetas])
    min_idx = np.argmin(sides)
    best_theta = thetas[min_idx]
    best_side = sides[min_idx]
    
    # Construct the return values
    angle_deg = float(best_theta)
    inner = []
    for p in points:
        inner.append([float(p[0]), float(p[1]), angle_deg])
        
    return inner, (0.0, 0.0), float(best_side), angle_deg
# EVOLVE-BLOCK-END
