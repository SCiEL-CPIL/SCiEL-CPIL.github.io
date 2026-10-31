import numpy as np
import time

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    sqrt3 = np.sqrt(3)
    
    # Hexagonal lattice: axial to cartesian
    # For flat-top hexagons with side 1, center-to-center distance = sqrt(3)
    # Axial (q,r) to cartesian: x = sqrt(3)*(q + r/2), y = 1.5*r
    
    # 11 hexagons in a compact hexagonal cluster
    axial_coords = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
        (2, -1), (2, 0), (-2, 1), (0, 2),
    ]
    
    inner_points = []
    for q, r in axial_coords:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        inner_points.append((x, y))
    
    # For each outer hexagon rotation, compute the minimal enclosing hexagon
    # by finding the optimal center and side length
    
    # A regular hexagon with center (cx, cy), side s, rotation alpha has:
    # - 6 edge normals at angles alpha + 90 + k*60 for k=0..5
    # - A point (px,py) is inside iff (px-cx)*nx_k + (py-cy)*ny_k <= s*sqrt(3)/2 for all k
    
    # For an inner hexagon with center (ix, iy), side 1, rotation beta:
    # Its support in direction (nx, ny) is the max over its 6 vertices of (vx*nx + vy*ny)
    
    def hex_support(ix, iy, side, angle_deg, nx, ny):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
        vx = ix + side * np.cos(angles)
        vy = iy + side * np.sin(angles)
        return np.max(vx * nx + vy * ny)
    
    best_s = float('inf')
    best_cx, best_cy, best_outer_angle = 0, 0, 0
    
    # Try outer rotations 0 to 30 degrees
    for outer_angle in np.linspace(0, 30, 31):
        if time.time() > deadline:
            break
        
        # Outer hexagon edge normals
        outer_normal_angles = np.radians(outer_angle) + np.array([90, 150, 210, 270, 330, 30]) * np.pi / 180
        normals = np.array([[np.cos(a), np.sin(a)] for a in outer_normal_angles])
        
        # Compute support of each inner hexagon in each outer normal direction
        # Inner hexagons have side=1, rotation=0
        supports = np.zeros((len(inner_points), 6))
        for j, (ix, iy) in enumerate(inner_points):
            for i in range(6):
                nx, ny = normals[i]
                supports[j, i] = hex_support(ix, iy, 1.0, 0.0, nx, ny)
        
        # For given center (cx, cy), required side:
        # s >= (supports[j,i] - cx*nx_i - cy*ny_i) / (sqrt(3)/2) for all j,i
        # Minimize over cx, cy
        
        # Grid search over cx, cy
        cx_range = np.linspace(-4, 4, 40)
        cy_range = np.linspace(-4, 4, 40)
        
        for cx in cx_range:
            for cy in cy_range:
                s_vals = np.zeros((len(inner_points), 6))
                for i in range(6):
                    nx, ny = normals[i]
                    s_vals[:, i] = (supports[:, i] - cx * nx - cy * ny) / (sqrt3 / 2.0)
                s = np.max(s_vals)
                if s < best_s:
                    best_s = s
                    best_cx, best_cy = cx, cy
                    best_outer_angle = outer_angle
    
    if best_s == float('inf'):
        # Fallback
        inner = [[x, y, 0.0] for x, y in inner_points]
        return inner, (0.0, 0.0), 10.0, 0.0
    
    inner = [[x, y, 0.0] for x, y in inner_points]
    return inner, (best_cx, best_cy), best_s, best_outer_angle

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
