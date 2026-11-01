"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    # 1. Define fixed inner configuration
    # 3 rows, 4 columns, offset tiling
    # Side length = 1
    # Width (across flats) = sqrt(3)
    # Height (across corners) = 2
    # Vertical distance between rows = 1.5
    # Horizontal distance between cols = sqrt(3)
    # Offset for odd rows = sqrt(3)/2
    
    sqrt3 = np.sqrt(3)
    inner_positions = []
    
    # Row 0
    for j in range(4):
        x = (j - 1.5) * sqrt3
        y = 0.0
        inner_positions.append([x, y, 0.0])
        
    # Row 1
    for j in range(4):
        x = (j - 1.0) * sqrt3
        y = 1.5
        inner_positions.append([x, y, 0.0])
        
    # Row 2
    for j in range(4):
        x = (j - 1.5) * sqrt3
        y = 3.0
        inner_positions.append([x, y, 0.0])
        
    # 2. Compute all vertices of inner hexagons
    all_verts = []
    for pos in inner_positions:
        verts = hexagon_vertices(pos[0], pos[1], 1.0, pos[2])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    # 3. Optimize outer hexagon
    # We sweep the angle of the outer hexagon and for each angle, find the minimal enclosing hexagon
    best_side = float('inf')
    best_center = (0, 0)
    best_angle = 0.0
    
    def objective_outer(params, angle_deg):
        cx, cy, side = params
        if side < 0.1:
            return 1000.0
        
        # Check if all points are inside the outer hexagon
        # Outer hexagon vertices are not needed, just the half-plane inequalities
        # Hexagon with center (cx, cy), side s, angle a
        # Normals are at angles a + 30, a + 90, ...
        # Apothem = s * sqrt(3) / 2
        apothem = side * sqrt3 / 2.0
        angle_rad = np.radians(angle_deg)
        
        # The 6 normals
        normal_angles = angle_rad + np.radians(np.arange(30, 390, 60))
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        # Project points onto normals
        # p_proj = points @ normals.T
        # Center projection
        c_proj = np.array([cx, cy]) @ normals.T
        
        # Check inequalities: |p_proj - c_proj| <= apothem
        diffs = all_verts @ normals.T - c_proj
        max_diff = np.max(np.abs(diffs))
        
        if max_diff <= apothem + 1e-9:
            return side
        else:
            return side + (max_diff - apothem) * 10.0

    # Sweep angle from 0 to 60
    angles = np.linspace(0, 60, 30)
    
    for ang in angles:
        if time.time() > deadline - 5:
            break
            
        # Initial guess for center: centroid of all vertices
        cx_init, cy_init = np.mean(all_verts, axis=0)
        # Initial guess for side: bounding box diagonal / 2
        x_range = np.max(all_verts[:, 0]) - np.min(all_verts[:, 0])
        y_range = np.max(all_verts[:, 1]) - np.min(all_verts[:, 1])
        side_init = max(x_range, y_range) * 0.8
        
        res = minimize(objective_outer, [cx_init, cy_init, side_init], 
                       args=(ang,), method='Nelder-Mead',
                       options={'maxiter': 200, 'xatol': 0.01, 'fatol': 0.01, 'adaptive': True, 'maxfev': 500})
        
        if res.fun < best_side:
            best_side = res.fun
            best_center = (res.x[0], res.x[1])
            best_angle = ang
            
    # Final validation
    # If the optimized side is invalid, expand it
    # Check validity
    def is_valid(inner, center, side, angle):
        for pos in inner:
            verts = hexagon_vertices(pos[0], pos[1], 1.0, pos[2])
            for vx, vy in verts:
                # Check point in hexagon
                dx = vx - center[0]
                dy = vy - center[1]
                angle_rad = np.radians(angle)
                cos_a = np.cos(-angle_rad)
                sin_a = np.sin(-angle_rad)
                local_x = dx * cos_a - dy * sin_a
                local_y = dx * sin_a + dy * cos_a
                
                apothem = side * sqrt3 / 2.0
                normal_angles = np.radians(np.arange(30, 390, 60))
                for na in normal_angles:
                    nx = np.cos(na)
                    ny = np.sin(na)
                    dist = local_x * nx + local_y * ny
                    if dist > apothem + 1e-9:
                        return False
        return True

    if not is_valid(inner_positions, best_center, best_side, best_angle):
        # Expand side until valid
        while not is_valid(inner_positions, best_center, best_side, best_angle) and best_side < 20:
            best_side += 0.1

    return inner_positions, best_center, best_side, best_angle
# EVOLVE-BLOCK-END
