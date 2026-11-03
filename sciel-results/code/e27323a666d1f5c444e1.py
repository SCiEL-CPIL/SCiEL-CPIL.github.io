import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Helper to get vertices of a regular hexagon
    def hex_vertices(center, angle_deg, side=1.0):
        angle_rad = np.radians(angle_deg)
        # Vertices are at angles 0, 60, 120, 180, 240, 300 relative to center
        # Standard hexagon with flat top has vertices at 30, 90, ...
        # Let's use pointy-top or flat-top?
        # The task says "regular hexagons". Orientation is free.
        # Let's define vertices at angles: angle + k * 60 degrees.
        angles = angle_rad + np.pi/3 * np.arange(6)
        x = center[0] + side * np.cos(angles)
        y = center[1] + side * np.sin(angles)
        return np.column_stack((x, y))

    # Helper to calculate the maximum projection of a set of points onto a direction
    def max_projection(points, direction):
        # points: (M, 2)
        # direction: (2,)
        # Returns max dot product
        return np.max(points @ direction)

    # Objective function for optimization
    # Variables:
    # 0: outer_x
    # 1: outer_y
    # 2: outer_angle_deg
    # 3: outer_side (R)
    # 4..: inner centers and angles
    #   For i in range(N):
    #     4 + 3*i: inner_x[i]
    #     5 + 3*i: inner_y[i]
    #     6 + 3*i: inner_angle[i]
    
    def objective(vars):
        ox = vars[0]
        oy = vars[1]
        o_ang = vars[2]
        R = vars[3]
        
        # Outer hexagon center and normals
        outer_center = np.array([ox, oy])
        # Normals of outer hexagon are perpendicular to edges.
        # If vertices are at angles o_ang + k*pi/3, edges are at o_ang + pi/6 + k*pi/3.
        # Normals are at o_ang + k*pi/3.
        normal_angles = np.radians(o_ang) + np.pi/3 * np.arange(6)
        normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
        
        # Apothem of outer hexagon
        apothem = R * np.sqrt(3) / 2.0
        
        penalty = 0.0
        
        # Check each inner hexagon
        for i in range(N):
            ix = vars[4 + 3*i]
            iy = vars[5 + 3*i]
            i_ang = vars[6 + 3*i]
            
            inner_center = np.array([ix, iy])
            # Get vertices of inner hexagon
            verts = hex_vertices(inner_center, i_ang, 1.0)
            # Translate vertices relative to outer center
            verts_rel = verts - outer_center
            
            # Check projection against each normal
            for k in range(6):
                n = normals[k]
                proj = np.max(verts_rel @ n)
                if proj > apothem:
                    penalty += (proj - apothem)**2
        
        # We want to minimize R.
        # If R is too small, penalty is huge.
        # We can just return R + penalty.
        # To ensure R doesn't go to 0, we can add a small term or rely on penalty.
        return R + 100.0 * penalty

    # Initial guess
    # Honeycomb arrangement
    s3 = np.sqrt(3)
    init_inners = [
        [0.0, 0.0],
        [1.5, s3/2], [1.5, -s3/2],
        [3.0, 0.0],
        [0.0, s3], [0.0, -s3],
        [1.5, 3*s3/2], [1.5, -3*s3/2],
        [3.0, s3], [3.0, -s3],
        [4.5, s3/2], [4.5, -s3/2]
    ]
    
    # Initial outer hexagon
    # Center of mass of inners
    cx = np.mean([p[0] for p in init_inners])
    cy = np.mean([p[1] for p in init_inners])
    
    x0 = [cx, cy, 0.0, 5.0] # R=5 initial guess
    for p in init_inners:
        x0.append(p[0])
        x0.append(p[1])
        x0.append(0.0) # Initial angle 0
    
    x0 = np.array(x0)
    
    best_result = None
    best_val = np.inf
    
    # Run Nelder-Mead
    # Max iterations
    max_iter = 5000
    
    while time.time() < deadline:
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-6, 'disp': False})
            
            if res.success or res.fun < best_val:
                best_val = res.fun
                best_result = res.x
            
            # If we found a valid solution (low penalty), we might refine
            # Check if penalty is zero
            ox, oy, o_ang, R = best_result[0], best_result[1], best_result[2], best_result[3]
            # Quick check for validity
            valid = True
            outer_center = np.array([ox, oy])
            normal_angles = np.radians(o_ang) + np.pi/3 * np.arange(6)
            normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
            apothem = R * np.sqrt(3) / 2.0
            
            for i in range(N):
                ix, iy, i_ang = best_result[4+3*i], best_result[5+3*i], best_result[6+3*i]
                inner_center = np.array([ix, iy])
                verts = hex_vertices(inner_center, i_ang, 1.0)
                verts_rel = verts - outer_center
                for k in range(6):
                    n = normals[k]
                    proj = np.max(verts_rel @ n)
                    if proj > apothem + 1e-5:
                        valid = False
                        break
                if not valid: break
            
            if valid:
                # We have a valid solution. Try to polish R down?
                # Or just accept it.
                # The objective minimizes R + penalty. If penalty is 0, it minimizes R.
                # So this should be close to optimal for this configuration.
                break
                
        except Exception as e:
            pass

    if best_result is None:
        # Fallback to initial guess
        best_result = x0

    # Extract final parameters
    ox, oy, o_ang, R = best_result[0], best_result[1], best_result[2], best_result[3]
    
    inner = []
    for i in range(N):
        ix = best_result[4 + 3*i]
        iy = best_result[5 + 3*i]
        i_ang = best_result[6 + 3*i]
        inner.append([ix, iy, i_ang])
        
    return inner, (ox, oy), R, o_ang

# EVOLVE-BLOCK-END
