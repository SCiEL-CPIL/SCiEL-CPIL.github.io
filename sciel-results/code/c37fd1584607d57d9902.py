"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    sqrt3 = math.sqrt(3)

    # Helper to calculate the minimum side length S of an outer hexagon
    # centered at (cx, cy) with angle theta (degrees) that contains N unit hexagons
    # centered at positions (x_i, y_i) with angle phi_i (degrees).
    def calc_min_side(positions, angles_deg, theta_deg, cx=0.0, cy=0.0):
        theta_rad = math.radians(theta_deg)
        
        max_proj = 0.0
        for k in range(6):
            normal_angle = math.radians(30 + k * 60) + theta_rad
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            
            # Max projection of all hexagons in this direction
            max_h = 0.0
            for i in range(len(positions)):
                x, y = positions[i]
                phi_rad = math.radians(angles_deg[i])
                
                # Support function of unit hexagon rotated by phi in direction (nx, ny)
                # Relative angle: alpha = normal_angle - phi_rad
                # h(alpha) = max( cos(alpha), cos(alpha-60), cos(alpha-120), ... )
                # Due to symmetry, we can check 6 vertices.
                # Vertices of unit hex at angle 0: (1,0), (0.5, sqrt3/2), ...
                # Projected vertex j: dot(v_j, n) = cos(phi + j*60 - normal_angle)
                # Let delta = normal_angle - phi_rad.
                # proj = cos(phi + j*60 - (phi + delta)) = cos(j*60 - delta)
                # We want max over j of cos(j*60 - delta).
                
                # Efficiently:
                # The maximum projection of a regular hexagon with circumradius 1 in any direction is:
                # h = cos( delta mod 60 - 30 ) if we align correctly?
                # Actually, the support function of a regular hexagon with circumradius R=1 is:
                # h(theta) = cos( (theta mod 60) - 30 ) * ? No.
                # Let's just compute the 6 projections. It's fast enough.
                
                h_val = 0.0
                for j in range(6):
                    ang = math.radians(j * 60) + phi_rad
                    vx = math.cos(ang)
                    vy = math.sin(ang)
                    proj = vx * nx + vy * ny
                    if proj > h_val:
                        h_val = proj
                
                total_proj = (x - cx) * nx + (y - cy) * ny + h_val
                if total_proj > max_h:
                    max_h = total_proj
            
            if max_h > max_proj:
                max_proj = max_h
        
        return max_proj * 2.0 / sqrt3

    # Generate the 12-hexagon cluster: 19-hex cluster minus 6 corners minus 1 side hex
    centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q) <= 2 and abs(r) <= 2 and abs(q+r) <= 2:
                x = q + r/2.0
                y = r * sqrt3/2.0
                centers.append((x, y))
                
    corner_centers = []
    side_centers = []
    inner_centers = []
    
    for x, y in centers:
        dist = math.sqrt(x*x + y*y)
        if dist > 3.8: # Corners at 4.0
            corner_centers.append((x, y))
        elif dist > 3.0: # Side hexes at sqrt(12) ~ 3.46
            side_centers.append((x, y))
        else:
            inner_centers.append((x, y))
            
    removed_side = side_centers[0] if side_centers else None
    
    final_centers = inner_centers + [c for c in side_centers if c != removed_side]
    
    if len(final_centers) != 12:
        positions = []
        for row in range(4):
            y = row * sqrt3
            for col in range(3):
                if row % 2 == 0:
                    x = (col - 1) * 2.0
                else:
                    x = (col - 1) * 2.0 + 1.0
                positions.append((x, y))
        final_centers = positions

    # Center the arrangement
    xs = [p[0] for p in final_centers]
    ys = [p[1] for p in final_centers]
    cx_offset = (min(xs) + max(xs)) / 2.0
    cy_offset = (min(ys) + max(ys)) / 2.0
    centered_positions = [(x - cx_offset, y - cy_offset) for x, y in final_centers]
    
    # Convert to numpy array for vectorized operations
    base_positions = np.array(centered_positions)
    
    # Objective function for rigid body parameters: (cx, cy, theta_deg, scale)
    def objective(params):
        if time.time() > deadline:
            return 1e10
        
        cx, cy, theta_deg, scale = params
        
        if scale < 1.0:
            return 1e10 + (1.0 - scale) ** 2

        theta_rad = math.radians(theta_deg)
        c = math.cos(theta_rad)
        s = math.sin(theta_rad)
        
        x_scaled = base_positions[:, 0] * scale
        y_scaled = base_positions[:, 1] * scale
        
        x_rot = x_scaled * c - y_scaled * s
        y_rot = x_scaled * s + y_scaled * c
        
        x_final = x_rot + cx
        y_final = y_rot + cy
        
        pos = list(zip(x_final, y_final))
        angles = [0.0] * 12
        
        # Calculate min side length of outer hexagon at (0,0) angle 0
        S = calc_min_side(pos, angles, 0.0, 0.0, 0.0)
        
        return S

    # Initial guess
    init_params = [0.0, 0.0, 0.0, 1.0]
    
    best_params = init_params[:]
    best_score = objective(init_params)
    
    # Grid search over rotation and scale
    rot_steps = np.linspace(0, 60, 25)
    scale_steps = np.linspace(1.0, 1.2, 10)
    
    for theta in rot_steps:
        for scale in scale_steps:
            # Optimize cx, cy for fixed theta, scale
            def obj_center(params):
                return objective([params[0], params[1], theta, scale])
            
            res = minimize(obj_center, [0.0, 0.0], method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
            
            score = res.fun
            if score < best_score:
                best_score = score
                best_params = [res.x[0], res.x[1], theta, scale]
                
            if time.time() > deadline - 20:
                break
        if time.time() > deadline - 20:
            break
    
    # Fine optimization around best
    if time.time() < deadline - 10:
        res = minimize(objective, best_params, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
    
    cx, cy, theta_deg, scale = best_params
    
    # Calculate final positions
    theta_rad = math.radians(theta_deg)
    c = math.cos(theta_rad)
    s = math.sin(theta_rad)
    
    x_scaled = base_positions[:, 0] * scale
    y_scaled = base_positions[:, 1] * scale
    
    x_rot = x_scaled * c - y_scaled * s
    y_rot = x_scaled * s + y_scaled * c
    
    x_final = x_rot + cx
    y_final = y_rot + cy
    
    pos = list(zip(x_final, y_final))
    angles = [0.0] * 12
    
    final_side = calc_min_side(pos, angles, 0.0, 0.0, 0.0)
    
    inner = []
    for i in range(12):
        inner.append([pos[i][0], pos[i][1], 0.0])
        
    return inner, (0.0, 0.0), final_side, 0.0

# EVOLVE-BLOCK-END
