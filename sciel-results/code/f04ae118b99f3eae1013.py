"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    sqrt3 = math.sqrt(3)

    # Helper to calculate the minimum side length S of an outer hexagon
    # centered at (cx, cy) with angle theta (degrees) that contains N unit hexagons
    # centered at positions (x_i, y_i) with angle 0 (aligned).
    def calc_min_side(positions, theta_deg, cx=0.0, cy=0.0):
        theta_rad = math.radians(theta_deg)
        
        max_proj = 0.0
        for k in range(6):
            normal_angle = math.radians(30 + k * 60) + theta_rad
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            
            # Support function of unit hexagon in direction (nx, ny)
            # Unit hexagon vertices (angle 0):
            unit_verts = [
                (1.0, 0.0),
                (0.5, sqrt3/2),
                (-0.5, sqrt3/2),
                (-1.0, 0.0),
                (-0.5, -sqrt3/2),
                (0.5, -sqrt3/2)
            ]
            
            h_unit = 0.0
            for ux, uy in unit_verts:
                proj = ux * nx + uy * ny
                if proj > h_unit:
                    h_unit = proj
            
            # Max over all centers
            for x, y in positions:
                proj = (x - cx) * nx + (y - cy) * ny + h_unit
                if proj > max_proj:
                    max_proj = proj
        
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
    # The positions are scaled and rotated, then translated.
    def objective(params):
        if time.time() > deadline:
            return 1e10
        
        cx, cy, theta_deg, scale = params
        
        # Check overlap penalty based on scaled distances
        # Distance between centers in base cluster is d_base.
        # Scaled distance is d_base * scale.
        # We need d_base * scale >= 2.0 for no overlap.
        # Since base cluster has min distance 2.0 (neighbors), scale must be >= 1.0.
        # If we allow scale < 1.0, we get overlaps.
        # Penalty for overlaps: sum((2.0 - d_scaled)^2) for d_scaled < 2.0
        # d_scaled = d_base * scale
        # d_base is fixed for the cluster.
        
        # We can precompute min distance in base cluster. It's 2.0.
        # So if scale < 1.0, all neighbors overlap.
        # Penalty = N_pairs * (2.0 - 2.0*scale)^2 = N_pairs * 4.0*(1-scale)^2
        # But this is an approximation. Let's just enforce scale >= 1.0 with a large penalty if < 1.0
        # Or better, optimize scale only if it helps?
        # Actually, the cluster is rigid. Scaling it changes the relative positions.
        # If we scale by S, the distance between neighbors becomes 2*S.
        # If S < 1, they overlap.
        # So we should constrain S >= 1.
        
        if scale < 1.0:
            return 1e10 + (1.0 - scale) ** 2

        theta_rad = math.radians(theta_deg)
        c = math.cos(theta_rad)
        s = math.sin(theta_rad)
        
        # Rotate and scale positions
        # R = [[c, -s], [s, c]]
        # pos_scaled_rotated = R @ (scale * pos_base)
        # pos_final = pos_scaled_rotated + [cx, cy]
        
        # We want to minimize the side length of the outer hexagon containing these points.
        # The outer hexagon is centered at (0,0) with rotation 0 for simplicity in calc_min_side?
        # No, calc_min_side takes positions and outer center/angle.
        # Let's fix the outer hexagon at center (0,0) and angle 0, and optimize the inner cluster's
        # position (cx, cy) and rotation (theta) and scale (scale).
        # Wait, the outer hexagon can also rotate.
        # But relative rotation is what matters.
        # Let's fix outer hexagon angle to 0, and optimize inner cluster rotation.
        # This is equivalent.
        
        # Transform base positions
        x_scaled = base_positions[:, 0] * scale
        y_scaled = base_positions[:, 1] * scale
        
        x_rot = x_scaled * c - y_scaled * s
        y_rot = x_scaled * s + y_scaled * c
        
        x_final = x_rot + cx
        y_final = y_rot + cy
        
        pos = list(zip(x_final, y_final))
        
        # Calculate min side length of outer hexagon at (0,0) angle 0
        S = calc_min_side(pos, 0.0, 0.0, 0.0)
        
        return S

    # Initial guess
    # Scale 1.0
    # Rotation 0
    # Center 0,0
    init_params = [0.0, 0.0, 0.0, 1.0]
    
    best_params = init_params[:]
    best_score = objective(init_params)
    
    # Grid search over rotation and scale
    # Rotation: 0 to 60 degrees (symmetry)
    # Scale: 1.0 to 1.5
    
    # Coarse grid
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
    
    # The outer hexagon is at (0,0) with angle 0
    # But we should check if rotating the outer hexagon helps?
    # Since we optimized the inner rotation relative to outer angle 0,
    # it covers all relative orientations.
    # However, the "score" is the side length.
    # The outer hexagon center and rotation are part of the return value.
    # We can return center (0,0) and angle 0.
    
    final_side = calc_min_side(pos, 0.0, 0.0, 0.0)
    
    inner = []
    for x, y in pos:
        inner.append([x, y, 0.0])
        
    return inner, (0.0, 0.0), final_side, 0.0

# EVOLVE-BLOCK-END
