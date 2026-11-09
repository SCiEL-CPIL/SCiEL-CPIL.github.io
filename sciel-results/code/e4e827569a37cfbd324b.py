import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    sqrt3 = math.sqrt(3.0)
    
    # 1. Generate Initial Static Packing (from Inspiration 1)
    # Select 12 central hexagons from the 19-hex cluster (radius 2)
    centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                x = sqrt3 * (q + r / 2.0)
                y = 1.5 * r
                centers.append((x, y))
                
    # Sort by distance from origin to pick the most compact 12
    centers.sort(key=lambda p: p[0]**2 + p[1]**2)
    selected_centers = centers[:12]
    
    # Scale slightly to ensure strict disjointness in the static start
    scale = 1.001
    
    # Initial variables:
    # p0: 12 centers (x,y) -> 24 vars
    # alphas: 12 rotations -> 12 vars
    # theta: outer rotation -> 1 var
    # Total: 37 vars
    
    p_init = np.zeros((N, 2))
    for i, (x, y) in enumerate(selected_centers):
        p_init[i] = [x * scale, y * scale]
        
    alphas_init = np.zeros(N)
    theta_init = 0.0
    
    # Flatten for SLSQP
    x0 = np.concatenate((p_init.flatten(), alphas_init, [theta_init]))
    
    # Vertices of a unit hexagon (side 1)
    # Center at origin. Vertices at distance 1.
    # Angles: 0, 60, 120, 180, 240, 300
    angles = np.array([k * np.pi / 3.0 for k in range(6)])
    unit_verts = np.column_stack((np.cos(angles), np.sin(angles)))
    
    eps = 1e-5
    min_dist = sqrt3 + eps
    min_dist_sq = min_dist**2

    def objective(x):
        if time.time() > deadline:
            return 1e10 # Timeout penalty
            
        # Unpack
        p = x[:24].reshape(N, 2)
        alphas = x[24:36]
        theta = x[36]
        
        # Calculate all vertices of all inner hexagons
        # Shape: (N, 6, 2)
        # For each hex i, rotate unit_verts by alphas[i] and add p[i]
        
        # Optimization: Pre-compute rotation matrices?
        # cos_a, sin_a
        cos_a = np.cos(alphas)
        sin_a = np.sin(alphas)
        
        # Rotation matrix for each hex: [[cos, -sin], [sin, cos]]
        # Apply to unit_verts
        # unit_verts: (6, 2)
        # We want (N, 6, 2)
        
        # vx = cos_a[:,None] * unit_verts[0] - sin_a[:,None] * unit_verts[1]
        # vy = sin_a[:,None] * unit_verts[0] + cos_a[:,None] * unit_verts[1]
        
        vx = cos_a[:, np.newaxis] * unit_verts[0, np.newaxis] - sin_a[:, np.newaxis] * unit_verts[1, np.newaxis]
        vy = sin_a[:, np.newaxis] * unit_verts[0, np.newaxis] + cos_a[:, np.newaxis] * unit_verts[1, np.newaxis]
        
        verts = np.stack((vx, vy), axis=-1) # (N, 6, 2)
        verts += p[:, np.newaxis, :] # Add centers
        
        # Calculate normals of outer hexagon
        # Normals at theta + pi/6 + k*pi/3
        normal_angles = theta + np.pi/6 + np.arange(6) * np.pi/3
        normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles))) # (6, 2)
        
        # Calculate max projection
        # We need max over all vertices and all normals of (v . n)
        # verts: (N, 6, 2) -> (N*6, 2)
        # normals: (6, 2)
        
        # Efficient computation:
        # proj[i, j, k] = verts[i, j] . normals[k]
        # We want max(proj)
        
        # Reshape verts to (N*6, 2)
        verts_flat = verts.reshape(-1, 2)
        
        # Compute dot products: (N*6, 6)
        # dots[i, k] = verts_flat[i] . normals[k]
        dots = verts_flat @ normals.T
        
        max_proj = np.max(dots)
        
        # Side length L = (2 / sqrt3) * max_proj
        L = (2.0 / sqrt3) * max_proj
        
        return L

    # Constraints: Non-overlap
    # d(i, j) >= min_dist
    # d(i, j)^2 >= min_dist_sq
    # min_dist_sq - ||p_i - p_j||^2 >= 0
    
    constraints = []
    for i in range(N):
        for j in range(i+1, N):
            def make_constraint(i, j):
                def con(x):
                    if time.time() > deadline:
                        return 0.0
                    p = x[:24].reshape(N, 2)
                    diff = p[i] - p[j]
                    dist_sq = np.dot(diff, diff)
                    return min_dist_sq - dist_sq
                return con
            constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})
            
    # Bounds
    # Positions: (-5, 5)
    # Angles: (-pi, pi)
    # Theta: (0, 2*pi)
    
    bounds = []
    for i in range(N):
        bounds.append((-5.0, 5.0)) # x
        bounds.append((-5.0, 5.0)) # y
    for i in range(N):
        bounds.append((-math.pi, math.pi)) # alpha
    bounds.append((0.0, 2.0 * math.pi)) # theta

    # Run Optimization
    p_opt = x0
    best_L = objective(x0)
    
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, 
                       options={'maxiter': 300, 'ftol': 1e-8})
        
        if res.fun < best_L:
            best_L = res.fun
            p_opt = res.x
    except Exception:
        pass

    # Extract results
    p_final = p_opt[:24].reshape(N, 2)
    alphas_final = p_opt[24:36]
    theta_final = p_opt[36]
    
    # Convert angles to degrees for output
    inner = []
    for i in range(N):
        inner.append([p_final[i, 0], p_final[i, 1], np.degrees(alphas_final[i])])
        
    outer_center = (0.0, 0.0)
    side = best_L
    angle = np.degrees(theta_final)
    
    return inner, outer_center, side, angle

# EVOLVE-BLOCK-END
