import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    sqrt3 = math.sqrt(3.0)
    
    # --- 1. Initial Configuration (from Inspiration 1) ---
    # Generate 19 centers from radius-2 hex grid
    centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                x = sqrt3 * (q + r / 2.0)
                y = 1.5 * r
                centers.append((x, y))
                
    # Select 12 closest to origin
    centers.sort(key=lambda p: p[0]**2 + p[1]**2)
    selected_centers = centers[:12]
    
    # Scale slightly to ensure strict disjointness initially
    scale = 1.005
    p0_pos = np.array([[x * scale, y * scale] for x, y in selected_centers])
    
    # Initial outer rotation
    theta0 = math.pi / 6 # 30 degrees, as in Inspiration 1
    
    # --- 2. Optimization Setup ---
    # Variables: x_1, y_1, ..., x_12, y_12, theta
    # Total 25 variables
    
    # Vertices of a unit hexagon (relative to center)
    angles = np.array([k * np.pi / 3.0 for k in range(6)])
    hex_verts = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Normals of the outer hexagon (relative to theta)
    # Outer hexagon side L. Inradius R = L * sqrt(3) / 2.
    # Constraint: dot(vertex, normal) <= R
    # We minimize the max dot product. Let M = max dot product.
    # Then L = 2 * M / sqrt(3).
    
    def get_outer_normals(theta):
        # Normals are at theta + pi/6 + k*pi/3
        normal_angles = theta + math.pi/6 + np.arange(6) * math.pi/3
        return np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    def objective(params):
        p_pos = params[:24].reshape(N, 2)
        theta = params[24]
        
        normals = get_outer_normals(theta)
        
        # Calculate max projection of all inner vertices onto outer normals
        # Inner vertices: p_pos[i] + hex_verts[v]
        # We need max over i, v, k of dot(p_pos[i] + hex_verts[v], normals[k])
        
        # Vectorized approach:
        # p_pos: (N, 2)
        # hex_verts: (6, 2)
        # normals: (6, 2)
        
        # For each inner hex i, calculate projections onto all normals
        # proj[i, v, k] = dot(p_pos[i] + hex_verts[v], normals[k])
        
        # To save memory/time, we can compute max over v for each i, k
        # max_v dot(p_pos[i] + hex_verts[v], n_k) = dot(p_pos[i], n_k) + max_v dot(hex_verts[v], n_k)
        
        # max_v dot(hex_verts[v], n_k) is the support function of the unit hexagon in direction n_k.
        # For a unit hexagon, the support function h(n) is the distance from center to the face normal to n.
        # Actually, max dot(v, n) for v in hexagon vertices.
        # Let's precompute this for the 6 normals relative to the hexagon's own frame?
        # No, the inner hexagons are all aligned (angle 0). The outer hexagon rotates.
        # So the relative angle between inner hex and outer normal changes.
        
        # Let's just compute it.
        # proj_base = p_pos @ normals.T  (N, 6)
        # proj_verts = hex_verts @ normals.T  (6, 6)
        # We need max over v of proj_verts[v, k] for each k.
        # max_vert_proj = np.max(proj_verts, axis=0)  (6,)
        
        proj_base = p_pos @ normals.T
        proj_verts = hex_verts @ normals.T
        max_vert_proj = np.max(proj_verts, axis=0)
        
        # Total max projection for each normal k is max over i of (proj_base[i, k] + max_vert_proj[k])
        # Since max_vert_proj[k] is constant for a given theta, 
        # max_i (proj_base[i, k] + C_k) = C_k + max_i proj_base[i, k]
        
        max_proj_per_normal = np.max(proj_base, axis=0) + max_vert_proj
        max_projection = np.max(max_proj_per_normal)
        
        return max_projection

    # Constraints: Non-overlap
    # dist(p_i, p_j) >= sqrt(3)
    eps = 1e-5
    min_dist_sq = (sqrt3 + eps)**2
    
    constraints = []
    for i in range(N):
        for j in range(i+1, N):
            def make_constraint(i, j):
                def con(params):
                    pi = params[2*i:2*i+2]
                    pj = params[2*j:2*j+2]
                    diff = pi - pj
                    return min_dist_sq - np.dot(diff, diff)
                return con
            constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})
            
    # Bounds
    # Positions should be within a reasonable range, say [-5, 5]
    # Theta in [0, pi/3)
    bounds = [(-5.0, 5.0)] * 24 + [(0.0, math.pi/3)]
    
    p0 = np.zeros(25)
    p0[:24] = p0_pos.flatten()
    p0[24] = theta0
    
    best_obj = objective(p0)
    best_params = p0
    
    # Try a few different initial theta values to escape local minima
    # The objective function is periodic with period pi/3
    # Let's try optimizing from a few different starting angles
    
    start_angles = [0.0, math.pi/12, math.pi/6, math.pi/4, math.pi/3 - 1e-6]
    
    for angle in start_angles:
        if time.time() > deadline - 20:
            break
            
        p_current = p0.copy()
        p_current[24] = angle
        
        try:
            res = minimize(objective, p_current, method='SLSQP', bounds=bounds, 
                           constraints=constraints, options={'maxiter': 300, 'ftol': 1e-10})
            
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x
        except Exception:
            pass
            
        # Also try Nelder-Mead for non-smooth robustness if SLSQP fails or improves little
        if time.time() > deadline - 10:
            break
            
        try:
            res_nm = minimize(objective, best_params, method='Nelder-Mead', 
                              options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res_nm.fun < best_obj:
                best_obj = res_nm.fun
                best_params = res_nm.x
        except Exception:
            pass

    # Extract results
    p_opt = best_params[:24].reshape(N, 2)
    theta_opt = best_params[24]
    
    # Calculate final side length
    # L = 2 * M / sqrt(3)
    side = 2.0 * best_obj / sqrt3
    
    # Add tiny safety margin
    side *= 1.001
    
    # Construct inner list
    # All inner hexagons are aligned (angle 0) in this model
    inner = []
    for i in range(N):
        inner.append([p_opt[i, 0], p_opt[i, 1], 0.0])
        
    outer_center = (0.0, 0.0)
    # The outer hexagon is rotated by theta_opt
    # The interface asks for "outer hexagon ... rotation". 
    # Usually, this means the angle of the outer hexagon.
    # In our objective, we defined normals based on theta.
    # A hexagon with normals at theta + pi/6 has vertices at theta + pi/6 + pi/2 = theta + 2pi/3?
    # Let's check.
    # Flat-top hexagon (vertex at 0): Normals at pi/6, pi/2, ...
    # If we rotate the hexagon by phi, normals are at phi + pi/6, ...
    # So if normals are at theta + pi/6, the hexagon is rotated by theta.
    angle = theta_opt
    
    return inner, outer_center, side, angle

# EVOLVE-BLOCK-END
