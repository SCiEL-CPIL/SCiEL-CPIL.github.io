import numpy as np
from scipy.optimize import minimize
import time

N = 12
SQRT3 = np.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Support function of a unit regular hexagon
    def h_unit(phi):
        phi = phi % (2 * np.pi)
        angles = np.arange(6) * np.pi / 3
        return np.max(np.cos(phi - angles))

    # Outer normals for Theta = 0
    outer_normals = np.array([
        [np.cos(np.radians(30 + 60*k)), np.sin(np.radians(30 + 60*k))] for k in range(6)
    ])
    
    def compute_penalty(vars):
        S = vars[-1]
        if S <= 0:
            return 1e10
        
        centers = vars[:2*N].reshape(N, 2)
        thetas = vars[2*N:3*N]
        
        penalty = 0.0
        
        # 1. Containment Penalty
        for i in range(N):
            c_i = centers[i]
            t_i = thetas[i]
            for k in range(6):
                n = outer_normals[k]
                lhs = np.dot(c_i, n) + h_unit(np.arctan2(n[1], n[0]) - t_i)
                rhs = S * SQRT3 / 2.0
                if lhs > rhs:
                    penalty += (lhs - rhs) ** 2
        
        # 2. Non-Overlap Penalty
        # Use a stricter check: Separating Axis Theorem for hexagons
        # For two convex polygons A and B, they are separated if there exists an axis n
        # such that proj_A(n) + h_A(n) < proj_B(n) + h_B(n) is false?
        # Actually, gap = (c_i - c_j) . n - h_i(n) - h_j(n).
        # If min_gap < 0, they overlap.
        
        for i in range(N):
            for j in range(i + 1, N):
                c_i = centers[i]
                c_j = centers[j]
                t_i = thetas[i]
                t_j = thetas[j]
                
                diff = c_i - c_j
                
                # Normals for i
                normals_i = np.array([
                    [np.cos(t_i + np.radians(30 + 60*l)), np.sin(t_i + np.radians(30 + 60*l))] 
                    for l in range(6)
                ])
                # Normals for j
                normals_j = np.array([
                    [np.cos(t_j + np.radians(30 + 60*l)), np.sin(t_j + np.radians(30 + 60*l))] 
                    for l in range(6)
                ])
                
                all_normals = np.vstack([normals_i, normals_j])
                
                min_gap = np.inf
                for n in all_normals:
                    angle_n = np.arctan2(n[1], n[0])
                    h_i_val = h_unit(angle_n - t_i)
                    h_j_val = h_unit(angle_n - t_j)
                    
                    gap = np.dot(diff, n) - h_i_val - h_j_val
                    if gap < min_gap:
                        min_gap = gap
                
                if min_gap < 0:
                    penalty += (min_gap) ** 2
        
        return penalty

    def is_valid(vars):
        penalty = compute_penalty(vars)
        return penalty < 1e-6 and vars[-1] > 0

    def objective(vars):
        if time.time() > deadline:
            return 1e10
        penalty = compute_penalty(vars)
        S = vars[-1]
        # High penalty weight to force validity
        return S + 1000.0 * penalty

    # --- Step 1: Generate a Guaranteed Valid Initial Configuration ---
    # 3 rows of 4 hexagons, staggered.
    # Spacing: x_step = SQRT3 * 1.01, y_step = 1.5 * 1.01
    # This ensures no overlap.
    spacing_x = SQRT3 * 1.02
    spacing_y = 1.5 * 1.02
    centers_init = []
    for r in range(3):
        for c in range(4):
            x = c * spacing_x
            y = r * spacing_y
            if r % 2 == 1:
                x += spacing_x / 2.0
            centers_init.append([x, y])
    centers_init = np.array(centers_init)
    centers_init -= centers_init.mean(axis=0)
    
    thetas_init = np.zeros(N)
    
    # Find minimal S for this fixed cluster (rotation 0 for outer)
    # We can also try rotating the outer hexagon to minimize S
    best_S_init = np.inf
    best_theta_outer = 0.0
    
    # Grid search for optimal outer rotation for this fixed cluster
    for theta_deg in np.arange(0, 60, 5):
        theta = np.radians(theta_deg)
        # Rotate outer normals
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        rotated_normals = outer_normals @ R.T
        
        max_req = 0
        for i in range(N):
            c_i = centers_init[i]
            t_i = thetas_init[i]
            for k in range(6):
                n = rotated_normals[k]
                lhs = np.dot(c_i, n) + h_unit(np.arctan2(n[1], n[0]) - t_i)
                req_s = lhs / (SQRT3 / 2.0)
                if req_s > max_req:
                    max_req = req_s
        
        if max_req < best_S_init:
            best_S_init = max_req
            best_theta_outer = theta_deg

    # Add small margin for safety
    S_init = best_S_init * 1.01
    vars_init = np.concatenate([centers_init.flatten(), thetas_init, [S_init]])
    
    # Verify this initial config is valid (it should be)
    if not is_valid(vars_init):
        # If somehow invalid (shouldn't be), increase S significantly
        S_init = 10.0
        vars_init[-1] = S_init

    best_vars = vars_init.copy()
    best_score = S_init

    # --- Step 2: Optimize ---
    # We will try to optimize the positions, rotations, and S.
    # Note: The outer hexagon rotation is fixed to 0 in the variable vector?
    # The interface asks for outer_angle_deg. 
    # In the previous code, outer angle was returned as 0.0.
    # If we allow outer rotation, we need to add it to vars.
    # Let's add outer rotation to vars: [x1..y12, t1..t12, S, theta_outer]
    
    def objective_full(vars):
        if time.time() > deadline:
            return 1e10
        S = vars[-2]
        theta_outer = vars[-1]
        if S <= 0:
            return 1e10
        
        centers = vars[:2*N].reshape(N, 2)
        thetas = vars[2*N:3*N]
        
        penalty = 0.0
        
        # Rotate outer normals by theta_outer
        R = np.array([[np.cos(theta_outer), -np.sin(theta_outer)], [np.sin(theta_outer), np.cos(theta_outer)]])
        rotated_normals = outer_normals @ R.T
        
        # 1. Containment
        for i in range(N):
            c_i = centers[i]
            t_i = thetas[i]
            for k in range(6):
                n = rotated_normals[k]
                lhs = np.dot(c_i, n) + h_unit(np.arctan2(n[1], n[0]) - t_i)
                rhs = S * SQRT3 / 2.0
                if lhs > rhs:
                    penalty += (lhs - rhs) ** 2
        
        # 2. Non-Overlap (same as before)
        for i in range(N):
            for j in range(i + 1, N):
                c_i = centers[i]
                c_j = centers[j]
                t_i = thetas[i]
                t_j = thetas[j]
                
                diff = c_i - c_j
                
                normals_i = np.array([
                    [np.cos(t_i + np.radians(30 + 60*l)), np.sin(t_i + np.radians(30 + 60*l))] 
                    for l in range(6)
                ])
                normals_j = np.array([
                    [np.cos(t_j + np.radians(30 + 60*l)), np.sin(t_j + np.radians(30 + 60*l))] 
                    for l in range(6)
                ])
                
                all_normals = np.vstack([normals_i, normals_j])
                
                min_gap = np.inf
                for n in all_normals:
                    angle_n = np.arctan2(n[1], n[0])
                    h_i_val = h_unit(angle_n - t_i)
                    h_j_val = h_unit(angle_n - t_j)
                    
                    gap = np.dot(diff, n) - h_i_val - h_j_val
                    if gap < min_gap:
                        min_gap = gap
                
                if min_gap < 0:
                    penalty += (min_gap) ** 2
        
        return S + 1000.0 * penalty

    def is_valid_full(vars):
        S = vars[-2]
        if S <= 0:
            return False
        penalty = objective_full(vars) - S
        return penalty < 1e-6

    # Initial vars for full optimization: [centers, thetas, S, theta_outer]
    # theta_outer is in radians for optimization, convert to deg for output
    vars_full_init = np.concatenate([vars_init, [np.radians(best_theta_outer)]])
    
    best_full_vars = vars_full_init.copy()
    best_full_score = best_S_init

    # Run optimization
    try:
        res = minimize(
            objective_full,
            vars_full_init,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8}
        )
        
        if res.success or res.fun < best_full_score:
            if is_valid_full(res.x):
                S_val = res.x[-2]
                if S_val < best_full_score:
                    best_full_score = S_val
                    best_full_vars = res.x.copy()
    except Exception:
        pass

    # Extract best solution
    centers, thetas, S, theta_outer_rad = best_full_vars[:2*N].reshape(N, 2), best_full_vars[2*N:3*N], best_full_vars[-2], best_full_vars[-1]
    
    inner = []
    for i in range(N):
        inner.append([centers[i, 0], centers[i, 1], np.degrees(thetas[i])])
    
    outer_center = (0.0, 0.0)
    outer_side = S
    outer_angle_deg = np.degrees(theta_outer_rad)

    return inner, outer_center, outer_side, outer_angle_deg
